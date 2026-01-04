import os
import time
from typing import Any, Dict, List, Optional, Union
import random
import requests
import logging
from pathlib import Path
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionToolParam, ChatCompletion

from .memory import Memory
from .config import LLMConfig

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "config.json"


class LLMClient:
    """Lightweight LLM client wrapper that works directly with Memory."""

    def __init__(self, config: Union[LLMConfig, Dict[str, Any], str] = str(DEFAULT_CONFIG)):
        self.config = LLMConfig.from_source(config)
        self.deployment_name: str = self.config.deployment_name

        self.config.azure_endpoint = self.config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.config.azure_api_version = self.config.azure_api_version or os.getenv("AZURE_API_VERSION")
        self.config.azure_api_key = self.config.azure_api_key or os.getenv("AZURE_API_KEY")
        self.config.tenant_id = self.config.tenant_id or os.getenv("AZURE_TENANT_ID")
        self.config.api_scope = self.config.api_scope or os.getenv("AZURE_API_SCOPE")

        self.azure_client: Optional[AzureOpenAI] = None

        self._init_client()

    def _init_client(self):
        cfg = self.config

        # 1) try AzureOpenAI with api_key
        if cfg.azure_endpoint and cfg.azure_api_version and cfg.azure_api_key:
            try:
                self.azure_client = AzureOpenAI(
                    azure_endpoint=cfg.azure_endpoint,
                    api_version=cfg.azure_api_version,
                    api_key=cfg.azure_api_key,
                )

                if cfg.log:
                    logger.info("Initialized AzureOpenAI client with api_key (deployment='%s')", self.deployment_name)
                return
            except Exception as e:
                logger.exception("Failed to init AzureOpenAI with api_key (deployment='%s'): %s", self.deployment_name, e)

        # 2) try AzureOpenAI with AzureCliCredential
        if cfg.azure_endpoint and cfg.azure_api_version:
            try:
                tenant_id = cfg.tenant_id or ""
                credential = AzureCliCredential(tenant_id=tenant_id)
                token_provider = get_bearer_token_provider(
                    credential,
                    cfg.api_scope or "https://cognitiveservices.azure.com/.default",
                )

                self.azure_client = AzureOpenAI(
                    azure_endpoint=cfg.azure_endpoint,
                    api_version=cfg.azure_api_version,
                    azure_ad_token_provider=token_provider,
                )

                if cfg.log:
                    logger.info("Initialized AzureOpenAI client with AzureCliCredential (deployment='%s')", self.deployment_name)
                return
            except Exception as e:
                logger.exception("Failed to init AzureOpenAI with AzureCliCredential (deployment='%s'): %s", self.deployment_name, e)

        raise RuntimeError("LLMClient initialization failed: no valid configuration for AzureOpenAI or local endpoint")


    def completion(
        self,
        payload: Union[Memory, List[Dict[str, Any]]],
        tools: Optional[List[ChatCompletionToolParam]] = None,
        max_retries: int = 8,
        retry_delay: float = 20.0,
        raw: bool = False,
        **kwargs: Any,
    ) -> Optional[Union[str, ChatCompletion]]:
        """Call the model with Memory or raw messages and retry."""

        messages = self._resolve_messages(payload)

        retries = 0
        start = time.time()

        while retries < max_retries:
            try:
                result = self._call_azure(messages, tools, raw=raw, **kwargs)

                duration = round(time.time() - start, 2)
                if self.config.log:
                    logger.info(
                        "Model '%s' responded in %.2fs (retry=%d)",
                        self.deployment_name,
                        duration,
                        retries,
                    )

                if result:
                    return result

                retries += 1
                delay = retry_delay + random.uniform(0, 10)
                logger.warning(
                    "Empty result from backend '%s', retrying in %.2fs (retry=%d/%d)",
                    "azure", delay, retries, max_retries,
                )
                time.sleep(delay)

            except Exception as e:
                retries += 1
                logger.exception(
                    "Error calling backend '%s' (retry=%d/%d): %s",
                    "azure", retries, max_retries, e,
                )
                if retries >= max_retries:
                    logger.error("Maximum retries reached for model '%s'. Aborting.", self.deployment_name)
                    return None

                delay = retry_delay + random.uniform(0, 10)
                logger.warning(
                    "Retrying backend '%s' in %.2fs after error",
                    "azure", delay,
                )
                time.sleep(delay)

        return None

    def _call_azure(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ChatCompletionToolParam]] = None,
        raw: bool = False,
        **kwargs: Any,
    ) -> Optional[Union[str, ChatCompletion]]:
        if self.azure_client is None:
            raise RuntimeError("AzureOpenAI client not initialized")

        try:
            reserved = {"messages", "model", "tools"}
            conflicts = reserved.intersection(kwargs)
            if conflicts:
                raise ValueError(f"Reserved completion args passed: {sorted(conflicts)}")

            request = {
                "messages": messages,
                "model": self.deployment_name,
                "tools": tools,
                "max_completion_tokens": self.config.max_tokens,
                # "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": self.config.stream,
                "stop": self.config.stop,
            }
            request.update(kwargs)

            completion = self.azure_client.chat.completions.create(**request)

            if tools or raw:
                return completion

            msg = completion.choices[0].message
            return (msg.content or "").strip() if msg and msg.content else None
        except Exception as e:
            logger.exception("AzureOpenAI call failed: %s", e)
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMClient":
        config = LLMConfig.from_dict(data.get("config", {}))
        return cls(config=config)

    def __repr__(self):
        return f"<LLMClient deployment='{self.deployment_name}'>"

    def _resolve_messages(self, payload: Union[Memory, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(payload, Memory):
            return payload.history
        if isinstance(payload, list):
            return payload
        raise TypeError("payload must be a Memory instance or a list of messages")
