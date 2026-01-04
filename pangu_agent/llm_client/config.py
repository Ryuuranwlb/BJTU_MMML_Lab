from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json
import yaml
import os


@dataclass
class LLMConfig:
    """Configuration class for LLMClient."""

    # Azure Config
    deployment_name: str = "o3-mini"
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    api_scope: Optional[str] = None

    # LLM parameters
    temperature: float = 0.3
    max_tokens: int = 2000
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None

    log: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_name": self.deployment_name,
            "azure_endpoint": self.azure_endpoint,
            "azure_api_version": self.azure_api_version,
            "azure_api_key": self.azure_api_key,
            "tenant_id": self.tenant_id,
            "api_scope": self.api_scope,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
            "stop": self.stop,
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_source(cls, source: Union[str, Dict[str, Any], "LLMConfig"]) -> "LLMConfig":
        """
        Supports:
        - LLMConfig → return as-is
        - dict → from_dict
        - JSON/YAML string → parse
        - JSON/YAML file path → read then parse
        """
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            return cls.from_dict(source)

        if isinstance(source, str):
            # File path or raw text
            if os.path.exists(source):
                print(f"Loading LLMConfig from file: {source}")
                with open(source, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = source

            # Try JSON
            try:
                return cls.from_dict(json.loads(text))
            except json.JSONDecodeError:
                pass

            # Try YAML
            try:
                data = yaml.safe_load(text)
                if isinstance(data, dict):
                    return cls.from_dict(data)
            except Exception:
                pass

            raise ValueError("Unable to parse config: not valid JSON / YAML / dict / LLMConfig")

        raise TypeError(f"Unsupported config type: {type(source)}")

    def save(self, path: str):
        data = self.to_dict()
        if path.endswith((".yml", ".yaml")):
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
