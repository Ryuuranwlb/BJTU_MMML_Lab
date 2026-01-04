import logging
import argparse
import os

from pangu_agent import LLMConfig, LLMClient, Memory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def test_llm_client():
    client = LLMClient()

    prompt = "Explain the theory of relativity in simple terms."

    response = client.completion(memory=Memory([{"role": "user", "content": prompt}]))
    print(response)


if __name__ == "__main__":
    test_llm_client()
