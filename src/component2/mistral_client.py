"""
mistral_client.py
=================
Handles all communication with Ollama (local Mistral 7B).
Works both locally and via ngrok tunnel.

Usage:
    from mistral_client import MistralClient
    client = MistralClient()           # local Ollama
    client = MistralClient(ngrok_url)  # via ngrok
    response = client.generate(prompt)
"""

import httpx
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL")


# ── Default config ─────────────────────────────────────────────────
MODEL_NAME       = "mistral:7b"

# Timeout in seconds — Mistral 7B on RTX 4060 takes ~15-30s per response
TIMEOUT_SECONDS  = 120


class MistralClient:
    """
    Client for Ollama-hosted Mistral 7B.
    Automatically falls back to local if ngrok URL not provided.
    """

    def __init__(self, base_url: str = None, model: str = MODEL_NAME):
        self.base_url = (base_url or LOCAL_OLLAMA_URL).rstrip('/')
        self.model    = model
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is reachable before proceeding."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = [m['name'] for m in resp.json().get('models', [])]
            if self.model not in models and not any(self.model.split(':')[0] in m for m in models):
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. Run: ollama pull {self.model}"
                )
            print(f"✅ Ollama connected at {self.base_url} | model: {self.model}")
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: ollama serve"
            )

    def generate(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        """
        Send a prompt to Mistral and return the response text.
        temperature=0.2 keeps outputs deterministic and clinical.
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p":       0.9,
                "repeat_penalty": 1.1,
            }
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=TIMEOUT_SECONDS
            )
            resp.raise_for_status()
            return resp.json().get('response', '').strip()

        except httpx.TimeoutException:
            raise RuntimeError(
                f"Mistral timed out after {TIMEOUT_SECONDS}s. "
                "Try reducing max_tokens or check GPU availability."
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} — {e.response.text}")

    def generate_json(self, prompt: str, temperature: float = 0.1) -> dict:
        """
        Generate a response and parse it as JSON.
        Prompts must explicitly ask for JSON output.
        """
        response = self.generate(prompt, temperature=temperature, max_tokens=512)

        # Strip markdown code fences if present
        clean = response.strip()
        if clean.startswith('```'):
            lines = clean.split('\n')
            clean = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON object from response
            import re
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            raise ValueError(f"Could not parse JSON from Mistral response:\n{response[:300]}")

    def is_available(self) -> bool:
        """Non-raising check for availability."""
        try:
            httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return True
        except:
            return False


def get_client(ngrok_url: str = None) -> MistralClient:
    """
    Factory function — returns a connected MistralClient.
    Tries ngrok URL first if provided, falls back to local.
    """
    if ngrok_url:
        try:
            client = MistralClient(base_url=ngrok_url)
            return client
        except RuntimeError as e:
            print(f"⚠️  ngrok connection failed: {e}")
            print("Falling back to local Ollama…")

    return MistralClient(base_url=LOCAL_OLLAMA_URL)


if __name__ == '__main__':
    import sys
    url    = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(url)

    print("\nTesting with a simple medical prompt…")
    resp = client.generate(
        "In one sentence, what does a haemoglobin of 7.5 g/dL indicate in a 45-year-old female?",
        max_tokens=100
    )
    print(f"Response: {resp}")