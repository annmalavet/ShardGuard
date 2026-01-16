from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Protocol

import google.generativeai as genai  # type: ignore
import httpx

logger = logging.getLogger(__name__)


class ToolArgsLLM(Protocol):
    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_response_sync(self, prompt: str) -> str:
        """Generate a response from the LLM (synchronous version)."""
        pass

    @abstractmethod
    def close(self):
        """Close any open connections."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models."""

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the HTTP client."""
        try:
            self.client = httpx.Client(timeout=300.0)
        except ImportError:
            logger.warning(
                "httpx not available. Ollama provider will use mock responses."
            )

    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Ollama."""
        if not self.client:
            return self._mock_response(prompt)

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 2048,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return self._mock_response(prompt, error=str(e))

    def generate_response_sync(self, prompt: str) -> str:
        """Generate a response using Ollama (synchronous)."""
        if not self.client:
            return self._mock_response(prompt)

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 2048,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return self._mock_response(prompt, error=str(e))

    def _mock_response(self, prompt: str, error: str | None = None) -> str:
        """Generate a mock response for testing or fallback."""
        content = (
            f"Error occurred: {error}"
            if error
            else "This is a mock response - httpx not available"
        )
        return json.dumps(
            {
                "original_prompt": prompt,
                "sub_prompts": [
                    {
                        "id": 1,
                        "content": content,
                        "opaque_values": {},
                        "suggested_tools": ["file-operations.read_file"],
                    }
                ],
            }
        )

    def close(self):
        """Close the HTTP client."""
        if self.client:
            self.client.close()


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider for remote models."""

    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Gemini client."""
        if not self.api_key:
            logger.warning(
                "No Gemini API key provided. Provider will use mock responses."
            )
            return

        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Gemini client with model: {self.model}")
        except ImportError:
            logger.warning(
                "google-generativeai not available. Gemini provider will use mock responses."
            )
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")

    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Gemini."""
        if not self.client or not self.api_key:
            return self._mock_response(prompt)

        try:
            # Gemini API is synchronous, so we'll run it as is
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_output_tokens": 2048,
                },
            )
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return self._mock_response(prompt, error=str(e))

    def generate_response_sync(self, prompt: str) -> str:
        """Generate a response using Gemini (synchronous)."""
        if not self.client or not self.api_key:
            return self._mock_response(prompt)

        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_output_tokens": 2048,
                },
            )
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return self._mock_response(prompt, error=str(e))

    def _mock_response(self, prompt: str, error: str | None = None) -> str:
        """Generate a mock response for testing or fallback."""
        content = (
            f"Error occurred: {error}"
            if error
            else "This is a mock response - Gemini API not available"
        )
        return json.dumps(
            {
                "original_prompt": prompt,
                "sub_prompts": [
                    {
                        "id": 1,
                        "content": content,
                        "opaque_values": {},
                        "suggested_tools": ["file-operations.read_file"],
                    }
                ],
            }
        )

    def close(self):
        """Close any connections (Gemini doesn't require explicit closing)."""
        pass


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(provider_type: str, model: str, **kwargs) -> LLMProvider:
        """Create an LLM provider based on the provider type."""
        if provider_type.lower() == "ollama":
            base_url = kwargs.get("base_url", "http://localhost:11434")
            return OllamaProvider(model=model, base_url=base_url)
        elif provider_type.lower() == "gemini":
            api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
            return GeminiProvider(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
