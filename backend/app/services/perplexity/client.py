"""Perplexity AI API client for stock research."""

import json
import os
from typing import Any, Literal, TypeVar

from perplexity import AsyncPerplexity, Perplexity
from pydantic import BaseModel

# Type-safe model selection
PerplexityModel = Literal[
    "sonar",  # Quick queries, cheapest ($1/$1 per 1M) - use for Stage 2
    "sonar-pro",  # Advanced search ($3/$15 per 1M) - complex queries
    "sonar-reasoning-pro",  # Enhanced reasoning ($2/$8 per 1M) - use for Stage 4
    "sonar-deep-research",  # Exhaustive research with citations ($2/$8 per 1M + $2/1M citations)
]

# Search recency filter options
SearchRecency = Literal["hour", "day", "week", "month", "year"]

T = TypeVar("T", bound=BaseModel)


class PerplexityClient:
    """Client for interacting with Perplexity AI API.

    Supports both synchronous and asynchronous operations with structured
    output using JSON schemas.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Perplexity client.

        Args:
            api_key: Perplexity API key. If not provided, uses PERPLEXITY_API_KEY env var.
        """
        self._api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self._api_key:
            raise ValueError("Perplexity API key is required. Set PERPLEXITY_API_KEY env var or pass api_key.")

        self._sync_client = Perplexity(api_key=self._api_key)
        self._async_client = AsyncPerplexity(api_key=self._api_key)

    def _build_json_schema(self, model_class: type[T]) -> dict[str, Any]:
        """Build a JSON schema from a Pydantic model for structured output.

        Args:
            model_class: Pydantic model class to convert to JSON schema

        Returns:
            JSON schema dict compatible with Perplexity API
        """
        schema = model_class.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "schema": schema,
                "name": model_class.__name__,
                "strict": True,
            },
        }

    def _extract_json_from_response(self, content: str, model: str) -> str:
        """Extract JSON from response, handling <think> tags from reasoning models.

        Based on Perplexity's official documentation:
        https://docs.perplexity.ai/guides/structured-outputs

        The sonar-reasoning-pro and sonar-deep-research models output a <think>
        section containing reasoning tokens, immediately followed by valid JSON.
        The response_format parameter does NOT remove these reasoning tokens.

        Args:
            content: Raw response content from Perplexity
            model: Model name for error messages

        Returns:
            JSON string extracted from the response

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        # Check if response contains <think> tags (reasoning models)
        if "<think>" in content:
            # Find the closing </think> tag using rfind (last occurrence)
            marker = "</think>"
            idx = content.rfind(marker)

            if idx == -1:
                # <think> tag opened but not closed - response was truncated
                raise ValueError(
                    f"Model '{model}' response was truncated during thinking phase. "
                    f"Try increasing max_tokens. "
                    f"Response preview: {content[:300]}..."
                )

            # Extract the substring after the marker
            json_str = content[idx + len(marker) :].strip()

            if not json_str:
                raise ValueError(
                    f"Model '{model}' returned thinking process but no JSON output. "
                    f"The response may have been truncated. "
                    f"Thinking preview: {content[:300]}..."
                )

            # Remove markdown code fence markers if present (per Perplexity docs)
            if json_str.startswith("```json"):
                json_str = json_str[len("```json") :].strip()
            if json_str.startswith("```"):
                json_str = json_str[3:].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-3].strip()

            return json_str

        # No think tags - return content as-is
        return content.strip()

    def chat(
        self,
        prompt: str,
        model: PerplexityModel = "sonar-pro",
        system_message: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request (synchronous).

        Args:
            prompt: User prompt/question
            model: Perplexity model to use
            system_message: Optional system message for context
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Response content as string
        """
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = self._sync_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        return str(content) if content else ""

    async def achat(
        self,
        prompt: str,
        model: PerplexityModel = "sonar-pro",
        system_message: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request (asynchronous).

        Args:
            prompt: User prompt/question
            model: Perplexity model to use
            system_message: Optional system message for context
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Response content as string
        """
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        return str(content) if content else ""

    def chat_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: PerplexityModel = "sonar-pro",
        system_message: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        search_recency_filter: SearchRecency | None = None,
        search_domain_filter: list[str] | None = None,
        search_context_size: Literal["low", "medium", "high"] | None = None,
        disable_search: bool = False,
    ) -> T:
        """Send a chat request with structured JSON output (synchronous).

        Args:
            prompt: User prompt/question
            response_model: Pydantic model class for response parsing
            model: Perplexity model to use
            system_message: Optional system message for context
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens in response
            search_recency_filter: Limit results to recent content (hour/day/week/month/year)
            search_domain_filter: Limit search to specific domains (e.g., ["twitter.com", "reddit.com"])
            search_context_size: Control search depth (low/medium/high)
            disable_search: If True, skip web search (for synthesis-only tasks)

        Returns:
            Parsed Pydantic model instance
        """
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response_format = self._build_json_schema(response_model)

        # Build optional parameters
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }

        if search_recency_filter:
            kwargs["search_recency_filter"] = search_recency_filter

        if search_domain_filter:
            kwargs["search_domain_filter"] = search_domain_filter

        if search_context_size:
            kwargs["web_search_options"] = {"search_context_size": search_context_size}

        if disable_search:
            kwargs["disable_search"] = True

        response = self._sync_client.chat.completions.create(**kwargs)

        raw_content = response.choices[0].message.content or ""

        # Handle empty response
        if not raw_content.strip():
            raise ValueError(
                f"Empty response from Perplexity model '{model}'. The model may not support structured output."
            )

        # Extract JSON from response (handles <think> tags from reasoning models)
        json_content = self._extract_json_from_response(raw_content, model)

        # Try to parse JSON, with helpful error message
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from Perplexity response. "
                f"Model: {model}, Error: {e}. "
                f"Extracted content: {json_content[:500]}... "
                f"Raw response preview: {raw_content[:300]}..."
            ) from e

        return response_model.model_validate(data)

    async def achat_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: PerplexityModel = "sonar-pro",
        system_message: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        search_recency_filter: SearchRecency | None = None,
        search_domain_filter: list[str] | None = None,
        search_context_size: Literal["low", "medium", "high"] | None = None,
        disable_search: bool = False,
    ) -> T:
        """Send a chat request with structured JSON output (asynchronous).

        Args:
            prompt: User prompt/question
            response_model: Pydantic model class for response parsing
            model: Perplexity model to use
            system_message: Optional system message for context
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens in response
            search_recency_filter: Limit results to recent content (hour/day/week/month/year)
            search_domain_filter: Limit search to specific domains (e.g., ["twitter.com", "reddit.com"])
            search_context_size: Control search depth (low/medium/high)
            disable_search: If True, skip web search (for synthesis-only tasks)

        Returns:
            Parsed Pydantic model instance
        """
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response_format = self._build_json_schema(response_model)

        # Build optional parameters
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }

        if search_recency_filter:
            kwargs["search_recency_filter"] = search_recency_filter

        if search_domain_filter:
            kwargs["search_domain_filter"] = search_domain_filter

        if search_context_size:
            kwargs["web_search_options"] = {"search_context_size": search_context_size}

        if disable_search:
            kwargs["disable_search"] = True

        response = await self._async_client.chat.completions.create(**kwargs)

        raw_content = response.choices[0].message.content or ""

        # Handle empty response
        if not raw_content.strip():
            raise ValueError(
                f"Empty response from Perplexity model '{model}'. The model may not support structured output."
            )

        # Extract JSON from response (handles <think> tags from reasoning models)
        json_content = self._extract_json_from_response(raw_content, model)

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from Perplexity response. "
                f"Model: {model}, Error: {e}. "
                f"Response preview: {json_content[:500]}..."
            ) from e

        return response_model.model_validate(data)


# Singleton instance
_perplexity_client: PerplexityClient | None = None


def get_perplexity_client() -> PerplexityClient:
    """Get or create the singleton Perplexity client instance."""
    global _perplexity_client
    if _perplexity_client is None:
        _perplexity_client = PerplexityClient()
    return _perplexity_client
