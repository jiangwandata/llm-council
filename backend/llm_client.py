"""Generic LLM client that supports multiple providers."""

from typing import Any, Dict, List, Optional
import httpx

from .config import (
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
)


async def _query_openrouter_model(
    model: str, messages: List[Dict[str, str]], timeout: float
) -> Optional[Dict[str, Any]]:
    """Query a model via the OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        return {
            "content": message.get("content"),
            "reasoning_details": message.get("reasoning_details"),
        }


async def _query_deepseek_model(
    model: str, messages: List[Dict[str, str]], timeout: float
) -> Optional[Dict[str, Any]]:
    """Query a model via the DeepSeek API."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        return {
            "content": message.get("content"),
            "reasoning_details": message.get("reasoning_details"),
        }


async def query_model(
    model: str, messages: List[Dict[str, str]], timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model using the configured provider.

    Args:
        model: Model identifier for the selected provider
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    try:
        if LLM_PROVIDER == "deepseek":
            return await _query_deepseek_model(model, messages, timeout)
        else:
            return await _query_openrouter_model(model, messages, timeout)
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str], messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)

    return {model: response for model, response in zip(models, responses)}
