"""
LLM provider adapter — currently supports Google Gemini via langchain-google-genai.

Design goals:
- Keep the rest of the app LLM-provider-agnostic.
- Handle common errors gracefully (rate limit, auth, quota).
- Easy to swap provider by adding a new factory in get_llm().
- Never log API keys.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM invocation fails after retries."""


def get_llm(provider: str = "gemini", model: str | None = None, **kwargs: Any) -> BaseChatModel:
    """
    Return a LangChain chat model for the given provider.

    Args:
        provider: "gemini" (default) | "ollama" | "groq"
        model: Model name override (uses config default if None).
        **kwargs: Additional kwargs passed to the model constructor.

    Raises:
        ValueError: If provider is not supported or API key is missing.
    """
    from rag_app.core.config import GEMINI_MODEL, GOOGLE_API_KEY  # noqa: PLC0415

    if provider == "gemini":
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY is not configured. "
                "Get a free key at https://aistudio.google.com/apikey and set it in .env"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("langchain-google-genai is required: pip install langchain-google-genai") from exc

        target_model = model or GEMINI_MODEL
        logger.info("Initializing Gemini LLM: model=%s", target_model)
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=GOOGLE_API_KEY,
            temperature=kwargs.get("temperature", 0.0),
            max_retries=kwargs.get("max_retries", 3),
            **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_retries")},
        )

    if provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("langchain-community is required for Ollama: pip install langchain-community") from exc
        target_model = model or "llama3.2"
        logger.info("Initializing Ollama LLM: model=%s", target_model)
        return ChatOllama(model=target_model, **kwargs)

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("langchain-groq is required: pip install langchain-groq") from exc
        import os
        groq_key = os.environ.get("GROQ_API_KEY", "")
        target_model = model or "llama3-8b-8192"
        logger.info("Initializing Groq LLM: model=%s", target_model)
        return ChatGroq(model=target_model, groq_api_key=groq_key, **kwargs)

    raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: gemini, ollama, groq")


def safe_invoke(llm: BaseChatModel, messages: list, retries: int = 3, base_delay: float = 2.0) -> str:
    """
    Invoke the LLM with exponential back-off retry on rate-limit errors.

    Returns the text content of the response.
    Raises LLMError on persistent failure.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = llm.invoke(messages)
            content = getattr(response, "content", str(response))
            return content
        except Exception as exc:
            err_str = str(exc).lower()
            last_error = exc
            if "rate" in err_str or "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                wait = base_delay * (2 ** attempt)
                logger.warning("Rate limit hit (attempt %d/%d). Waiting %.1fs…", attempt + 1, retries, wait)
                time.sleep(wait)
            elif "api_key" in err_str or "authentication" in err_str or "permission" in err_str:
                raise LLMError(f"Authentication error — check GOOGLE_API_KEY: {exc}") from exc
            else:
                logger.error("LLM error (attempt %d/%d): %s", attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep(base_delay)
    raise LLMError(f"LLM call failed after {retries} attempts: {last_error}") from last_error
