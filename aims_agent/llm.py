import os
import time

from dotenv import load_dotenv

load_dotenv()

# OpenAI (api.openai.com) — when OPENAI_API_KEY is set (takes precedence)
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# OpenRouter — when only OPENROUTER_API_KEY is set
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

RETRY_CODES = (429, 502, 503, 504)
MAX_RETRIES = 3
RETRY_DELAY = 5


def _is_retriable(completion) -> bool:
    if not hasattr(completion, "error") or not completion.error:
        return False
    err = completion.error
    if isinstance(err, dict) and err.get("code") in RETRY_CODES:
        return True
    return False


def _http_retry_status(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            c = err.get("code")
            if isinstance(c, int):
                return c
    return None


def LMF_LLM(prompt: str) -> str:
    """
    Chat completion via the OpenAI Python SDK.

    Do not import ``openai`` at module level: importing ``aims_agent`` (e.g. in
    tests) must work even when ``openai`` is not installed until a call is made.

    Providers (first match wins):
    - OPENAI_API_KEY → official OpenAI (OPENAI_MODEL, default gpt-4o-mini)
    - OPENROUTER_API_KEY → OpenRouter (OPENROUTER_MODEL, default gemma free tier)
    """
    try:
        from openai import APIStatusError, OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' package is required for LMF_LLM. Install: pip install openai"
        ) from e

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    router_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()

    if openai_key:
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        client = OpenAI(api_key=openai_key)
        provider_label = "OpenAI"
    elif router_key:
        model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=router_key)
        provider_label = "OpenRouter"
    else:
        raise RuntimeError(
            "Set OPENAI_API_KEY (OpenAI) or OPENROUTER_API_KEY (OpenRouter) in .env"
        )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        except APIStatusError as e:
            status = _http_retry_status(e)
            if status in RETRY_CODES and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(
                    f"[LLM] HTTP {status}, retry in {wait}s ({attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
                last_error = e
                continue
            raise RuntimeError(f"LLM call failed: {e}") from e
        if completion.choices:
            content = completion.choices[0].message.content
            return content if content is not None else ""
        if _is_retriable(completion):
            last_error = getattr(completion, "error", None)
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(
                    f"[LLM] empty response / gateway error, after {wait}s retry ({attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
                continue
        msg = f"{provider_label} returned no choices (empty or rate-limited?)."
        if hasattr(completion, "error") and completion.error:
            msg += f" API error: {completion.error}"
        raise RuntimeError(msg)

    raise RuntimeError(last_error or "LLM call failed after retries")
