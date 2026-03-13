import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
RETRY_CODES = (502, 503, 504)
MAX_RETRIES = 3
RETRY_DELAY = 5


def _is_retriable(completion) -> bool:
    if not hasattr(completion, "error") or not completion.error:
        return False
    err = completion.error
    if isinstance(err, dict) and err.get("code") in RETRY_CODES:
        return True
    return False


def LMF_LLM(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        if completion.choices:
            content = completion.choices[0].message.content
            return content if content is not None else ""
        if _is_retriable(completion):
            last_error = getattr(completion, "error", None)
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"[LLM] 504/502/503，{wait}s 后重试 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
        msg = "OpenRouter returned no choices (empty or rate-limited?)."
        if hasattr(completion, "error") and completion.error:
            msg += f" API error: {completion.error}"
        raise RuntimeError(msg)

    raise RuntimeError(last_error or "LLM call failed after retries")
