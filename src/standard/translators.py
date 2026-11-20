"""Translation engines: Google Translate, Niutrans, and local Ollama models."""

import httpx

from .retry import retry_with_backoff
from .llm_client import OLLAMA_BASE_URL, chat_completions
from deep_translator import GoogleTranslator

LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "it": "Italian", "nl": "Dutch",
    "fi": "Finnish",
}


def ollama_translate(
    text: str,
    source: str,
    target: str,
    *,
    model: str = "translategemma:27b",
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 600,
    max_chunk_chars: int = 1000,
    prompt_style: str = "plain",
) -> str:
    """Translate locally using one user message per bounded text chunk.

    Each model receives translation-only instructions without rewrite history
    or a separate system message. TranslateGemma has its own prompt profile.
    """
    if source not in LANGUAGE_NAMES or target not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported local translation language: {source!r} -> {target!r}")
    if isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int) or max_chunk_chars < 1:
        raise ValueError("translation.max_chunk_chars must be a positive integer.")
    if prompt_style not in {"plain", "translategemma"}:
        raise ValueError("translation.prompt_style must be 'plain' or 'translategemma'.")
    if not text.strip():
        return text

    source_name, target_name = LANGUAGE_NAMES[source], LANGUAGE_NAMES[target]
    translations = []
    for chunk in _split_local_text(text, max_chunk_chars):
        if not chunk.strip():
            translations.append(chunk)
            continue
        if prompt_style == "translategemma":
            # Use the prompt recommended for TranslateGemma training/inference.
            # https://ollama.com/library/translategemma
            prompt = (
                f"You are a professional {source_name} ({source}) to "
                f"{target_name} ({target}) translator. Your goal is to accurately "
                f"convey the meaning and nuances of the original {source_name} "
                f"text while adhering to {target_name} grammar, vocabulary, and "
                "cultural sensitivities.\n"
                f"Produce only the {target_name} translation, without any "
                "additional explanations or commentary. Please translate the "
                f"following {source_name} text into {target_name}:\n\n\n{chunk}"
            )
        else:
            prompt = (
                f"Translate the passage from {source_name} ({source}) into "
                f"{target_name} ({target}).\n"
                "Keep its meaning, names, numbers, and paragraph structure. "
                "Return the translated passage alone, without commentary:\n\n\n"
                f"{chunk}"
            )
        translations.append(chat_completions(
            [{"role": "user", "content": prompt}],
            api_key="", base_url=base_url, model=model,
            temperature=0.0, timeout=timeout, provider="ollama",
        ))
    return "\n\n".join(translations)


def _split_local_text(text: str, max_len: int) -> list[str]:
    """Bound each chunk, including long sentences without whitespace."""
    chunks = []
    while len(text) > max_len:
        boundary = text.rfind("\n\n", 0, max_len)
        if boundary > 0:
            boundary += 2
        else:
            boundary = text.rfind(" ", 0, max_len) + 1
        boundary = boundary or max_len
        chunks.append(text[:boundary])
        text = text[boundary:]
    if text:
        chunks.append(text)
    return chunks


def google_translate(text: str, source: str, target: str) -> str:
    """Translate text using Google Translate.

    Args:
        text: Text to translate.
        source: Source language code.
        target: Target language code.

    Returns:
        Translated text.
    """
    translator = GoogleTranslator(source=source, target=target)

    # Handle long texts by chunking (Google has ~5000 char limit)
    if len(text) > 4500:
        chunks = _split_text(text, max_len=4500)
        return " ".join(translator.translate(chunk) for chunk in chunks)

    return translator.translate(text)


def niutrans_translate(text: str, source: str, target: str, api_key: str) -> str:
    """Translate text using Niutrans API.

    Args:
        text: Text to translate.
        source: Source language code.
        target: Target language code.
        api_key: Niutrans API key.

    Returns:
        Translated text.
    """
    def _call():
        resp = httpx.post(
            "https://api.niutrans.com/NiuTransServer/translation",
            json={
                "from": source,
                "to": target,
                "apikey": api_key,
                "src_text": text,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    data = retry_with_backoff(_call)

    if "tgt_text" in data:
        return data["tgt_text"]
    elif "error_msg" in data:
        raise RuntimeError(f"Niutrans error: {data['error_msg']}")
    else:
        raise RuntimeError(f"Unexpected Niutrans response: {data}")


def _split_text(text: str, max_len: int = 4500) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_len:
            if current:
                chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks if chunks else [text]
