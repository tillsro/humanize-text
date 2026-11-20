"""
Standard Pipeline v1.5.1 — Multi-language translation chain with LLM humanization.

Pipeline (4 steps):
  Step 1: Input (EN) → Chinese — DeepSeek humanization rewrite
  Step 2: Chinese → Japanese — DeepSeek humanization rewrite (with history)
  Step 3: Japanese → Finnish — Google Translate or local Ollama translation
  Step 4: Finnish → Target (EN) — Niutrans or local Ollama translation

This chain was selected after empirical testing against AI detectors on
50+ sample texts. See `examples/showcase/` for input/output traces of all
4 intermediate steps on 5 real samples.
"""

import json
import time
import click
import toml

from .llm_client import resolve_llm_config
from .translators import LANGUAGE_NAMES, google_translate, niutrans_translate, ollama_translate
from .llm_rewriter import llm_rewrite


def _validate_pipeline_config(config: dict, target_lang: str = "en") -> None:
    """Pre-flight check: validate required config before any API calls.

    Steps 1-2 spend real LLM credits, so a missing key that only surfaces at
    Step 4 wastes money. Collect every problem and raise them together so the
    operator can fix everything in one pass.
    """
    errors = []

    translation = config.get("translation", {})
    translation_provider = translation.get("provider", "google_niutrans")
    if translation_provider not in {"google_niutrans", "ollama"}:
        errors.append("translation.provider must be 'google_niutrans' or 'ollama'")

    niutrans_key = config.get("api_keys", {}).get("niutrans_api_key", "")
    if translation_provider == "google_niutrans" and not niutrans_key:
        errors.append(
            "api_keys.niutrans_api_key is empty (required for Step 4 — "
            "get a key at niutrans.com)"
        )

    intermediate = config.get("pipeline", {}).get("intermediate_lang", "fi")
    known_codes = LANGUAGE_NAMES
    if intermediate not in known_codes:
        errors.append(
            f"pipeline.intermediate_lang={intermediate!r} is not a recognized "
            f"language code (known: {', '.join(sorted(known_codes))})"
        )

    if translation_provider == "ollama":
        if target_lang not in known_codes:
            errors.append(f"Unsupported local target language: {target_lang!r}")
        max_chars = translation.get("max_chunk_chars", 1000)
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            errors.append("translation.max_chunk_chars must be a positive integer")
        timeout = translation.get("timeout", 600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            errors.append("translation.timeout must be a positive number of seconds")
        for field in ("model", "base_url"):
            if field in translation and (
                not isinstance(translation[field], str) or not translation[field].strip()
            ):
                errors.append(f"translation.{field} must be a nonempty string")

    if errors:
        raise ValueError(
            "Pipeline config validation failed:\n  - " + "\n  - ".join(errors)
        )


def run_standard_pipeline(text: str, config: dict, target_lang: str = "en") -> dict:
    """Run the Standard humanization pipeline.

    Args:
        text: Input text to humanize.
        config: Configuration dict loaded from config.toml.
        target_lang: Target language code for final output (default: "en").

    Returns:
        dict with keys:
            - 'result': final humanized text
            - 'steps': list of {step, engine, direction, output, length}
            - 'processing_time_ms': total elapsed time in milliseconds
    """
    llm = resolve_llm_config(config)
    _validate_pipeline_config(config, target_lang)
    translation = config.get("translation", {})
    local_translation = translation.get("provider", "google_niutrans") == "ollama"
    # Collect local translation settings before any rewrite requests.
    translation_options = {
        key: translation[key]
        for key in ("model", "base_url", "timeout", "max_chunk_chars")
        if key in translation
    }
    intermediate_lang = config.get("pipeline", {}).get("intermediate_lang", "fi")
    engine_name = llm["display_name"]

    steps = []
    start = time.time()

    # Step 1: LLM — Input (EN) → Chinese humanization rewrite
    step1 = llm_rewrite(
        text=text,
        target_language="中文",
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        model=llm["model"],
        history=None,
        temperature=llm["temperature"],
        extra_headers=llm["extra_headers"],
        provider=llm["provider"],
        timeout=llm["timeout"],
    )
    steps.append({
        "step": 1, "engine": engine_name,
        "direction": "Input → Chinese (中文改写)",
        "output": step1, "length": len(step1),
    })

    # Step 2: LLM — Chinese → Japanese (carries step 1 as history)
    step2 = llm_rewrite(
        text=step1,
        target_language="日语",
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        model=llm["model"],
        history={"input": text, "output": step1},
        temperature=llm["temperature"],
        extra_headers=llm["extra_headers"],
        provider=llm["provider"],
        timeout=llm["timeout"],
    )
    steps.append({
        "step": 2, "engine": engine_name,
        "direction": "Chinese → Japanese (日语改写)",
        "output": step2, "length": len(step2),
    })

    # Step 3: Japanese → intermediate language (first translation hop)
    if local_translation:
        step3 = ollama_translate(step2, source="ja", target=intermediate_lang, **translation_options)
        translation_engine = f"Ollama ({translation_options.get('model', 'translategemma:27b')})"
    else:
        step3 = google_translate(step2, source="ja", target=intermediate_lang)
        translation_engine = "Google"
    steps.append({
        "step": 3, "engine": translation_engine,
        "direction": f"Japanese → {intermediate_lang.upper()} (一轮翻译)",
        "output": step3, "length": len(step3),
    })

    # Step 4: Intermediate language → target (second translation hop)
    if local_translation:
        step4 = ollama_translate(step3, source=intermediate_lang, target=target_lang, **translation_options)
    else:
        step4 = niutrans_translate(
            step3,
            source=intermediate_lang,
            target=_lang_code_to_niutrans(target_lang),
            api_key=config["api_keys"]["niutrans_api_key"],
        )
        translation_engine = "Niutrans"
    steps.append({
        "step": 4, "engine": translation_engine,
        "direction": f"{intermediate_lang.upper()} → {target_lang.upper()} (二轮翻译)",
        "output": step4, "length": len(step4),
    })

    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "result": step4,
        "steps": steps,
        "processing_time_ms": elapsed_ms,
    }


def _lang_code_to_niutrans(code: str) -> str:
    """Map common language codes to Niutrans format."""
    mapping = {
        "en": "en", "zh": "zh", "ja": "ja", "ko": "ko",
        "fr": "fr", "de": "de", "es": "es", "pt": "pt",
        "ru": "ru", "ar": "ar", "it": "it", "nl": "nl",
        "fi": "fi",
    }
    return mapping.get(code, code)


@click.command()
@click.option("--input", "input_text", required=True, help="Input text or path to text file")
@click.option("--target", default="en", help="Target language code (default: en)")
@click.option("--config", default="config/config.toml", help="Config file path")
@click.option("--output", default=None, help="Output file path")
@click.option("--verbose", is_flag=True, help="Show step-by-step progress")
@click.option("--json", "json_output", is_flag=True, help="Output the full result (with step traces) as JSON")
def main(input_text, target, config, output, verbose, json_output):
    """Run the Standard humanization pipeline."""
    import os

    if os.path.isfile(input_text):
        with open(input_text, "r", encoding="utf-8") as f:
            input_text = f.read()

    cfg = toml.load(config)
    result = run_standard_pipeline(input_text, cfg, target_lang=target)

    if json_output:
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(payload)
            click.echo(f"Written to {output}")
        else:
            click.echo(payload)
        return

    if verbose:
        click.echo("\n--- Pipeline Steps ---")
        for s in result["steps"]:
            click.echo(f"  Step {s['step']}: {s['engine']} | {s['direction']} | {s['length']} chars")
        click.echo(f"  Total: {result['processing_time_ms']}ms\n")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(result["result"])
        click.echo(f"Written to {output}")
    else:
        click.echo(result["result"])


if __name__ == "__main__":
    main()
