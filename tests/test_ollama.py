"""Local pipeline contracts, using HTTP simulations rather than a running model."""

import copy
import json
from pathlib import Path

import httpx
import pytest
import toml
from click.testing import CliRunner

from src.standard import llm_client, pipeline, translators


@pytest.fixture(autouse=True)
def clear_llm_overrides(monkeypatch):
    for name in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY",
        "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ATLASCLOUD_API_KEY",
        "ATLAS_CLOUD_API_KEY", "ORCAROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def local_config():
    path = Path(__file__).parent.parent / "config" / "config.ollama.example.toml"
    return toml.load(path)


def _simulate_http(monkeypatch, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)

    def post(url, **kwargs):
        assert kwargs.pop("trust_env") is False
        return client.post(url, **kwargs)

    monkeypatch.setattr(llm_client.httpx, "post", post)
    return client


def _completion(text, finish_reason="stop"):
    return httpx.Response(200, json={"choices": [{
        "message": {"role": "assistant", "content": text},
        "finish_reason": finish_reason,
    }]})


def _forbid_cloud(*args, **kwargs):
    pytest.fail("A cloud translation service was called in local mode")


def test_ollama_needs_no_key_and_ignores_legacy_cloud_model():
    resolved = llm_client.resolve_llm_config({
        "llm": {"provider": "ollama"},
        "pipeline": {"model": "deepseek-chat"},
    })
    assert resolved["base_url"] == "http://localhost:11434/v1"
    assert resolved["model"] == "qwen3.8:27b"
    assert not resolved["api_key"]
    assert resolved["timeout"] == 600


def test_ollama_retains_explicit_model_and_environment_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "custom-local-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://mac.local:11434/v1/")
    resolved = llm_client.resolve_llm_config({"llm": {"model": "qwen2.5:7b"}})
    assert resolved["model"] == "custom-local-model"
    assert resolved["base_url"] == "http://mac.local:11434/v1/"


def test_local_cli_runs_all_four_steps_without_cloud_calls(monkeypatch, local_config, tmp_path):
    """Exercise TOML -> CLI -> real pipeline -> HTTP payloads -> JSON trace."""
    monkeypatch.setattr(pipeline, "google_translate", _forbid_cloud)
    monkeypatch.setattr(pipeline, "niutrans_translate", _forbid_cloud)
    monkeypatch.setenv("HTTP_PROXY", "http://unexpected-proxy.invalid:8080")
    outputs = ["这是测试。", "これはテストです。", "Tämä on testi.", "This is a test."]
    requests = []

    def handler(request):
        assert str(request.url) == "http://localhost:11434/v1/chat/completions"
        assert "authorization" not in request.headers
        assert request.extensions["timeout"]["read"] == 600
        requests.append(json.loads(request.content))
        return _completion(outputs[len(requests) - 1])

    config_path = tmp_path / "local.toml"
    config_path.write_text(toml.dumps(local_config), encoding="utf-8")
    with _simulate_http(monkeypatch, handler):
        result = CliRunner().invoke(pipeline.main, [
            "--input", "A test draft.", "--config", str(config_path), "--json",
        ])
    assert result.exit_code == 0, result.output
    trace = json.loads(result.output)
    assert trace["result"] == outputs[-1]
    assert [step["output"] for step in trace["steps"]] == outputs
    assert all("Ollama" in step["engine"] for step in trace["steps"])
    assert [req["model"] for req in requests] == [
        "qwen3.8:27b", "qwen3.8:27b", "translategemma:27b",
        "hf.co/bartowski/utter-project_EuroLLM-22B-Instruct-2512-GGUF:Q4_K_M",
    ]
    assert trace["steps"][2]["engine"] == "Ollama (translategemma:27b)"
    assert "EuroLLM-22B-Instruct-2512-GGUF:Q4_K_M" in trace["steps"][3]["engine"]
    assert requests[1]["messages"][2] == {"role": "assistant", "content": outputs[0]}
    for request, source, target, text in (
        (requests[2], "Japanese (ja)", "Finnish (fi)", outputs[1]),
        (requests[3], "Finnish (fi)", "English (en)", outputs[2]),
    ):
        assert request["temperature"] == 0.0
        assert request["stream"] is False
        assert len(request["messages"]) == 1
        message = request["messages"][0]
        assert message["role"] == "user"
        assert source in message["content"] and target in message["content"]
        assert message["content"].endswith("\n\n\n" + text)
    assert requests[2]["messages"][0]["content"].startswith(
        "You are a professional Japanese (ja) to Finnish (fi) translator."
    )
    assert requests[3]["messages"][0]["content"].startswith(
        "Translate the passage from Finnish (fi) into English (en)."
    )


def test_default_pipeline_keeps_google_and_niutrans(monkeypatch):
    rewrites = iter(["Chinese", "Japanese"])
    monkeypatch.setattr(pipeline, "llm_rewrite", lambda **kwargs: next(rewrites))
    monkeypatch.setattr(pipeline, "ollama_translate", _forbid_cloud)
    calls = []

    def google(text, source, target):
        calls.append(("Google", text, source, target))
        return "Finnish"

    def niutrans(text, source, target, api_key):
        calls.append(("Niutrans", text, source, target, api_key))
        return "English"

    monkeypatch.setattr(pipeline, "google_translate", google)
    monkeypatch.setattr(pipeline, "niutrans_translate", niutrans)
    trace = pipeline.run_standard_pipeline("Input", {"api_keys": {
        "deepseek_api_key": "test-llm", "niutrans_api_key": "test-translation",
    }})
    assert trace["result"] == "English"
    assert [step["engine"] for step in trace["steps"]] == ["DeepSeek", "DeepSeek", "Google", "Niutrans"]
    assert calls == [
        ("Google", "Japanese", "ja", "fi"),
        ("Niutrans", "Finnish", "fi", "en", "test-translation"),
    ]


@pytest.mark.parametrize("field,value,error", [
    ("provider", "olama", "translation.provider"),
    ("max_chunk_chars", 0, "max_chunk_chars"),
    ("max_chunk_chars", True, "max_chunk_chars"),
    ("timeout", -1, "timeout"),
    ("model", "", "model"),
    ("base_url", "", "base_url"),
    ("prompt_style", "unsupported", "prompt_style"),
])
def test_bad_translation_config_fails_before_rewriting(monkeypatch, local_config, field, value, error):
    if field == "provider":
        local_config["translation"][field] = value
    else:
        # Even a malformed final stage must fail before spending any inference.
        local_config["translation"]["step4"][field] = value
    monkeypatch.setattr(pipeline, "llm_rewrite", _forbid_cloud)
    with pytest.raises(ValueError, match=error):
        pipeline.run_standard_pipeline("Input", local_config)


def test_unsupported_local_target_fails_before_rewriting(monkeypatch, local_config):
    monkeypatch.setattr(pipeline, "llm_rewrite", _forbid_cloud)
    with pytest.raises(ValueError, match="target language"):
        pipeline.run_standard_pipeline("Input", local_config, target_lang="xx")


def test_long_local_translation_sends_every_character_in_bounded_chunks(monkeypatch):
    chunks = []
    text = "文" * 2300 + "\n\n" + "A sentence with spaces. " * 60

    def completion(messages, **kwargs):
        chunks.append(messages[0]["content"].split("\n\n\n", 1)[1])
        assert kwargs["model"] == "translategemma:12b"
        assert kwargs["base_url"] == "http://mac.local:11434/v1"
        assert kwargs["timeout"] == 900
        return f"Translation {len(chunks)}"

    monkeypatch.setattr(translators, "chat_completions", completion)
    result = translators.ollama_translate(
        text, "ja", "fi", model="translategemma:12b",
        base_url="http://mac.local:11434/v1", timeout=900,
    )
    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= 1000 for chunk in chunks)
    assert result.split("\n\n") == [f"Translation {i + 1}" for i in range(len(chunks))]


@pytest.mark.parametrize("content,reason,error", [
    ("Incomplete", "length", "truncated"),
    ("  ", "stop", "empty text"),
    (None, "stop", "empty text"),
])
def test_incomplete_local_response_is_not_used(monkeypatch, content, reason, error):
    with _simulate_http(monkeypatch, lambda request: _completion(content, reason)):
        with pytest.raises(ValueError, match=error):
            translators.ollama_translate("Test", "fi", "en")


@pytest.mark.parametrize("missing_step", [1, 4])
def test_missing_local_model_fails_without_falling_back(monkeypatch, local_config, missing_step):
    monkeypatch.setattr(pipeline, "google_translate", _forbid_cloud)
    monkeypatch.setattr(pipeline, "niutrans_translate", _forbid_cloud)
    requests = []
    outputs = ["Chinese", "Japanese", "Finnish", "English"]

    def handler(request):
        requests.append(str(request.url))
        if len(requests) == missing_step:
            return httpx.Response(404, json={"error": "model not found"})
        return _completion(outputs[len(requests) - 1])

    with _simulate_http(monkeypatch, handler):
        with pytest.raises(httpx.HTTPStatusError):
            pipeline.run_standard_pipeline("Test", local_config)
    assert requests == ["http://localhost:11434/v1/chat/completions"] * missing_step


def test_local_llm_with_hosted_translation_still_requires_niutrans(local_config):
    config = copy.deepcopy(local_config)
    config.pop("translation")
    with pytest.raises(ValueError, match="niutrans_api_key"):
        pipeline.run_standard_pipeline("Test", config)


def test_legacy_shared_model_still_applies_to_both_translation_stages(monkeypatch, local_config):
    local_config["translation"].pop("step3")
    local_config["translation"].pop("step4")
    local_config["translation"]["model"] = "translategemma:12b"
    outputs = ["Chinese", "Japanese", "Finnish", "English"]
    models = []

    def handler(request):
        models.append(json.loads(request.content)["model"])
        return _completion(outputs[len(models) - 1])

    with _simulate_http(monkeypatch, handler):
        result = pipeline.run_standard_pipeline("Test", local_config)
    assert result["result"] == "English"
    assert models == ["qwen3.8:27b", "qwen3.8:27b", "translategemma:12b", "translategemma:12b"]


def test_final_stage_overrides_connection_and_timeout(monkeypatch, local_config):
    local_config["translation"]["step4"].update({
        "base_url": "http://mac.local:11434/v1/", "timeout": 900,
    })
    requests = []
    outputs = ["Chinese", "Japanese", "Finnish", "English"]

    def handler(request):
        requests.append(request)
        return _completion(outputs[len(requests) - 1])

    with _simulate_http(monkeypatch, handler):
        pipeline.run_standard_pipeline("Test", local_config)
    assert all(request.url.host == "localhost" for request in requests[:3])
    assert str(requests[3].url) == "http://mac.local:11434/v1/chat/completions"
    assert requests[3].extensions["timeout"]["read"] == 900
    assert requests[2].extensions["timeout"]["read"] == 600


@pytest.mark.parametrize("override", ["not a table", {"provider": "deepseek"}])
def test_malformed_stage_overrides_fail_before_rewriting(monkeypatch, local_config, override):
    local_config["translation"]["step4"] = override
    monkeypatch.setattr(pipeline, "llm_rewrite", _forbid_cloud)
    with pytest.raises(ValueError, match="translation.step4"):
        pipeline.run_standard_pipeline("Test", local_config)
