# Run the complete pipeline locally with Ollama

The Standard Pipeline can run all four steps on your Mac. A general multilingual
model handles the two rewrites, and TranslateGemma replaces both translation
services. The local configuration requires no API keys.

| Step | Local model | Languages |
| --- | --- | --- |
| 1 | Qwen 3.8 27B | Input -> Chinese rewrite |
| 2 | Qwen 3.8 27B | Chinese -> Japanese rewrite, with history |
| 3 | TranslateGemma 27B | Japanese -> Finnish translation |
| 4 | TranslateGemma 27B | Finnish -> English translation |

This keeps the original language chain. Using one local translation model changes
the two-engine design, so results can differ from Google Translate and Niutrans.
Check meaning and terminology on your own examples, especially Japanese -> Finnish.

## Mac setup

You need Python 3.10+ and the native [Ollama app](https://ollama.com/download/mac).
Open Ollama before running the pipeline. Use native Ollama on macOS for GPU
acceleration; Docker Desktop on macOS does not provide GPU passthrough.
Use a current Ollama release with Qwen 3.8 support.

In Terminal, clone this fork and install the Python package:

```bash
git clone https://github.com/tillsro/humanize-text.git
cd humanize-text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Download the two local models:

```bash
ollama pull qwen3.8:27b
ollama pull translategemma:27b
ollama list
```

The default quantized downloads are about 18 GB and 17 GB respectively. These are
a starting point for a Mac with 64 GB of unified memory. Runtime use also includes
context buffers, macOS, and other applications; speed has not been measured on
your Mac. To reduce memory use, choose `qwen2.5:7b` for rewriting or
`translategemma:12b` / `translategemma:4b` for translation. Download the chosen model
and update its name in the configuration.

Run directly with the supplied local configuration:

```bash
humanize-text --config config/config.ollama.example.toml --input "Your text here" --verbose
```

For a UTF-8 text file and a saved result:

```bash
humanize-text --config config/config.ollama.example.toml --input draft.txt --output rewritten.txt
```

To make local mode your default, copy the example to `config/config.toml`, or edit
an existing configuration so both `[llm].provider` and `[translation].provider` are
`"ollama"`. Setting only the LLM provider still leaves the default Google/Niutrans
translation stages enabled.

## Configuration

See [config/config.ollama.example.toml](../config/config.ollama.example.toml) for a
complete configuration. The two model names are independent:

- `[llm].model` selects the rewrite model. Ollama defaults to `qwen3.8:27b`, even if
  the legacy `[pipeline].model` still names a cloud model.
- `[translation].model` selects the translator and defaults to `translategemma:27b`.
- Both `base_url` values default to `http://localhost:11434/v1`.
- Both request timeouts default to 600 seconds for Ollama to allow for model loading.
- `translation.max_chunk_chars` defaults to 1000. Longer passages are split into
  bounded chunks; translated chunks are separated by blank lines. Chunking can
  change paragraph breaks and lose context between chunks. The rewrite stages
  still receive the whole input and need enough model context for it.
- The local translator accepts `en`, `zh`, `ja`, `ko`, `fr`, `de`, `es`, `pt`, `ru`,
  `ar`, `it`, `nl`, and `fi` as source/target codes.

The existing `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, and `LLM_API_KEY` environment
variables still override the `[llm]` settings. Clear old cloud overrides when
switching to local mode. Translation settings are taken from `[translation]`.
Ollama requests bypass environment-configured HTTP proxies. Failed local requests
raise an error; the app does not fall back to a cloud provider.

For the default hosted translators, leave `[translation]` out or set
`provider = "google_niutrans"`. That mode still requires a Niutrans key. The LLM
provider and the translation provider can also be selected independently.

## Keeping inference on your Mac

The example uses localhost and downloaded model tags. After installing packages
and downloading models, inference can run offline. Use local model tags rather
than cloud model tags. Ollama also supports disabling its cloud features: set
`disable_ollama_cloud` to `true` in `~/.ollama/server.json`, preserving any other
settings in that file, then restart Ollama.

If a model is missing, use `ollama list` and download the exact tag from the
configuration. If the server is unavailable, start the Ollama app. A truncated
response raises an error rather than silently passing incomplete text to the next
stage; shorten the input or increase the model context/output limit.

The automated tests use simulated Ollama responses to check request routing,
language order, chunking, and cloud-service isolation. Live model quality and Mac
performance require testing on the machine that runs Ollama.

## References

- [TranslateGemma models and translation prompt guidance](https://ollama.com/library/translategemma)
- [Qwen 3.8 27B](https://ollama.com/library/qwen3.8:27b)
- [Ollama's OpenAI-compatible local API](https://docs.ollama.com/api/openai-compatibility)
- [Ollama local FAQ, cloud settings, and macOS GPU notes](https://docs.ollama.com/faq)
