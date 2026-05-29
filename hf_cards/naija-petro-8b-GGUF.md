---
license: apache-2.0
language:
- en
base_model: Shinzmann/naija-petro-8b
pipeline_tag: text-generation
tags:
- petroleum-engineering
- oil-and-gas
- nigeria
- qwen3
- gguf
- llama-cpp
- ollama
- quantized
---

# Naija-Petro 8B — GGUF

GGUF quantizations of [`Shinzmann/naija-petro-8b`](https://huggingface.co/Shinzmann/naija-petro-8b) for CPU/edge inference with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), LM Studio, and compatible runtimes.

See the full [model card](https://huggingface.co/Shinzmann/naija-petro-8b) for training details, intended use, and limitations. For Nigeria-specific accuracy, use these weights with the [Naija-Petro RAG system](https://github.com/Mystique1337/naija-petro).

## Available quantizations

| File suffix | Method | Notes |
|---|---|---|
| `Q4_K_M` | 4-bit (k-quant, medium) | Best size/quality trade-off — recommended default |
| `Q8_0`   | 8-bit | Near-lossless; larger and slower |

## Usage

**Ollama**
```bash
ollama run hf.co/Shinzmann/naija-petro-8b-GGUF:Q4_K_M
```

**llama.cpp**
```bash
# download a specific quant, then:
./llama-cli -hf Shinzmann/naija-petro-8b-GGUF:Q4_K_M \
  -p "Explain the material balance equation for an undersaturated reservoir." \
  -c 4096
```

**Python (llama-cpp-python)**
```python
from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="Shinzmann/naija-petro-8b-GGUF",
    filename="*Q4_K_M.gguf",
    n_ctx=4096,
)
print(llm.create_chat_completion(messages=[
    {"role": "system", "content": "You are Naija-Petro, an expert petroleum-engineering AI assistant."},
    {"role": "user", "content": "How do you interpret a Horner plot?"},
])["choices"][0]["message"]["content"])
```

## License
Apache-2.0 (inherited from Qwen3-8B). Validate outputs with qualified engineers before operational use.
