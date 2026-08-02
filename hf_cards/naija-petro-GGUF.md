---
license: apache-2.0
language:
- en
base_model: Shinzmann/naija-petro
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

# Naija-Petro (32B) GGUF

GGUF quantizations of [`Shinzmann/naija-petro`](https://huggingface.co/Shinzmann/naija-petro) (the 32B variant) for inference with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), LM Studio, and compatible runtimes.

See the full [model card](https://huggingface.co/Shinzmann/naija-petro) for training details, intended use, and limitations. For a lighter footprint, consider the [8B GGUF](https://huggingface.co/Shinzmann/naija-petro-8b-GGUF). For Nigeria-specific accuracy, use these weights with the [Naija-Petro RAG system](https://github.com/Mystique1337/naija-petro).

## Available quantizations

| File suffix | Method | Notes |
|---|---|---|
| `Q4_K_M` | 4-bit (k-quant, medium) | Smallest; recommended default for 32B on limited RAM/VRAM |
| `Q5_K_M` | 5-bit (k-quant, medium) | Higher quality, larger |
| `Q8_0`   | 8-bit | Near-lossless; largest and slowest |

> 32B GGUF files are large. Q4_K_M is the practical choice for most machines; ensure you have enough RAM/VRAM + disk for the chosen quant.

## Usage

**Ollama**
```bash
ollama run hf.co/Shinzmann/naija-petro-GGUF:Q4_K_M
```

**llama.cpp**
```bash
./llama-cli -hf Shinzmann/naija-petro-GGUF:Q4_K_M \
  -p "What are the screening criteria for CO2 EOR?" -c 4096
```

## License
Apache-2.0 (inherited from Qwen3-32B). Validate outputs with qualified engineers before operational use.
