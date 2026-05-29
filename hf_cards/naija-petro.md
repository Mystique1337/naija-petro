---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3-32B
tags:
- petroleum-engineering
- oil-and-gas
- nigeria
- qwen3
- unsloth
- lora
- fine-tuned
---

# Naija-Petro (32B)

**Naija-Petro** is a [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) model fine-tuned (QLoRA, [Unsloth](https://github.com/unslothai/unsloth)) on ~20,000 synthetic petroleum-engineering instruction–response pairs. It is the **highest-quality** variant of the Naija-Petro family; for fast/low-cost serving see the [8B variant](https://huggingface.co/Shinzmann/naija-petro-8b).

> ⚠️ The base training data is **general/global** petroleum knowledge. For Nigeria-specific facts (regulation, the PIA 2021, NUPRC/NMDPRA/NNPC), pair this model with the [Naija-Petro RAG system](https://github.com/Mystique1337/naija-petro), which grounds answers in verifiable Nigerian sources.

## Model details

- **Developed by:** Naija-Petro project (Hugging Face: [`Shinzmann`](https://huggingface.co/Shinzmann))
- **Model type:** Decoder-only causal LM, instruction-tuned
- **Language:** English
- **License:** Apache-2.0 (inherited from Qwen3-32B)
- **Finetuned from:** [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B)
- **Domain:** Petroleum engineering — drilling, reservoir, production, completions, EOR, well testing, petroleum geoscience

### Model sources
- **Repository:** https://github.com/Mystique1337/naija-petro
- **GGUF (llama.cpp / Ollama):** [`Shinzmann/naija-petro-GGUF`](https://huggingface.co/Shinzmann/naija-petro-GGUF)
- **Lighter variant (8B):** [`Shinzmann/naija-petro-8b`](https://huggingface.co/Shinzmann/naija-petro-8b)

## Uses

### Direct use
Technical question answering and explanation across petroleum-engineering subdomains: concepts, equations and derivations, workflow guidance, and terminology — as a study aid and engineering decision-support tool.

### Downstream use
Backbone for retrieval-augmented assistants (see the project repo), further domain fine-tuning, or distillation into smaller models.

### Out-of-scope use
Not for autonomous operational, safety-critical, or financial decisions; not a substitute for licensed engineering judgment, official regulations, or field data. General-purpose chat is not its focus.

## Bias, risks, and limitations
- Trained largely on **synthetic** data generated from a scraped corpus; it can be confidently wrong ("hallucinate"), especially on numerical specifics and **Nigeria-specific** regulation/economics.
- English only. May reflect biases of its base model and source literature.
- Knowledge is static as of training; use the RAG layer for current/local facts.
- 32B requires a GPU for practical inference; use the 8B or a GGUF quant for lighter setups.

**Recommendation:** Always validate outputs with qualified engineers and primary sources before any operational use.

## How to get started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Shinzmann/naija-petro"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

SYSTEM = (
    "You are Naija-Petro, an expert petroleum-engineering AI assistant. Provide "
    "precise, technically accurate answers; include equations, units, and "
    "practical considerations."
)
messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "What are the screening criteria for CO2 EOR?"},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 enable_thinking=False, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, temperature=0.4, top_p=0.9)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

**Ollama (GGUF):**
```bash
ollama run hf.co/Shinzmann/naija-petro-GGUF:Q4_K_M
```

## Training details

### Data
~20,000 instruction–response pairs generated with **NVIDIA NeMo Data Designer** from a scraped, de-duplicated petroleum corpus (arXiv, Semantic Scholar, OpenAlex, Crossref, DOE/OSTI, PetroWiki, the SLB glossary, EIA, and more), with an LLM-as-judge quality-scoring pass. Pipeline and EDA are in the [project notebooks](https://github.com/Mystique1337/naija-petro/tree/main/notebooks).

### Procedure
QLoRA (4-bit NF4) with Unsloth on a single A100 80GB.

| Hyperparameter | Value |
|---|---|
| Base model | Qwen3-32B |
| Method | QLoRA, 4-bit NF4 |
| LoRA rank / alpha / dropout | 64 / 128 / 0.0 |
| Epochs | 2 |
| Effective batch size | 64 (8 × 8 grad-accum) |
| Learning rate / scheduler | 2e-4 / cosine, 5% warmup |
| Weight decay | 0.01 |
| Max sequence length | 2048 |
| Seed | 42 |

### Training results (Weights & Biases)
Trained for 2 epochs (446 optimiser steps) on a single A100 80GB.

| Metric | Value |
|---|---|
| Final training loss | ≈ 0.74 |
| Final validation loss | ≈ 0.79 |
| Epochs / steps | 2 / 446 |

## Evaluation
Training converged to a low validation loss (≈ 0.79; see *Training results*) — lower than the 8B variant (≈ 0.86), consistent with the larger model's higher capacity. A downstream 30-question internal benchmark across six subdomains (drilling, reservoir, production, completions, EOR, well testing), scored by an LLM-as-judge, is being finalised; task-level scores are not yet reported here. Qualitatively, the fine-tuned model produces detailed, well-structured, equation- and unit-aware answers in the target domain. Treat all outputs as expert-validated decision support.

## Technical specifications
- **Architecture:** Qwen3 (decoder-only transformer), 32B parameters
- **Adapter:** LoRA merged into 16-bit weights; also distributed as GGUF quantizations
- **Software:** Unsloth, 🤗 Transformers, PEFT, TRL, bitsandbytes
- **Hardware:** 1× NVIDIA A100 80GB

## Citation
```bibtex
@misc{naijapetro2025,
  title  = {Naija-Petro: A Petroleum-Engineering Language Model},
  author = {Naija-Petro project},
  year   = {2025},
  url    = {https://huggingface.co/Shinzmann/naija-petro}
}
```

## Model card contact
Open an issue at https://github.com/Mystique1337/naija-petro.
