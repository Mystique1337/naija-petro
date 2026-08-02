<!--
This is the standard dataset card content for the Naija-Petro corpus.
No dataset repo exists yet. When you create one (e.g. Shinzmann/naija-petro-corpus),
upload the data and use this file as its README.md (keep the YAML front matter).
-->
---
license: apache-2.0
language:
- en
task_categories:
- text-generation
- question-answering
size_categories:
- 10K<n<100K
tags:
- petroleum-engineering
- oil-and-gas
- synthetic
- instruction-tuning
- nvidia-data-designer
pretty_name: Naija-Petro Petroleum Engineering Corpus
---

# Naija-Petro Petroleum Engineering Corpus

A synthetic, instruction-tuning dataset of **~20,000+ petroleum-engineering instruction-response pairs**, generated with **NVIDIA NeMo Data Designer** from a scraped, de-duplicated corpus of open petroleum literature. It is the training set for the [Naija-Petro](https://huggingface.co/Shinzmann/naija-petro) models.

## Dataset summary
- **Domain:** petroleum engineering covering drilling, reservoir, production, completions, EOR, well testing, petroleum geoscience
- **Size:** ~20,000+ pairs after quality filtering and de-duplication
- **Language:** English
- **Use:** supervised fine-tuning / instruction tuning of LLMs

## Supported tasks
- **Text generation / instruction following:** generate a technically accurate response to a petroleum-engineering instruction.
- **Question answering:** seed-grounded QA derived from real literature.

## Dataset structure

Provided in three formats:

| Format | File | Schema |
|---|---|---|
| Alpaca | `final_alpaca_format.jsonl` | `{ "instruction", "input", "output" }` |
| ShareGPT | `final_sharegpt_format.jsonl` | `{ "conversations": [{from:"human", value}, {from:"gpt", value}] }` |
| Full (Parquet) | `final_full_dataset.parquet` | instruction, response, `pipeline`, and (where available) category, complexity level, subdomain, and LLM-judge quality scores |

A typical record:
```json
{"instruction": "Explain the material balance equation for an undersaturated reservoir.",
 "input": "",
 "output": "The material balance equation relates cumulative production to ..."}
```

## Dataset creation

### Curation rationale
General LLMs underperform on specialised petroleum-engineering tasks. This corpus distils open literature into diverse, high-quality instruction-response pairs for domain fine-tuning.

### Source data
Scraped from 25+ open sources, including arXiv, Semantic Scholar, OpenAlex, Crossref, DOE/OSTI, DOAJ, Unpaywall, PetroWiki, the SLB Oilfield Glossary, Wikipedia, and EIA. Sources were consolidated, de-duplicated, and chunked into a seed corpus.

### Generation
Three NVIDIA Data Designer pipelines:
1. **Knowledge-based** instruction-response generation across sampled categories, complexity levels, and subdomains.
2. **Seed-grounded QA** using scraped text as context for literature-grounded answers.
3. **Quality scoring** via LLM-as-judge (technical accuracy, completeness, usefulness).

### Filtering
Empty/short records removed (instruction > 20 chars, response > 50 chars) and de-duplicated by instruction.

## Considerations & limitations
- **Synthetic data**: responses are model-generated and may contain inaccuracies; not authoritative.
- Reflects biases of the source literature and the generating models.
- English only; **general/global** petroleum knowledge, with limited Nigeria-specific coverage (addressed at inference time by the project's RAG layer).
- Intended for research and education; **validate before operational use.**

## Licensing
Released under Apache-2.0. Underlying source documents remain under their respective licenses; this dataset comprises synthetic text derived from openly accessible material.

## Citation
```bibtex
@misc{naijapetrocorpus2025,
  title  = {Naija-Petro Petroleum Engineering Corpus},
  author = {Naija-Petro project},
  year   = {2025},
  note   = {Synthetic instruction-tuning dataset generated with NVIDIA NeMo Data Designer}
}
```
