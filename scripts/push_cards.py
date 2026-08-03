#!/usr/bin/env python3
"""Push the standardized Hugging Face model cards.

Reads HF_TOKEN from the environment (or .env) and uploads each card in hf_cards/
to its repo as README.md. The dataset card is intentionally NOT pushed (no
dataset repo exists yet).

    python scripts/push_cards.py --dry-run     # validate + show plan
    python scripts/push_cards.py               # upload (needs a write HF_TOKEN)
"""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

HF_USER = os.environ.get("HF_USERNAME", "Shinzmann")
CARDS_DIR = Path(__file__).resolve().parent.parent / "hf_cards"

# card filename -> repo name (under HF_USER)
CARDS = {
    "naija-petro.md":          "naija-petro",
    "naija-petro-8b.md":       "naija-petro-8b",
    "naija-petro-GGUF.md":     "naija-petro-GGUF",
    "naija-petro-8b-GGUF.md":  "naija-petro-8b-GGUF",
}


class CardError(Exception):
    """A problem with the cards on disk, reported as one line rather than a traceback."""


def die(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def validate() -> list[tuple[Path, str]]:
    plan = []
    problems: list[str] = []
    for fname, repo in CARDS.items():
        path = CARDS_DIR / fname
        if not path.exists():
            problems.append(f"missing card: {path}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{fname}: cannot read ({exc})")
            continue
        if not text.lstrip().startswith("---"):
            problems.append(f"{fname}: missing YAML front matter")
            continue
        plan.append((path, f"{HF_USER}/{repo}"))
    if problems:
        raise CardError("; ".join(problems))
    return plan


def main() -> int:
    dry = "--dry-run" in sys.argv
    try:
        plan = validate()
    except CardError as exc:
        return die(str(exc))

    print(f"Cards directory: {CARDS_DIR}")
    for path, repo in plan:
        print(f"  {path.name:24s} -> https://huggingface.co/{repo}")
    print("  (dataset_card.md is not pushed, no dataset repo yet)\n")

    if dry:
        print("Dry run OK. Re-run without --dry-run to upload.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        return die("HF_TOKEN not set. Add it to .env or export it, then retry.")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return die("the 'huggingface_hub' package is not installed. Run: pip install huggingface_hub")

    api = HfApi(token=token)
    for path, repo in plan:
        try:
            api.upload_file(
                path_or_fileobj=path.read_bytes(),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="model",
                commit_message="Update model card (standardized)",
            )
        except Exception as exc:
            # Hub errors can quote the request; strip anything that looks like the
            # token before it reaches the terminal or a CI log.
            detail = str(exc).replace(token, "<HF_TOKEN>")
            return die(f"upload of {path.name} to {repo} failed: {type(exc).__name__}: {detail}")
        print(f"  pushed {path.name} -> https://huggingface.co/{repo}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
