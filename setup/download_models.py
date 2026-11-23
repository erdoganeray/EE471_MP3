"""Download the three models the app uses. Safe to re-run: finished files are skipped, partial ones resume."""
import time

from huggingface_hub import snapshot_download

MODELS = {
    # text -> image: SD-Turbo, fp16 weights only (about 2.6 GB instead of 10+ GB)
    "stabilityai/sd-turbo": ["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json",
                             "*.fp16.safetensors"],
    # speech -> text: Whisper large-v3 turbo (fp16, about 1.6 GB)
    "openai/whisper-large-v3-turbo": ["*.json", "*.txt", "model.safetensors"],
    # chatbot: SmolLM2-360M-Instruct (about 0.7 GB)
    "HuggingFaceTB/SmolLM2-360M-Instruct": ["*.json", "*.txt", "model.safetensors"],
}


def download(repo, patterns, max_tries=50):
    for attempt in range(1, max_tries + 1):
        try:
            print(f"\n=== {repo} attempt {attempt}", flush=True)
            path = snapshot_download(repo, allow_patterns=patterns)
            print(f"OK  {repo} -> {path}", flush=True)
            return True
        except Exception as e:  # dropped connection etc. -> wait and resume
            wait = min(10 * attempt, 120)
            print(f"ERR {repo}: {type(e).__name__}: {e} -> retry in {wait}s", flush=True)
            time.sleep(wait)
    return False


if __name__ == "__main__":
    failed = [repo for repo, patterns in MODELS.items() if not download(repo, patterns)]
    print("\nDONE." if not failed else f"\nFAILED: {failed}")
