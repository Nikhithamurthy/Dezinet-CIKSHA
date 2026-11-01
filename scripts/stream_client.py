import json
import sys
import time

import httpx

URL = "http://127.0.0.1:8000/v1/stream"
HEADERS = {"X-API-Key": "dev-key-123"}


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Explain retrieval augmented generation"
    start = time.perf_counter()
    first_token_at: float | None = None
    with httpx.stream("POST", URL, json={"prompt": prompt}, headers=HEADERS, timeout=30) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "token" in data:
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - start
                    print(data["token"], end="", flush=True)
    print()
    if first_token_at is not None:
        print(f"TTFT: {first_token_at * 1000:.0f} ms")


if __name__ == "__main__":
    main()
