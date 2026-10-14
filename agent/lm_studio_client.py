
import os
import requests


def chat_completion(
    messages,
    max_tokens=2048,
    temperature=0,
    model=None,
):
    import config

    model = model or config.LM_STUDIO_MODEL
    import config

    base_url = os.getenv(
        "LM_STUDIO_BASE_URL",
        "http://localhost:1234/v1"
    ).rstrip("/")

    model = config.LM_STUDIO_MODEL

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"LM Studio connection failed: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"LM Studio HTTP {response.status_code}\n"
            f"Model: {model}\n"
            f"Details: {response.text[:3000]}"
        )

    data = response.json()

    if not data.get("choices"):
        raise RuntimeError(
            f"LM Studio returned no choices: {data}"
        )

    choice = data["choices"][0]
    content = choice.get("message", {}).get("content")

    if not content:
        raise RuntimeError(
            "LM Studio returned empty content. "
            f"Finish reason: {choice.get('finish_reason')}"
        )

    return content
