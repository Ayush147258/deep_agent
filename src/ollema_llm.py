import requests


def ask_ollama(prompt):

    url = "http://localhost:11434/api/generate"

    data = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
    }

    try:
        r = requests.post(url, json=data)
        return r.json()["response"]

    except Exception as e:
        return f"Ollama error: {e}"