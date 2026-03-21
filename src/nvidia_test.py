from config import NVIDIA_API_KEY
from openai import OpenAI

client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)

resp = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "Explain AI agents"}
    ]
)

print(resp.choices[0].message.content)