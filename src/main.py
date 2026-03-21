from config import GOOGLE_API_KEY
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what is AI agent in simple terms"
)

print(resp.text)