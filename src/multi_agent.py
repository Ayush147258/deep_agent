from config import GOOGLE_API_KEY, TAVILY_API_KEY, NVIDIA_API_KEY

from google import genai
from tavily import TavilyClient
from openai import OpenAI


# ---------- Gemini ----------
gemini = genai.Client(api_key=GOOGLE_API_KEY)


# ---------- Tavily ----------
search = TavilyClient(api_key=TAVILY_API_KEY)


# ---------- NVIDIA ----------
nvidia = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)


def research(query):

    print("Searching...")
    s = search.search(
        query=query,
        max_results=5
    )

    text = str(s)

    print("Gemini thinking...")
    g = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Use search results to answer.

Search:
{text}

Question:
{query}
"""
    )

    gemini_answer = g.text


    print("NVIDIA thinking...")
    n = nvidia.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": f"""
Search results:
{text}

Question:
{query}

Give professional answer.
"""
            }
        ]
    )

    nvidia_answer = n.choices[0].message.content


    print("\n----- GEMINI -----\n")
    print(gemini_answer)

    print("\n----- NVIDIA -----\n")
    print(nvidia_answer)


if __name__ == "__main__":
    research(
        "latest AI models in 2026"
    )