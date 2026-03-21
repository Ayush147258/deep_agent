from config import GOOGLE_API_KEY, TAVILY_API_KEY, NVIDIA_API_KEY

from google import genai
from tavily import TavilyClient
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- Gemini ----------
gemini = genai.Client(api_key=GOOGLE_API_KEY)


# ---------- NVIDIA ----------
nvidia = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)


# ---------- Search ----------
search = TavilyClient(api_key=TAVILY_API_KEY)


# ---------- Memory ----------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="memory",
    embedding_function=embeddings,
)


def memory_search(query):

    docs = db.similarity_search(query, k=3)

    text = ""

    for d in docs:
        text += d.page_content + "\n"

    return text


def research(query):

    print("Memory search...")
    mem = memory_search(query)

    print("Web search...")
    web = search.search(
        query=query,
        max_results=5
    )

    web_text = str(web)


    print("Gemini thinking...")

    g = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Memory:
{mem}

Web:
{web_text}

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
Memory:
{mem}

Web:
{web_text}

Question:
{query}
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
        "latest AI models 2026"
    )