from config import GOOGLE_API_KEY, TAVILY_API_KEY

from google import genai
from tavily import TavilyClient

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- LLM ----------
gemini = genai.Client(api_key=GOOGLE_API_KEY)


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

    resp = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Use memory and web.

Memory:
{mem}

Web:
{web_text}

Question:
{query}
"""
    )

    print(resp.text)


if __name__ == "__main__":

    research(
        "What are AI agents"
    )