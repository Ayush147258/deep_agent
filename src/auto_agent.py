from config import GOOGLE_API_KEY, TAVILY_API_KEY, NVIDIA_API_KEY

from google import genai
from tavily import TavilyClient
from openai import OpenAI

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- LLM ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
)


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


# ---------- TOOLS ----------

@tool
def memory_tool(query: str) -> str:
    docs = db.similarity_search(query, k=3)

    text = ""

    for d in docs:
        text += d.page_content + "\n"

    return text


@tool
def web_tool(query: str) -> str:
    r = search.search(
        query=query,
        max_results=5
    )
    return str(r)


@tool
def nvidia_tool(query: str) -> str:

    r = nvidia.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": query}],
    )

    return r.choices[0].message.content


tools = [
    memory_tool,
    web_tool,
    nvidia_tool,
]


llm_with_tools = llm.bind_tools(tools)


def run_agent(question):

    result = llm_with_tools.invoke(question)

    print(result)


if __name__ == "__main__":

    run_agent(
        "What are latest AI models and how agents work"
    )