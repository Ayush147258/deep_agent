from config import GOOGLE_API_KEY, TAVILY_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from tavily import TavilyClient


# -------- LLM --------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
)


# -------- Tavily --------
search_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def web_search(query: str) -> str:
    """Search the internet"""
    result = search_client.search(
        query=query,
        max_results=5,
    )
    return str(result)


tools = [web_search]


# Bind tools to LLM (new style)
llm_with_tools = llm.bind_tools(tools)


def run_agent(question: str):
    response = llm_with_tools.invoke(
        [HumanMessage(content=question)]
    )
    print(response)


if __name__ == "__main__":
    run_agent(
        "What are the newest AI models in 2026?"
    )