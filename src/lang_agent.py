from config import GOOGLE_API_KEY, TAVILY_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

search = TavilyClient(api_key=TAVILY_API_KEY)


def research(query):

    result = search.search(
        query=query,
        max_results=5
    )

    text = str(result)

    prompt = f"""
You are a professional research AI.

Use the search data below to answer.

Search:
{text}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print(response.content)


if __name__ == "__main__":
    research("latest AI models 2026")