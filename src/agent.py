from config import GOOGLE_API_KEY, TAVILY_API_KEY
from google import genai
from tavily import TavilyClient


# init clients
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
search_client = TavilyClient(api_key=TAVILY_API_KEY)


def research(query):

    # 1 search web
    search_result = search_client.search(
        query=query,
        max_results=5
    )

    text_data = str(search_result)

    # 2 send to Gemini
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
You are a research AI.
Use the search results below to answer.

Search results:
{text_data}

Question:
{query}

Give detailed professional answer.
"""
    )

    print(response.text)


if __name__ == "__main__":
    research("latest AI models 2026")