from config import TAVILY_API_KEY
from tavily import TavilyClient

client = TavilyClient(api_key=TAVILY_API_KEY)

result = client.search(
    query="latest AI models 2026",
    max_results=5
)

print(result)