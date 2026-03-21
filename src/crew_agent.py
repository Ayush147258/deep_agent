from config import GOOGLE_API_KEY, TAVILY_API_KEY

from crewai import Agent, Task, Crew, LLM
from tavily import TavilyClient


# -------- LLM (Gemini) --------

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
)


# -------- Search --------

search = TavilyClient(api_key=TAVILY_API_KEY)


def search_tool(query):
    return str(
        search.search(
            query=query,
            max_results=5
        )
    )


# -------- Agents --------

researcher = Agent(
    role="Researcher",
    goal="Find information",
    backstory="Expert searcher",
    llm=llm,
    verbose=True,
)

analyst = Agent(
    role="Analyst",
    goal="Analyze data",
    backstory="Expert thinker",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Write final answer",
    backstory="Professional writer",
    llm=llm,
    verbose=True,
)



# -------- Tasks --------

task1 = Task(
    description="Find latest AI models 2026",
    expected_output="List of latest AI models",
    agent=researcher,
)

task2 = Task(
    description="Analyze research results",
    expected_output="Analysis of models",
    agent=analyst,
)

task3 = Task(
    description="Write final answer",
    expected_output="Final detailed explanation",
    agent=writer,
)


# -------- Crew --------

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[task1, task2, task3],
)


if __name__ == "__main__":
    result = crew.kickoff()
    print(result)