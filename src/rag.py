from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# create DB
db = Chroma(
    persist_directory="memory",
    embedding_function=embeddings,
)


def add_text(text):

    db.add_texts([text])
    db.persist()


def search_text(query):

    docs = db.similarity_search(query, k=3)

    for d in docs:
        print(d.page_content)


if __name__ == "__main__":

    add_text("AI agents can use tools and memory")
    add_text("Gemini is Google AI model")
    add_text("CrewAI is multi agent framework")

    print("Search result:\n")

    search_text("What is AI")