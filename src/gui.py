from ollama_llm import ask_ollama
from config import GOOGLE_API_KEY, TAVILY_API_KEY

from google import genai
from tavily import TavilyClient

import gradio as gr
import logging

logging.basicConfig(level=logging.INFO)


# ---------------- CHROMA ----------------

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings


print("[1/4] Clients loaded...")

gemini = genai.Client(api_key=GOOGLE_API_KEY)
search = TavilyClient(api_key=TAVILY_API_KEY)


# ---------------- EMBEDDINGS ----------------

print("[2/4] Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="memory",
    embedding_function=embeddings,
)

print("[3/4] Memory ready")


# ---------------- MEMORY ----------------

def memory_search(query):

    try:

        docs = db.similarity_search(query, k=8)

        texts = []

        for d in docs:
            if len(d.page_content) > 20:
                texts.append(d.page_content)

        return "\n---\n".join(texts) or "No memories yet."

    except Exception as e:
        return f"Memory error: {e}"


def memory_save(user_msg, ai_reply):

    try:

        db.add_texts(
            [f"User: {user_msg}\nAssistant: {ai_reply}"]
        )

    except Exception:
        pass


# ---------------- WEB ----------------

def web_search(query):

    try:

        results = search.search(
            query=query,
            max_results=5
        )

        lines = []

        for r in results.get("results", []):

            lines.append(
                f"• {r.get('title','')}: {r.get('content','')[:300]}"
            )

        return "\n".join(lines) or "No results."

    except Exception as e:

        return f"Search error: {e}"


# ---------------- SESSION ----------------

session = []

MAX_TURNS = 40


SYSTEM = """
You are ARIA — Advanced Research Intelligence Assistant.

Rules:
- Maintain conversation consistency
- Use memory first
- Use web for current facts
- Give detailed structured answers
- Explain reasoning step by step
- Never give one-line replies to complex questions
- If unsure, say unsure

Format:
- Use headings
- Use bullet points
- Be logical
"""


# ---------------- CHAT ----------------

def chat(message, history):

    global session

    session.append({
        "role": "user",
        "content": message
    })

    recent = session[-MAX_TURNS:]

    context = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}"
        for t in recent
    )

    mem = memory_search(message)

    web = web_search(message)

    msg = message.lower()


    # -------- MODEL ROUTING --------

    model_name = "gemini-2.5-flash"

    if len(message) > 120:
        model_name = "gemini-2.5-pro"

    if "research" in msg:
        model_name = "gemini-2.5-pro"

    if "explain" in msg:
        model_name = "gemini-2.5-pro"

    if "why" in msg:
        model_name = "gemini-2.5-pro"

    if "how" in msg:
        model_name = "gemini-2.5-pro"

    if "detail" in msg:
        model_name = "gemini-2.5-pro"


    # -------- PROMPT (FIXED) --------

    prompt = f"""
{SYSTEM}

Conversation:
{context}

Memory:
{mem}

Web:
{web}

User:
{message}

Answer as ARIA:
"""


    # -------- GEMINI --------

    try:

        r = gemini.models.generate_content(
            model=model_name,
            contents=prompt
        )

        answer = r.text.strip()

    except Exception as e:

        print("Gemini failed → using Ollama")

        answer = ask_ollama(prompt)


    session.append({
        "role": "assistant",
        "content": answer
    })

    memory_save(message, answer)

    return answer


# ---------------- UI ----------------

print("[4/4] Building UI...")

demo = gr.ChatInterface(
    fn=chat,
    title="ARIA — AI Research Assistant",
    description="Gemini + Tavily + Memory + Ollama",
)


print("Launching on http://localhost:7860")


try:

    demo.launch(
        server_name="localhost",
        server_port=7860,
        debug=True,
    )

except OSError:

    print("Port busy, using 7861")

    demo.launch(
        server_name="localhost",
        server_port=7861,
        debug=True,
    )