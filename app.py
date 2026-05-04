import os

# =========================
# CONFIG
# =========================

APP_MODE = os.getenv("APP_MODE", "colab")
print(f"Running in mode: {APP_MODE}")

# =========================
# DATA
# =========================

documents = [
    "Employees are entitled to 20–22 days of paid annual leave per year. Work-from-home is permitted for up to 3 days weekly with prior manager approval.",
    "The standard work schedule is 9:00 AM to 6:00 PM with a one-hour break. Flexible start times are allowed.",
    "Employees are eligible for 10–12 days of paid sick leave annually. A medical certificate may be required after extended absence.",
    "Parental leave includes maternity and paternity benefits as per company policy.",
    "Overtime must be pre-approved and is compensated either financially or with time off."
]

# =========================
# LAZY RAG SETUP
# =========================

retriever = None
rag_chain = None


def init_rag():
    global retriever, rag_chain

    if retriever is not None:
        return

    print("Initializing RAG pipeline...")

    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Convert to docs
    docs = [Document(page_content=d) for d in documents]

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an HR assistant. Answer ONLY from the context.

    Context:
    {context}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG initialized successfully")


def answer_question(question: str) -> str:
    init_rag()
    return rag_chain.invoke(question)


# =========================
# FASTAPI (Render)
# =========================

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "HR RAG API is running"}


@app.post("/ask")
def ask(q: Query):
    answer = answer_question(q.question)
    return {"answer": answer}


# =========================
# GRADIO (Hugging Face)
# =========================

if APP_MODE == "gradio":
    import gradio as gr

    def gradio_fn(question):
        return answer_question(question)

    demo = gr.Interface(
        fn=gradio_fn,
        inputs="text",
        outputs="text",
        title="HR Assistant",
        description="Ask HR-related questions"
    )

    if __name__ == "__main__":
        demo.launch()


# =========================
# COLAB (CLI)
# =========================

if APP_MODE == "colab":
    if __name__ == "__main__":
        print("Interactive mode (type 'exit' to quit)\n")

        while True:
            q = input("You: ")
            if q.lower() in ["exit", "quit"]:
                break

            print("Bot:", answer_question(q))