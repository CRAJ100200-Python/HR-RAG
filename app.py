import os

# =========================
# 1. CONFIG
# =========================

APP_MODE = os.getenv("APP_MODE", "colab")  
# default = colab (safe fallback)

print(f"Running in mode: {APP_MODE}")


# =========================
# 2. COMMON FUNCTION (stub)
# =========================

def answer_question(question: str) -> str:
    # placeholder for now
    return f"[Stub Answer] You asked: {question}"


# =========================
# 3. FASTAPI MODE (Render)
# =========================

if APP_MODE == "api":
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class Query(BaseModel):
        question: str

    @app.get("/")
    def home():
        return {"message": "API is running"}

    @app.post("/ask")
    def ask(q: Query):
        return {"answer": answer_question(q.question)}


# =========================
# 4. GRADIO MODE (HF Spaces)
# =========================

elif APP_MODE == "gradio":
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
# 5. COLAB MODE (CLI)
# =========================

elif APP_MODE == "colab":
    print("Entering interactive mode (type 'exit' to quit)\n")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break

        ans = answer_question(q)
        print("Bot:", ans)