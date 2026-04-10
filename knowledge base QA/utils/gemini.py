import os
from google import genai

def ask_gemini(context: str, question: str) -> str:
    client = genai.Client(api_key=os.getenv("gemini_key"))
    prompt = f"""Answer the user's question using only the context below.

Context:
{context}

Question:
{question}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text