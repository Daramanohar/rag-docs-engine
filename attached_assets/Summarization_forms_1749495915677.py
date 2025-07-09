# Summarization_forms.py
from groq import Groq

def summarize_text(text, groq_key):
    client = Groq(api_key=groq_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an AI that summarizes form content concisely."},
            {"role": "user", "content": f"Summarize the following form data:\n{text}"}
        ]
    )

    return response.choices[0].message.content
