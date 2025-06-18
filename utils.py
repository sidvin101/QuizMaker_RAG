import PyPDF2
import os
import openai
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import re

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")

# Initialize clients
pc = Pinecone(api_key=pinecone_api_key)

# --- Core Functions ---

def extract_text(pdf_path):
    """Extracts full text from a PDF file."""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return full_text

def chunk_text(text, chunk_size=1000):
    """Splits text into overlapping or fixed-size chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_text(text):
    """Generates a single embedding using OpenAI."""
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def store_embeddings(chunks, namespace):
    """Stores embedded chunks into Pinecone under a specific namespace."""
    index = pc.Index(host=pinecone_index_host)
    vectors = [
        {
            "id": f"{namespace}_chunk_{i}",
            "values": embed_text(chunk),
            "metadata": {"text": chunk}
        }
        for i, chunk in enumerate(chunks)
    ]
    index.upsert(vectors=vectors)
    index.close()

def generate_qa(context, num_questions=2):
    """Uses GPT-4 to generate MCQs based on input context."""
    client = openai.OpenAI(api_key=openai_api_key)
    prompt = (
        f"Based on the following document content, generate {num_questions} "
        "multiple-choice questions (MCQs) to test understanding. For each question, provide:\n"
        "- The question\n"
        "- Four answer options labeled A, B, C, and D\n"
        "- The correct answer letter (A/B/C/D) and a brief explanation\n\n"
        "Format:\n"
        "Q1. <question>\n"
        "A. Option 1\n"
        "B. Option 2\n"
        "C. Option 3\n"
        "D. Option 4\n"
        "Answer: <letter>\n"
        "Explanation: <short explanation>\n\n"
        f"Document:\n{context}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in educational content generation."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def parse_mcqs(raw_text):
    """Parses raw GPT output into structured question format."""
    pattern = (
        r"Q\d+\.\s*(.*?)\n"
        r"A\.\s*(.*?)\n"
        r"B\.\s*(.*?)\n"
        r"C\.\s*(.*?)\n"
        r"D\.\s*(.*?)\n"
        r"Answer:\s*([A-D])\n"
        r"Explanation:\s*(.*?)(?=\nQ\d+\.|\Z)"
    )

    matches = re.finditer(pattern, raw_text, re.DOTALL)
    questions = []

    for match in matches:
        question_text, a, b, c, d, answer, explanation = match.groups()
        questions.append({
            "question": question_text.strip(),
            "options": {
                "A": a.strip(), "B": b.strip(), "C": c.strip(), "D": d.strip()
            },
            "correct": answer.strip(),
            "explanation": explanation.strip()
        })

    return questions

