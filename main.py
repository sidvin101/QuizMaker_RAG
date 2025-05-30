import PyPDF2
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import streamlit as st

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Embedding function
def embed_text(text):
    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Chunking function
def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Store vectors in Pinecone
def store_embeddings(chunks, namespace):
    index = pc.Index(host=os.getenv("PINECONE_INDEX_HOST"))
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

# Generate questions with answers
def generate_qa(context, num_questions=5):
    client = OpenAI(api_key=openai_api_key)
    prompt = (
        f"Based on the following document content, generate {num_questions} "
        "unique question and answer pairs to test understanding. Provide each "
        "question followed by a correct answer.\n\n"
        f"{context}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in educational content generation."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Main Streamlit app
def main():
    st.title("PDF QA Generator")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        # Extract text from PDF
        reader = PyPDF2.PdfReader(uploaded_file)
        full_text = "".join(page.extract_text() for page in reader.pages)

        # Chunk and embed
        chunks = chunk_text(full_text)
        namespace = os.path.splitext(uploaded_file.name)[0]
        store_embeddings(chunks, namespace)

        st.success("PDF processed and embeddings stored successfully.")

        if st.button("Generate Questions"):
            context = " ".join(chunks[:3])
            qa_output = generate_qa(context)
            st.subheader("Generated Questions and Answers")
            st.write(qa_output)

if __name__ == "__main__":
    main()
