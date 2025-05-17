from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
import os

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_FILENAME = "data/React CheatSheet.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "react_pdf_hyde"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = """
You are a smart PDF assistant designed to help users understand a PDF document’s content. Your task is to provide accurate, clear, and concise responses based on the user’s query and the provided PDF excerpts. Follow these guidelines:

1. Query Handling:
   - For specific queries, extract relevant information directly.
   - For general queries, provide a concise overview.

2. Use Excerpts Only:
   - Base your response solely on the provided excerpts.
   - If the info isn’t there, say: "The PDF does not contain this information."

3. Response Style:
   - Use simple, clear language.
   - Provide a direct answer followed by brief reasoning or context if necessary.

If the query is unclear, ask for clarification.
"""


def load_pdf_document(pdf_filename):
    pdf_path = Path(__file__).parent / pdf_filename
    loader = PyPDFLoader(file_path=pdf_path)
    return loader.load()


def split_pdf_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )


def create_vector_store(split_docs, embedder):
    return QdrantVectorStore.from_documents(
        documents=split_docs,
        embedding=embedder,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )


def get_language_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY
    )


def generate_hypothetical_answer(query, llm):
    hypo_prompt = f"Generate a short, hypothetical answer to the question: {query}"
    response = llm.invoke(hypo_prompt)
    return response.content.strip()


def embed_hypothetical_answer(text, embedder):
    return embedder.embed_query(text)


def get_similar_chunks(vector_store, embedding, k=5):
    return vector_store.similarity_search_by_vector(embedding, k=k)


def build_final_prompt(query, context):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Based on the following PDF excerpts, answer the question.\n\n"
        "Excerpts:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Assistant:"
    )


def generate_final_response(query, vector_store, llm, embedder):
    print(f"\n[Question] {query}\n")
    
    hypothetical = generate_hypothetical_answer(query, llm)
    print("[Hypothetical Answer]")
    print(hypothetical + "\n")

    hypo_embedding = embed_hypothetical_answer(hypothetical, embedder)
    similar_docs = get_similar_chunks(vector_store, hypo_embedding)
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    prompt = build_final_prompt(query, context)
    response = llm.invoke(prompt)

    print("[Answer]")
    return response.content.strip()


def main():
    try:
        print("[Loading PDF and preparing environment...]\n")
        docs = load_pdf_document(PDF_FILENAME)
        split_docs = split_pdf_into_chunks(docs)
        embedder = get_embedder()
        vector_store = create_vector_store(split_docs, embedder)
        llm = get_language_model()

        print("[READY] PDF Assistant using HyDE is ready!\n")

        while True:
            query = input("Ask a question about the PDF (type 'exit' to quit): ")
            if query.lower() == 'exit':
                print("\nGoodbye!")
                break
            if not query.strip():
                print("Please enter a valid question.\n")
                continue

            answer = generate_final_response(query, vector_store, llm, embedder)
            print(answer)
            print("-" * 80)

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
