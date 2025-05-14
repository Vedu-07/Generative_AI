from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def load_pdf_documents(pdf_file_path):
    loader = PyPDFLoader(file_path=pdf_file_path)
    return loader.load()


def split_into_chunks(documents, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def generate_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )


def store_chunks_in_qdrant(chunks, embedding_model):
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name="pdf_chunks"
    )

def load_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY
    )


SYSTEM_PROMPT = """
You are a smart PDF assistant. Answer user queries using only the provided PDF excerpts.

- For summaries, give a brief overview of key points.
- For specific questions, extract and present relevant info directly.
- For explanations, start with a simple overview, then add detail if needed.
- If the info isn't in the excerpts, reply: "The PDF does not contain this information."

Be clear, concise, and avoid unnecessary jargon. Structure your answers to match the user's intent.
If the query is unclear, ask the user to clarify the question once again.
"""


def create_query_variations(user_query, model, num_variations=3):
    prompt = f"Generate {num_variations} different ways to ask the question: {user_query}"
    response = model.invoke(prompt)
    variations = response.content.split("\n")
    return [user_query] + [v.strip() for v in variations if v.strip()]


def search_chunks_for_all_queries(queries, vector_store, top_k=3):
    all_results = []
    for query in queries:
        docs = vector_store.similarity_search(query, k=top_k)
        all_results.extend(docs)
    return all_results


def remove_duplicate_chunks(documents):
    seen = set()
    unique = []
    for doc in documents:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique


def answer_question(user_query, relevant_chunks, model):
    context_text = "\n\n...\n\n".join([doc.page_content for doc in relevant_chunks])
    full_prompt = SYSTEM_PROMPT + f"\n\nPDF Excerpts:\n{context_text}\n\nUser's Question: {user_query}\n\nAnswer:"
    response = model.invoke(full_prompt)
    return response.content


def ask_pdf_question(user_query, vector_store, chat_model):
    query_versions = create_query_variations(user_query, chat_model)


    print("\n🔁 Query Variations:")
    for idx, q in enumerate(query_versions, 1):
        print(f"{idx}. {q}")

    all_matches = search_chunks_for_all_queries(query_versions, vector_store)
    unique_chunks = remove_duplicate_chunks(all_matches)
    return answer_question(user_query, unique_chunks, chat_model)


def main():
    print("📘 Welcome to the PDF Chat Assistant!")
    pdf_path = Path("data") / "React CheatSheet.pdf"


    documents = load_pdf_documents(pdf_path)
    chunks = split_into_chunks(documents)
    embeddings = generate_embeddings()
    vector_store = store_chunks_in_qdrant(chunks, embeddings)
    chat_model = load_chat_model()


    while True:
        user_input = input("\nAsk something about the PDF (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        if not user_input:
            print("❗ Please enter a valid question.")
            continue
        try:
            response = ask_pdf_question(user_input, vector_store, chat_model)
            print("\n📎 Answer:\n", response)
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
