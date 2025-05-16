from pathlib import Path
import os
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


SYSTEM_PROMPT = """
You are a smart assistant that answers questions using PDF excerpts. Use Chain of Thought reasoning explicitly.

Follow this structure:
- Step 1: Identify key information from the excerpts.
- Step 2: Analyze how it connects to the user’s question.
- Step 3: Derive the final answer logically and clearly.

If the info is not available in the excerpts, reply: "The PDF does not contain this information."

Always show these 3 steps followed by the final answer in a clearly labeled section.
"""



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



def retrieve_documents(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)

def construct_cot_prompt(query, context):
    return (
        SYSTEM_PROMPT + "\n\n"
        "Excerpts:\n" + context + "\n\n"
        f"Question: {query}\n\n"
        "Step-by-step reasoning:\n"
        "Step 1: Identify the key information in the excerpts related to the question.\n"
        "Step 2: Analyze how this information applies to the question.\n"
        "Step 3: Formulate a clear, concise answer based on the analysis.\n\n"
        "Final Answer:"
    )

def chat_with_cot(query, vector_store, model):
    retrieved_docs = retrieve_documents(vector_store, query)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    cot_prompt = construct_cot_prompt(query, context)
    response = model.invoke(cot_prompt)
    return response.content



def main():
    print("📘 Welcome to the Chain of Thought PDF Assistant!")

    pdf_path = Path("data") / "React CheatSheet.pdf"
    if not pdf_path.exists():
        print(f"❗ PDF not found at {pdf_path}")
        return

    print("📄 Loading PDF...")
    documents = load_pdf_documents(pdf_path)

    print("🔪 Splitting PDF into chunks...")
    chunks = split_into_chunks(documents)

    print("🧠 Generating embeddings...")
    embeddings = generate_embeddings()

    print("📥 Storing chunks into Qdrant...")
    vector_store = store_chunks_in_qdrant(chunks, embeddings)

    print("💬 Loading language model...")
    llm = load_chat_model()

    print("✅ Ready to answer using Chain of Thought reasoning!")

    while True:
        query = input("\n🔎 Ask a question (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        if not query:
            print("❗ Please enter a valid question.")
            continue

        try:
            print("\n🧠 Thinking step-by-step...\n")
            answer = chat_with_cot(query, vector_store, llm)
            print("📎 Response:\n")
            print(answer)
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
