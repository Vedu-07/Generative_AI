from pathlib import Path
from dotenv import load_dotenv
import os


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate

# load dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# loading and splitting pdf
pdf_path = Path("pdfs/React CheatSheet.pdf")
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# create embeddings
embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
)

# store in vector db
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="pdf_rag_chat"
)

# set up AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)

# System prompt
SYSTEM_PROMPT = """
        You are a smart PDF assistant. Answer user queries using only the provided PDF excerpts.

        - For **summaries**, give a brief overview of key points.
        - For **specific questions**, extract and present relevant info directly.
        - For **explanations**, start with a simple overview, then add detail if needed.
        - If the info isn't in the excerpts, reply: "The PDF does not contain this information."

        Be clear, concise, and avoid unnecessary jargon. Structure your answers to match the user's intent.
        If query is unclear ask user to clarify the question once again
        """


# Optional prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "excerpts"],
    template=SYSTEM_PROMPT +
             "\n\nUser Query: {query}\n\nRelevant PDF Excerpts:\n{excerpts}\n\nAssistant:"
)

# Chat loop
while True:
    query = input("Ask a question about the PDF (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    full_prompt = prompt_template.format(query=query, excerpts=context)

    response = llm.invoke(full_prompt)
    print("\nAssistant:", response.content, "\n")