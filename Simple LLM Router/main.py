import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def research_model(query):
    return f"[Research Model]: Searching scholarly info on '{query}'..."

def code_model(query):
    return f"[Code Model]: Writing/debugging for '{query}'..."

def image_model(query):
    return f"[Image Model]: Handling image-related request: '{query}'..."

def data_model(query):
    return f"[Data Model]: Analyzing or visualizing data for: '{query}'..."

def chat_model(query):
    return f"[Chat Model]: Responding to general query: '{query}'..."


def gemini_router(query):
    routing_prompt = f"""
        You are an intelligent query router for an AI system. Your task is to decide which domain best matches the user's query. Return only ONE of the following labels (no explanations):

        - research: for academic, scientific, or factual research queries.
        - code: for anything related to programming, debugging, writing code, or technical logic.
        - image: for image generation, image editing, or image descriptions.
        - data: for data analysis, charts, statistics, or anything involving numbers.
        - chat: for general conversation, small talk, or when no category fits well.

        Examples:
        Query: "Generate Python code for binary search"
        Label: code

        Query: "Explain the French Revolution"
        Label: research

        Query: "Visualize sales data from last year"
        Label: data

        Query: "Create a futuristic landscape image"
        Label: image

        Query: "How's the weather today?"
        Label: chat

        Now classify this:
        Query: "{query}"
        Label:"""

    response = model.generate_content(routing_prompt)
    intent = response.text.strip().lower()

    routing_map = {
        "research": (research_model, "Research Model"),
        "code": (code_model, "Code Model"),
        "image": (image_model, "Image Model"),
        "data": (data_model, "Data Model"),
        "chat": (chat_model, "Chat Model")
    }

    return routing_map.get(intent, (chat_model, "Chat Model"))

def main():
    print("🤖 Optimized LLM Router using Gemini 2.0 Flash")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            selected_model, model_name = gemini_router(user_query)
            result = selected_model(user_query)
            print(f"Routed to: {model_name}")
            print(f"Response: {result}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
