import os
import gradio as gr
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled

# --- 1. Load Environment and Initialize LLM Client for OpenRouter ---
load_dotenv()

llm_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# --- 2. Define a Tool as a Standard Python Function ---
@function_tool
def get_capital_city(country: str):
    """Gets the capital city for a given country."""
    print(f"--- Tool called: get_capital_city(country='{country}') ---")
    capitals = {
        "france": "Paris",
        "germany": "Berlin",
        "japan": "Tokyo",
        "usa": "Washington, D.C."
    }
    return capitals.get(country.lower(), "Unknown")

# --- 3. Define the Agent ---
set_tracing_disabled(disabled=True)

assistant = Agent(
    name="FirstAssistant",
    model=OpenAIChatCompletionsModel(model="models/gemini-2.0-flash-lite", openai_client=llm_client),
    instructions="You are a helpful geography tutor. Use your tools to answer questions.",
    tools=[get_capital_city]
)

# --- 4. Define the Prediction Function for Gradio ---
async def predict(message, history):
    """Runs the agent with a given query and returns the result."""
    print(f"> Query: {message}")
    result = await Runner.run(assistant, message)
    print(f"< Assistant: {result.final_output}")
    return result.final_output

# --- 5. Create and Launch the Gradio Interface ---
if __name__ == "__main__":
    ui = gr.ChatInterface(predict,
                          title="Geography Tutor",
                          description="Ask me about the capital cities of France, Germany, Japan, or the USA.",
                          theme="soft")
    ui.launch(inbrowser=True)
