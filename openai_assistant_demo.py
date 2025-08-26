import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled

# --- 1. Load Environment and Initialize LLM Client for OpenRouter ---
# The openai-agents SDK is provider-agnostic, so we can configure
# the standard OpenAI client to point to OpenRouter.
load_dotenv()

llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# --- 2. Define a Tool as a Standard Python Function ---
# The SDK will automatically convert this into a tool schema for the LLM.
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
# An Agent is the core building block. It combines an LLM with instructions and tools.
# We pass the OpenRouter client to the agent.
set_tracing_disabled(disabled=True)

assistant = Agent(
    name="FirstAssistant",
    model=OpenAIChatCompletionsModel(model="google/gemma-3-27b-it:free", openai_client=llm_client), # Specify a free model from OpenRouter
    instructions="You are a helpful geography tutor. Use your tools to answer questions.",
    tools=[get_capital_city] # Pass the Python function directly
)

# --- 4. Run the Agent ---
# The Runner handles the entire execution loop: calling the LLM, invoking tools,
# and getting the final response.
async def run_agent_demo(query: str):
    """Runs the agent with a given query and prints the result."""
    print(f"> Query: {query}")
    # The Runner manages the conversation state automatically.
    result = await Runner.run(assistant, query)
    print(f"< Assistant: {result.final_output}")

async def main():
    print("--- Starting demo using the 'openai-agents' SDK with OpenRouter ---")
    await run_agent_demo("What is the capital of japan?")
    await run_agent_demo("What is the capital of Germany?")
    await run_agent_demo("What is the largest planet in our solar system?")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
