
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with OpenRouter settings
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# --- 1. Define a tool ---
def get_weather(city: str):
    """Gets the weather for a given city."""
    if "tokyo" in city.lower():
        return json.dumps({"city": "Tokyo", "temperature": "15 C", "conditions": "Cloudy"})
    elif "san francisco" in city.lower():
        return json.dumps({"city": "San Francisco", "temperature": "12 C", "conditions": "Foggy"})
    elif "paris" in city.lower():
        return json.dumps({"city": "Paris", "temperature": "18 C", "conditions": "Sunny"})
    else:
        return json.dumps({"city": city, "temperature": "unknown", "conditions": "unknown"})

# --- 2. Define the list of tools for the model ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city, e.g., San Francisco",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# --- 3. The Agent Loop ---
def agent_loop():
    """A simple agent loop to demonstrate tool use."""
    
    messages = [{"role": "system", "content": "You are a helpful assistant. When you need to get the current weather, you must use the 'get_weather' tool. For other questions, answer directly."}]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="google/gemini-flash-1.5", # A free model on OpenRouter
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            # --- 4. Handle tool calls ---
            available_functions = {
                "get_weather": get_weather,
            }
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    city=function_args.get("city")
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            
            second_response = client.chat.completions.create(
                model="google/gemini-flash-1.5",
                messages=messages,
            )
            final_response_message = second_response.choices[0].message
            print(f"Assistant: {final_response_message.content}")
            messages.append(final_response_message)
        else:
            # --- 5. Handle direct responses ---
            print(f"Assistant: {response_message.content}")
            messages.append(response_message)

if __name__ == "__main__":
    agent_loop()
