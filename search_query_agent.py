import os
import gradio as gr
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# --- 1. Define the Pydantic output models ---
class ClarificationQuestion(BaseModel):
    question: str = Field(description="A clarifying question to ask the user.")

class ClarificationQuestions(BaseModel):
    questions: List[ClarificationQuestion]

class SearchQueryOption(BaseModel):
    search_query: str = Field(description="A suggested search query.")
    rationale: str = Field(description="The rationale for using this specific query.")

class SearchQuery(BaseModel):
    queries: List[SearchQueryOption]

class OptimizedPrompt(BaseModel):
    prompt: str = Field(description="An optimized prompt.")

# --- 2. Define the Agents ---
set_tracing_disabled(disabled=True)

clarification_agent = Agent(
    name="ClarificationAgent",
    model=OpenAIChatCompletionsModel(model="models/gemini-2.0-flash-lite", openai_client=llm_client),
    instructions="You are an expert in understanding user queries. Your goal is to ask three clarifying questions to help the user provide more context to their query.",
    output_type=ClarificationQuestions,
)

search_query_assistant = Agent(
    name="SearchQueryAssistant",
    model=OpenAIChatCompletionsModel(model="models/gemini-2.0-flash-lite", openai_client=llm_client),
    instructions="""Your helpful assistant who helps on generating queries for the given user query and clarifications. 
Users want to get three different ways of writing their query.
Please provide 3 different ways of writing the search query and the rationale for using each suggested search query.""",
    output_type=SearchQuery,
)

prompt_generator_agent = Agent(
    name="PromptGeneratorAgent",
    model=OpenAIChatCompletionsModel(model="models/gemini-2.0-flash-lite", openai_client=llm_client),
    instructions="""You are an expert in prompt engineering. Your goal is to generate an optimized prompt for the given user query and clarifications.""",
    output_type=OptimizedPrompt,
)

# --- 3. Define the Gradio App ---
state = {"step": "start"}

async def chat_with_assistant(message, history, generator_type):
    global state

    if state["step"] == "start":
        # Get clarifying questions
        clarifications = await Runner.run(clarification_agent, message)
        state["original_query"] = message
        state["questions"] = [q.question for q in clarifications.final_output.questions]
        state["answers"] = []
        state["step"] = "clarify"
        state["current_question_index"] = 0
        state["generator_type"] = generator_type
        return state["questions"][0]

    elif state["step"] == "clarify":
        # Store the user's answer
        state["answers"].append(message)
        state["current_question_index"] += 1

        # If there are more questions, ask the next one
        if state["current_question_index"] < len(state["questions"]):
            return state["questions"][state["current_question_index"]]
        else:
            # All questions answered, generate search queries or prompt
            combined_input = f"Original Query: {state['original_query']}\n\nClarifications:\n"
            for q, a in zip(state["questions"], state["answers"]):
                combined_input += f"- {q}: {a}\n"

            if state["generator_type"] == "Search Query Generator":
                result = await Runner.run(search_query_assistant, combined_input)
                output = ""
                for i, query_option in enumerate(result.final_output.queries):
                    output += f"**Option {i+1}:**\n"
                    output += f"**Search Query:**\n{query_option.search_query}\n\n"
                    output += f"**Rationale:**\n{query_option.rationale}\n\n"
            else:
                result = await Runner.run(prompt_generator_agent, combined_input)
                output = f"**Optimized Prompt:**\n{result.final_output.prompt}"
            
            # Reset the state for the next query
            state = {"step": "start"}
            return output

if __name__ == "__main__":
    with gr.Blocks(theme="soft") as ui:
        gr.Markdown("<h1>Query and Prompt Generator</h1>")
        with gr.Row():
            generator_type = gr.Radio(
                ["Search Query Generator", "Prompt Generator"],
                label="Select Generator Type",
                value="Search Query Generator",
            )
        gr.ChatInterface(
            chat_with_assistant,
            additional_inputs=[generator_type],
        )
    ui.launch(inbrowser=True)   
