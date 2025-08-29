import os
import gradio as gr
import enchant
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from pydantic import BaseModel, Field
from typing import List
import contextlib
import io
import traceback

#this python script is to generate the code for the given requirements. this module contains LLM agents.
#one LLM agent generates only the python code for the given requirements. then, generated python code is passed to another agent.
#second agent reviews the code whether all the requirements are satisfied or not. 
# also, second agent covers all the aspects of a software like performance, security, test case coverage etc. 
# this module uses agents library and Gemini LLM. gemini llm model is configured in the .env file. 
# the code review feedback from second agent must be passed to first agent for the code improvements. 
# this back and forth conversation must not continue more than 3 times. 
# if there are any pending enhancements at the endof the final review, then those must the provided to user for manual review. 
#agents must use handoffs, sessions, tools if requried. 

load_dotenv()

llm_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key=os.getenv("GEMINI_API_KEY"),
    )

# Create a LLM model for the agents
model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-latest")
model = OpenAIChatCompletionsModel(
    model=model_name,
    openai_client=llm_client,
)

class CodeGenerationRequest(BaseModel):
    """A request to generate code."""
    requirements: str = Field(description="The requirements for the code to be generated.")
    language: str = Field(description="The programming language to use.")

class CodeReviewRequest(BaseModel):
    """A request to review code."""
    code: str = Field(description="The code to be reviewed.")

class CodeReview(BaseModel):
    """A review of the code."""
    feedback: str = Field(description="The feedback on the code.")
    satisfied: bool = Field(description="Whether the code satisfies all requirements.")

class FinalCode(BaseModel):
    """The final code and any pending enhancements."""
    code: str = Field(description="The final generated code.")
    pending_enhancements: str = Field(description="Any pending enhancements to be manually reviewed.")

class CodeSafetyRequest(BaseModel):
    """A request to check the safety of the code."""
    code: str = Field(description="The code to be checked.")

class CodeSafetyResponse(BaseModel):
    """The response from the code safety check."""
    safe: bool = Field(description="Whether the code is safe to execute.")
    reason: str = Field(description="The reason why the code is safe or not.")

coder_agent = Agent(
    name="Coder",
    instructions="You are a coding expert. Generate code in the requested language.",
    model=model,
    output_type=CodeReviewRequest,
)

reviewer_agent = Agent(
    name="Reviewer",
    instructions="You are a code review expert. Review the given Python code for requirement satisfaction, performance, security, and test case coverage.",
    model=model,
    output_type=CodeReview,
)

code_safety_agent = Agent(
    name="CodeSafetyInspector",
    instructions="You are a strict code safety expert. Your only purpose is to inspect Python code for any unsafe operations. You must reject any code that uses modules like 'os', 'sys', 'subprocess', 'shutil', or functions like 'open', 'eval', 'exec'. You must also reject any code that attempts to access the file system, make network requests, or execute shell commands. Only approve code that is absolutely safe for execution in a restricted environment.",
    model=model,
    output_type=CodeSafetyResponse,
)

async def generate_code_and_review(requirements: str, language: str):
    """
    Generate code based on the given requirements using a Coder and a Reviewer agent.
    """
    coder_instructions = f"You are a coding expert. Generate code in {language} for the following requirements: {requirements}"
    code_review_request = await Runner.run(coder_agent, coder_instructions)
    
    for _ in range(3):
        code_review = await Runner.run(reviewer_agent, code_review_request.final_output.code)
        if code_review.final_output.satisfied:
            break
        
        # Provide the feedback to the coder agent to improve the code
        code_review_request = await Runner.run(
            coder_agent,
            f"The following is a review of the code you generated:\n{code_review.final_output.feedback}\nPlease improve the code based on this feedback."
        )

    return code_review_request.final_output.code, code_review.final_output.feedback if not code_review.final_output.satisfied else ""

async def run_code(code: str, user_input: str):
    """
    Run the given code with the given user input after checking for safety.
    """
    print(f"Generated code:\n{code}")
    safety_check = await Runner.run(code_safety_agent, code)
    if not safety_check.final_output.safe:
        return f"Code is not safe to execute: {safety_check.final_output.reason}"

    try:
        # Create a restricted environment to execute the code
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "sum": sum,
                "min": min,
                "max": max,
                "__import__": __import__
            }
        }
        
        # Capture the output of the code
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, restricted_globals)
            # Assuming the function to be tested is the last one defined in the code
            func_name = [name for name, obj in restricted_globals.items() if callable(obj)][-1]
            func = restricted_globals[func_name]
            result = func(user_input)
            print(result)
            
        return output.getvalue()
    except Exception as e:
        return f"Error executing code: {e}\n{traceback.format_exc()}"

def handle_language_change(language):
    if language == "Python":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

with gr.Blocks() as demo:
    gr.Markdown("## AI Code Generator and Tester")
    with gr.Row():
        with gr.Column():
            language_dropdown = gr.Dropdown(["Python", "Java", "JavaScript", "C++"], label="Select Language", value="Python")
            requirements_input = gr.Textbox(lines=5, label="Enter your requirements here")
            generate_button = gr.Button("Generate Code")
            generated_code = gr.Code(label="Generated Code", language="python")
            pending_enhancements = gr.Textbox(label="Pending Enhancements", interactive=False)
        with gr.Column():
            user_input = gr.Textbox(lines=2, label="Enter the string input for the generated function", visible=True)
            run_button = gr.Button("Run Code", visible=True)
            execution_output = gr.Textbox(label="Execution Output", interactive=False, visible=True)

    language_dropdown.change(
        handle_language_change,
        inputs=language_dropdown,
        outputs=[user_input, run_button, execution_output]
    )

    generate_button.click(
        generate_code_and_review,
        inputs=[requirements_input, language_dropdown],
        outputs=[generated_code, pending_enhancements]
    )

    run_button.click(
        run_code,
        inputs=[generated_code, user_input],
        outputs=execution_output
    )

demo.launch()