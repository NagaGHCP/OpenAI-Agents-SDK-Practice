import os
import gradio as gr
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from pydantic import BaseModel, Field
from typing import List

#this python script is to build LLM agents which helps users in practicing english grammer, vocabulary, spellings. 
#want to create dedicated agents for each learning task like grammer practice, vocabulary, spellings. 
#learning tasks are in the form of quizes or fill inthe blanks format. 
#agents shall ask question and validate the user response and ask next question. at the end of the test, agent shall return the quiz results. 
#this app shall be build using gradio, OpenAI Agents SDK framework.

load_dotenv()

llm_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key=os.getenv("GEMINI_API_KEY"),
    )

# Create a runner for the agents
model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-latest")
model = OpenAIChatCompletionsModel(
    model=model_name,
    openai_client=llm_client,
)


# Pydantic Models for Agent Outputs



class QuizQuestion(BaseModel):
    """A single quiz question with multiple choice options."""
    question: str = Field(..., description="The quiz question.")
    options: List[str] = Field(..., description="A list of multiple choice options.")
    correct_answer: str = Field(..., description="The correct answer from the options.")

class GrammarQuiz(BaseModel):
    """A full grammar quiz with multiple questions."""
    title: str = Field(..., description="The title of the quiz.")
    questions: List[QuizQuestion] = Field(..., description="A list of grammar questions.")

class VocabularyExercise(BaseModel):
    """A vocabulary exercise, like fill-in-the-blanks."""
    sentence: str = Field(..., description="A sentence with a missing word (indicated by a blank).")
    word_options: List[str] = Field(..., description="A list of words to choose from to fill the blank.")
    correct_word: str = Field(..., description="The correct word for the blank.")

class SpellingTest(BaseModel):
    """A spelling test with a word to spell."""
    word_to_spell: str = Field(..., description="The word for the user to spell.")
    correct_spelling: str = Field(..., description="The correct spelling of the word.")

# Agents for Learning English


GrammarPracticeAgent = Agent(
    name="GrammarPracticeAgent",
    model=model,
    instructions="You are an english tutor who is expert in testing students grammer using grammar quizzes",
    output_type=GrammarQuiz
)

VocabularyPracticeAgent = Agent(name="VocabularyPracticeAgent",
                                model=model,
                                instructions="You are an english tutor who is expert in testing students vocabulary using vocabulary exercises",
                                output_type=VocabularyExercise)

SpellingPracticeAgent = Agent(
    name="SpellingPracticeAgent",
    model=model,
    instructions="You are an english tutor who is expert in testing students spelling using spelling tests.",
    output_type=SpellingTest
)


#grammar_agent = GrammarPracticeAgent(model)
#vocabulary_agent = VocabularyPracticeAgent(model)
#spelling_agent = SpellingPracticeAgent(model)

# Gradio Interface
async def get_grammar_quiz(level: str):
    try:
        quiz = await Runner.run(GrammarPracticeAgent, f"Create a {level} level grammar quiz with 3 random questions.")
        print(quiz)
        return quiz.final_output
    except Exception as e:
        print(f"Error getting grammar quiz: {e}")
        return None

async def get_vocabulary_exercise(level: str):
    try:
        exercise = await Runner.run(VocabularyPracticeAgent, f"Create a {level} level vocabulary random exercise.")
        print(exercise)
        return exercise.final_output
    except Exception as e:
        print(f"Error getting vocabulary exercise: {e}")
        return None

async def get_spelling_test(level: str):
    try:
        test = await Runner.run(SpellingPracticeAgent, f"Create a {level} level random spelling test.")
        print(test)
        return test.final_output
    except Exception as e:
        print(f"Error getting spelling test: {e}")
        return None

with gr.Blocks() as demo:
    gr.Markdown("# English Language Tutor")

    with gr.Tab("Grammar Practice"):
        grammar_quiz_state = gr.State()
        grammar_level = gr.Radio(["Beginner", "Intermediate", "Advanced"], label="Select Difficulty", value="Beginner")
        start_grammar_quiz = gr.Button("Start Grammar Quiz")
        
        with gr.Column(visible=False) as grammar_quiz_col:
            grammar_quiz_title = gr.Markdown()
            q1 = gr.Radio(visible=False)
            q2 = gr.Radio(visible=False)
            q3 = gr.Radio(visible=False)
            question_radios = [q1, q2, q3]
            submit_grammar_button = gr.Button("Submit Answers")
            grammar_results = gr.Markdown()

        def display_grammar_quiz(quiz):
            if quiz is None:
                return {grammar_quiz_col: gr.update(visible=True), grammar_quiz_title: "Error: Could not generate quiz."}
            updates = {grammar_quiz_col: gr.update(visible=True), grammar_quiz_title: f"## {quiz.title}"}
            for i, q in enumerate(quiz.questions):
                updates[question_radios[i]] = gr.update(label=q.question, choices=q.options, visible=True)
            return updates

        def check_grammar_answers(quiz, ans1, ans2, ans3):
            if quiz is None:
                return gr.update(value="")
            answers = [ans1, ans2, ans3]
            score = 0
            for i, q in enumerate(quiz.questions):
                if answers[i] == q.correct_answer:
                    score += 1
            return gr.update(value=f"You scored {score} out of {len(quiz.questions)}")

        start_grammar_quiz.click(
            get_grammar_quiz, 
            inputs=[grammar_level], 
            outputs=[grammar_quiz_state]
        ).then(
            display_grammar_quiz,
            inputs=[grammar_quiz_state],
            outputs=[grammar_quiz_col, grammar_quiz_title] + question_radios
        )

        submit_grammar_button.click(
            check_grammar_answers,
            inputs=[grammar_quiz_state] + question_radios,
            outputs=[grammar_results]
        )

    with gr.Tab("Vocabulary Builder"):
        vocab_exercise_state = gr.State()
        vocab_level = gr.Radio(["Beginner", "Intermediate", "Advanced"], label="Select Difficulty", value="Beginner")
        start_vocab_exercise = gr.Button("Start Vocabulary Exercise")

        with gr.Column(visible=False) as vocab_exercise_col:
            vocab_sentence = gr.Markdown()
            vocab_options = gr.Radio(label="Choose the correct word")
            check_vocab_button = gr.Button("Check Answer")
            vocab_result = gr.Markdown()

        def display_vocab_exercise(exercise):
            if exercise is None:
                return {vocab_exercise_col: gr.update(visible=True), vocab_sentence: "Error: Could not generate exercise."}
            return {
                vocab_exercise_col: gr.update(visible=True),
                vocab_sentence: exercise.sentence,
                vocab_options: gr.update(choices=exercise.word_options, label="Choose the correct word")
            }

        def check_vocab_answer(exercise, answer):
            if exercise is None:
                return gr.update(value="")
            if answer == exercise.correct_word:
                return gr.update(value="Correct!")
            else:
                return gr.update(value=f"Incorrect. The correct answer is {exercise.correct_word}")

        start_vocab_exercise.click(
            get_vocabulary_exercise,
            inputs=[vocab_level],
            outputs=[vocab_exercise_state]
        ).then(
            display_vocab_exercise,
            inputs=[vocab_exercise_state],
            outputs=[vocab_exercise_col, vocab_sentence, vocab_options]
        )

        check_vocab_button.click(
            check_vocab_answer,
            inputs=[vocab_exercise_state, vocab_options],
            outputs=[vocab_result]
        )

    with gr.Tab("Spelling Practice"):
        spelling_test_state = gr.State()
        spelling_level = gr.Radio(["Beginner", "Intermediate", "Advanced"], label="Select Difficulty", value="Beginner")
        start_spelling_test = gr.Button("Start Spelling Test")

        with gr.Column(visible=False) as spelling_test_col:
            word_to_spell_output = gr.Markdown()
            user_spelling_input = gr.Textbox(label="Your Spelling")
            check_spelling_button = gr.Button("Check Spelling")
            spelling_result = gr.Markdown()

        def display_spelling_test(test):
            if test is None:
                return {spelling_test_col: gr.update(visible=True), word_to_spell_output: "Error: Could not generate test."}
            return {
                spelling_test_col: gr.update(visible=True),
                word_to_spell_output: f"Spell the word: **{test.word_to_spell}**"
            }

        def check_spelling(test, user_spelling):
            if test is None:
                return gr.update(value="")
            if user_spelling.lower() == test.correct_spelling.lower():
                return gr.update(value="Correct!")
            else:
                return gr.update(value=f"Incorrect. The correct spelling is **{test.correct_spelling}**")

        start_spelling_test.click(
            get_spelling_test,
            inputs=[spelling_level],
            outputs=[spelling_test_state]
        ).then(
            display_spelling_test,
            inputs=[spelling_test_state],
            outputs=[spelling_test_col, word_to_spell_output]
        )

        check_spelling_button.click(
            check_spelling,
            inputs=[spelling_test_state, user_spelling_input],
            outputs=[spelling_result]
        )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
