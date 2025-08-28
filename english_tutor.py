import os
import gradio as gr
import enchant
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

class VocabularyQuiz(BaseModel):
    """A vocabulary quiz with multiple questions."""
    title: str = Field(..., description="The title of the quiz.")
    questions: List[VocabularyExercise] = Field(..., description="A list of vocabulary exercises.")

class SpellingQuiz(BaseModel):
    """A spelling quiz with a list of words, some spelled correctly and some incorrectly."""
    words: List[str] = Field(..., description="A list of 5 to 10 words with a mix of correct and incorrect spellings.")
    correct_spellings: List[str] = Field(..., description="A list of the correctly spelled words from the list.")

# Agents for Learning English


GrammarPracticeAgent = Agent(
    name="GrammarPracticeAgent",
    model=model,
    instructions="You are an english tutor who is expert in testing students grammer using grammar quizzes",
    output_type=GrammarQuiz
)

VocabularyPracticeAgent = Agent(name="VocabularyPracticeAgent",
                                model=model,
                                instructions="You are an expert English tutor. Your task is to create a vocabulary quiz with 5 to 10 questions. For each question, provide a sentence with a blank. You must provide four word options: one that is clearly the correct answer, and three that are clearly incorrect, either grammatically or semantically. The goal is to test the user's vocabulary, not their ability to guess from a list of similar words.",
                                output_type=VocabularyQuiz)

SpellingPracticeAgent = Agent(
    name="SpellingPracticeAgent",
    model=model,
    instructions="You are an expert English tutor and linguist. Your task is to create a spelling quiz. For a given difficulty level, create a list of 5 to 10 words. This list MUST contain a mix of correctly and incorrectly spelled words. Each time you are asked for a quiz, you must provide a new and different set of words. Double-check your work to ensure that the `correct_spellings` list is accurate and only contains words that are genuinely spelled correctly from the `words` list.",
    output_type=SpellingQuiz
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

async def get_vocabulary_quiz(level: str):
    try:
        quiz = await Runner.run(VocabularyPracticeAgent, f"Create a {level} level vocabulary quiz.")
        print(quiz)
        return quiz.final_output
    except Exception as e:
        print(f"Error getting vocabulary quiz: {e}")
        return None

async def get_spelling_quiz(level: str):
    try:
        quiz_runner = await Runner.run(SpellingPracticeAgent, f"Create a {level} level spelling quiz.")
        
        # Validate and correct the spelling quiz
        spell_checker = enchant.Dict("en_US")
        validated_correct_spellings = [word for word in quiz_runner.final_output.words if spell_checker.check(word)]
        quiz_runner.final_output.correct_spellings = validated_correct_spellings
        
        print(quiz_runner.final_output)
        return quiz_runner.final_output
    except Exception as e:
        print(f"Error getting spelling quiz: {e}")
        return None

with gr.Blocks() as demo:
    gr.Markdown("# English Language Tutor")

    with gr.Tabs():
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
            vocab_quiz_state = gr.State()
            vocab_level = gr.Radio(["Beginner", "Intermediate", "Advanced"], label="Select Difficulty", value="Beginner")
            start_vocab_quiz = gr.Button("Start Vocabulary Quiz")

            with gr.Column(visible=False) as vocab_quiz_col:
                vocab_quiz_title = gr.Markdown()
                
                vocab_questions = []
                for i in range(10):
                    vocab_questions.append(gr.Markdown(visible=False))
                    vocab_questions.append(gr.Radio(visible=False))

                submit_vocab_button = gr.Button("Submit Answers")
                vocab_results = gr.Markdown()

            def display_vocab_quiz(quiz):
                if quiz is None:
                    return {vocab_quiz_col: gr.update(visible=True), vocab_quiz_title: "Error: Could not generate quiz."}
                
                updates = {vocab_quiz_col: gr.update(visible=True), vocab_quiz_title: f"## {quiz.title}"}
                
                for i, q in enumerate(quiz.questions):
                    updates[vocab_questions[i*2]] = gr.update(value=f"**Question {i+1}:** {q.sentence}", visible=True)
                    updates[vocab_questions[i*2+1]] = gr.update(label="Options", choices=q.word_options, visible=True)

                # Hide unused questions
                for i in range(len(quiz.questions), 10):
                    updates[vocab_questions[i*2]] = gr.update(visible=False)
                    updates[vocab_questions[i*2+1]] = gr.update(visible=False)

                return updates

            def check_vocab_answers(quiz, *answers):
                if quiz is None:
                    return gr.update(value="")
                score = 0
                for i, q in enumerate(quiz.questions):
                    if answers[i] == q.correct_word:
                        score += 1
                return gr.update(value=f"You scored {score} out of {len(quiz.questions)}")

            start_vocab_quiz.click(
                get_vocabulary_quiz,
                inputs=[vocab_level],
                outputs=[vocab_quiz_state]
            ).then(
                display_vocab_quiz,
                inputs=[vocab_quiz_state],
                outputs=[vocab_quiz_col, vocab_quiz_title] + vocab_questions
            )

            submit_vocab_button.click(
                check_vocab_answers,
                inputs=[vocab_quiz_state] + [q for q in vocab_questions if isinstance(q, gr.Radio)],
                outputs=[vocab_results]
            )

        with gr.Tab("Spelling Practice"):
            spelling_quiz_state = gr.State()
            spelling_level = gr.Radio(["Beginner", "Intermediate", "Advanced"], label="Select Difficulty", value="Beginner")
            start_spelling_quiz = gr.Button("Start Spelling Quiz")

            with gr.Column(visible=False) as spelling_quiz_col:
                spelling_quiz_prompt = gr.Markdown("Select the words that are spelled correctly.")
                spelling_words = gr.CheckboxGroup(label="Words")
                submit_spelling_button = gr.Button("Submit Answers")
                spelling_results = gr.Markdown()

            def display_spelling_quiz(quiz):
                if quiz is None:
                    return {spelling_quiz_col: gr.update(visible=True), spelling_words: gr.update(choices=[])}
                return {
                    spelling_quiz_col: gr.update(visible=True),
                    spelling_words: gr.update(choices=quiz.words)
                }

            def check_spelling_answers(quiz, selected_words):
                if quiz is None:
                    return gr.update(value="")
                
                correct_selections = set(selected_words) & set(quiz.correct_spellings)
                
                score = len(correct_selections)
                
                # Penalize for selecting incorrectly spelled words
                for word in selected_words:
                    if word not in quiz.correct_spellings:
                        score -= 1
                
                # Ensure score is not negative
                score = max(0, score)

                return gr.update(value=f"You scored {score} out of {len(quiz.correct_spellings)}. Correct words are: {', '.join(quiz.correct_spellings)}")

            start_spelling_quiz.click(
                get_spelling_quiz,
                inputs=[spelling_level],
                outputs=[spelling_quiz_state]
            ).then(
                lambda: (None, None),
                inputs=None,
                outputs=[spelling_words, spelling_results]
            ).then(
                display_spelling_quiz,
                inputs=[spelling_quiz_state],
                outputs=[spelling_quiz_col, spelling_words]
            )

            submit_spelling_button.click(
                check_spelling_answers,
                inputs=[spelling_quiz_state, spelling_words],
                outputs=[spelling_results]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
