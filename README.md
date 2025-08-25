# OpenAI Agents SDK Practice

This project is a workspace for practicing and exploring the key features of the OpenAI Assistants API (Agents SDK) using Python, Jupyter Notebooks, and the `uv` package manager.

## Prerequisites

- Python 3.9+
- An OpenAI API key

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd OpenAI-Agents-SDK-Practice
    ```

2.  **Install `uv`:**
    If you don't have `uv` installed, you can install it via `pip`:
    ```bash
    pip install uv
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Set up your API Key:**
    Create a `.env` file by copying the example and then add your OpenAI API key to it.
    ```bash
    cp .env.example .env
    # Now, open .env and add your key
    ```

## How to Use

1.  **Start JupyterLab:**
    ```bash
    jupyter lab
    ```

2.  **Open a notebook:**
    Once JupyterLab opens in your browser, navigate to the `notebooks` directory and open `01_assistant_creation.ipynb` to get started.