# VA.gov Chatbot

A GPT-4 powered chatbot that can answer questions about VA benefits and services using content from VA.gov.

## Features

- Web scraping of VA.gov content
- Document processing and embedding using OpenAI's text-embedding-3-small
- Vector storage using ChromaDB
- GPT-4 powered chat interface using Streamlit
- Retrieval Augmented Generation (RAG) for accurate, source-backed responses
- Support for multiple LLMs (OpenAI and Claude) with model selection dropdowns

## Prerequisites

1. Install Homebrew (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot-ai
```

2. Set up Python environment using pyenv:

First, install pyenv if you haven't already (macOS):
```bash
brew install pyenv
```

Add pyenv to your shell configuration:
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

Install and set up Python 3.11.8:
```bash
pyenv install 3.11.8
pyenv local 3.11.8
```

3. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

4. Update pip to latest version:
```bash
pip install --upgrade pip
```

5. Install dependencies:

You have two options for installation due to tiktoken requirements:

Option A - With Rust (Recommended for best performance):
```bash
# Install Rust compiler
brew install rust

# Install project dependencies
pip install -r requirements.txt
```

Option B - Using pre-built wheels (Easier but may not be optimized):
```bash
# Edit requirements.txt to use pre-built wheels
# Add these lines at the top of requirements.txt:
# tiktoken>=0.3.3 ; platform_system != "Windows"
# tiktoken>=0.3.3 ; platform_system == "Windows" and python_version >= "3.11"

pip install -r requirements.txt
```

Note: Option A is recommended as it ensures optimal performance and better compatibility with your system. Tiktoken, OpenAI's tokenizer library, uses Rust for high-performance text processing.

6. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_claude_api_key_here
```

## Running the Scraper and Embeddings

1. Run the scraper to collect VA.gov content:
```bash
python scraper.py
```

2. Generate embeddings from the scraped content:
```bash
python embeddings.py
```

## Running the Chatbot

1. Start the Streamlit app:
```bash
streamlit run chatbot.py
```

2. Access the chatbot in your browser at:
   - http://localhost:8501

3. Use the dropdowns to select two LLMs (OpenAI and Claude) and ask questions about VA benefits and services.

Note: The chatbot starts with a fresh chat history each time you run it. Chat history is saved to `chat_history.json` during the session for later evaluation.

## LLM Comparison Dashboard

After running the chatbot and generating chat logs, you can compare and analyze model responses and DeepEval scores using the included dashboard:

1. Start the dashboard app:
   ```bash
   streamlit run compare_llm_results.py
   ```
2. The dashboard will display a table of all chat turns, with columns:
   - `timestamp`, `user utterance`, `model1`, `response1`, `score1`, `model2`, `response2`, `score2`
   - The table uses the full width of the page for easier viewing.
3. You can filter by user utterance, sort by score difference, and view score distributions and best examples.

This makes it easy to compare LLM outputs and evaluation scores side by side.

## LLM Evaluation and Comparison Setup

To enable automatic evaluation and comparison of LLM responses, install the DeepEval library:

```bash
pip install deepeval
```

This is required for the LLM Comparison Dashboard and for logging DeepEval scores in the chatbot.

## Additional Notes

- The chatbot supports multiple LLMs, including OpenAI's GPT-4 and GPT-3.5, and Anthropic's Claude models.
- Each model is tagged as "Cheap" or "Expensive" for clarity.
- The bot displays responses from both selected models side by side.

## Evaluation

The chatbot saves chat history to `chat_history.json` during your session. To run evaluations on the collected conversations:

```bash
python run_evaluation.py
```

This will:
- Load the chat history
- Run various metrics (relevancy, faithfulness, contextual precision, etc.)
- Save results to `chat_eval_log.csv`

## Dashboard

View evaluation results and compare model performance:

```bash
streamlit run dashboard.py
```

The dashboard shows:
- Side-by-side comparison of model responses
- Individual metrics for each response
- Best and worst examples
- Metric distributions
- Ability to sort and filter results

## Project Structure

- `chatbot.py`: Main chatbot interface
- `deep_eval.py`: Deep evaluation logic
- `run_evaluation.py`: Script to run evaluations
- `dashboard.py`: Visualization of evaluation results
- `chat_history.json`: Stored conversations
- `chat_eval_log.csv`: Evaluation metrics