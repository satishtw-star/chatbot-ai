# VA.gov Chatbot

A GPT-4 powered chatbot that can answer questions about VA benefits and services using content from VA.gov.

## Features

- Web scraping of VA.gov content
- Document processing and embedding using OpenAI's text-embedding-3-small
- Vector storage using ChromaDB
- GPT-4 powered chat interface using Streamlit
- Retrieval Augmented Generation (RAG) for accurate, source-backed responses

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