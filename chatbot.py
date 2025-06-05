import streamlit as st
from embeddings import DocumentProcessor
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
import openai
import anthropic
from deep_eval import DeepEvaluator
import csv
from datetime import datetime
import json

load_dotenv()

class VAChatbot:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")  # Load Claude API key
        self.deep_evaluator = DeepEvaluator()
        # Define available models with their tags
        self.available_models = {
            "GPT-4 (Expensive)": {"provider": "openai", "model": "gpt-4", "temperature": 0.7},
            "GPT-3.5 (Cheap)": {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.7},
            "Claude-3-Opus (Expensive)": {"provider": "claude", "model": "claude-3-opus-20240229", "temperature": 0.7},
            "Claude-3.7-Sonnet (Cheap)": {"provider": "claude", "model": "claude-3-7-sonnet-20250219", "temperature": 0.7}
        }
        self.system_prompt = """You are a helpful VA assistant that answers questions about VA benefits and services using information from VA.gov.
        Always use the provided context from VA.gov to answer questions accurately. If the context contains specific information about VA.gov processes or requirements, use that information.
        Do not use any content or reasoning that is not provided to you. 
        
        SOURCE CITATION REQUIREMENTS:
        - You MUST cite the source URL for EVERY piece of information you provide
        - Format citations as: [Source: URL]
        - If you use information from multiple sources, cite each one separately
        - If you cannot find relevant information in the provided context, say so and do not make up information
        
        IMPORTANT MEDICAL DISCLAIMER:
        - Do not provide any medical advice, diagnoses, or treatment recommendations
        - Do not interpret medical conditions or symptoms
        - For medical questions, always direct users to:
          * Their VA healthcare provider
          * The nearest VA medical center
          * The My HealtheVet secure messaging system
          * Emergency services (911) for urgent medical needs
        
        For login issues, follow this structure:
        1. First, ask clarifying questions to understand the specific issue:
           - Which sign-in method are you using? (Login.gov, ID.me, or DS Logon)
           - What specific error message are you seeing?
           - Have you been able to sign in before?
           - Are you trying to create a new account or access an existing one?
        
        2. Based on their response, explain the available sign-in options and provide specific troubleshooting steps:
           - For Login.gov/ID.me issues:
             * Verify identity requirements
             * Check for common error messages
             * Try clearing browser cache/cookies
           - For DS Logon issues:
             * Note that it's available through September 30, 2025
             * Provide DS Logon-specific troubleshooting
        
        3. Finally, provide support resources:
           - MyVA411 support line (800-698-2411)
           - Links to relevant VA.gov resources
           - When to contact VA support
        
        Always cite the source URL when providing information.
        If you're not sure about something, ask for more details before providing guidance.
        If a question appears to be seeking medical advice, respond with the medical disclaimer and direct them to appropriate VA healthcare resources."""

    def get_responses(self, query: str, chat_history: list, model1: str, model2: str) -> tuple[str, str, str]:
        # Get relevant documents
        relevant_docs = self.doc_processor.query_documents(query)
        
        # Create context from relevant documents
        context = "\n\n".join([
            f"Content: {doc['text']}\nSource: {doc['metadata']['url']}"
            for doc in relevant_docs
        ])
        
        # Prepare messages
        messages = [
            SystemMessage(content=f"{self.system_prompt}\n\nContext:\n{context}"),
        ]
        
        # Add chat history
        filtered_history = self.deep_evaluator.process_chat_history(chat_history)
        for message in filtered_history:
            messages.append(
                HumanMessage(content=message["content"]) if message["role"] == "user"
                else AIMessage(content=message["content"])
            )
                
        # Add current query
        messages.append(HumanMessage(content=query))
        
        # Model 1
        model1_config = self.available_models[model1]
        if model1_config["provider"] == "openai":
            llm1 = ChatOpenAI(model=model1_config["model"], temperature=model1_config["temperature"])
            response1 = llm1.invoke(messages)
        else:
            # Claude integration for model 1
            client = anthropic.Client(api_key=anthropic.api_key)
            raw_response1 = client.messages.create(
                model=model1_config["model"],
                messages=[{"role": "user", "content": query}],
                temperature=model1_config["temperature"],
                max_tokens=1000
            )
            response1 = AIMessage(content=self.deep_evaluator.process_claude_response(raw_response1.content))

        # Model 2
        model2_config = self.available_models[model2]
        if model2_config["provider"] == "openai":
            llm2 = ChatOpenAI(model=model2_config["model"], temperature=model2_config["temperature"])
            response2 = llm2.invoke(messages)
        else:
            # Claude integration
            client = anthropic.Client(api_key=anthropic.api_key)
            raw_response2 = client.messages.create(
                model=model2_config["model"],
                messages=[{"role": "user", "content": query}],
                temperature=model2_config["temperature"],
                max_tokens=1000
            )
            response2 = AIMessage(content=self.deep_evaluator.process_claude_response(raw_response2.content))

        # Process responses
        processed1, processed2 = self.deep_evaluator.process_model_responses(
            query=query,
            response1=response1.content,
            response2=response2.content,
            context=context,
            model1=model1,
            model2=model2,
            chat_history=filtered_history
        )
        return processed1, processed2, context

def main():
    st.title("VA.gov Assistant")
    st.write("Ask me anything about VA benefits and services!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Load existing chat history if available
        if os.path.exists('chat_history.json'):
            with open('chat_history.json', 'r') as f:
                st.session_state.chat_history = json.load(f)

    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = VAChatbot()

    # Model selection dropdowns
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Select Model 1", list(st.session_state.chatbot.available_models.keys()), index=1)  # Default to GPT-3.5 (Cheap)
    with col2:
        model2 = st.selectbox("Select Model 2", list(st.session_state.chatbot.available_models.keys()), index=3)  # Default to Claude-3-Sonnet (Cheap)

    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if query := st.chat_input("Type your question here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(query)

        # Get and display assistant responses
        with st.chat_message("assistant"):
            col1, col2 = st.columns(2)
            response1, response2, context = st.session_state.chatbot.get_responses(query, st.session_state.chat_history, model1, model2)
            with col1:
                st.markdown(f"**{model1}**")
                st.write(response1)
            with col2:
                st.markdown(f"**{model2}**")
                st.write(response2)

        # Update chat history
        st.session_state.chat_history.extend([
            {"role": "user", "content": query, "context": context},
            {"role": "assistant", "content": f"Model 1 ({model1}): {response1}\nModel 2 ({model2}): {response2}", "context": context}
        ])
        
        # Save chat history to file
        with open('chat_history.json', 'w') as f:
            json.dump(st.session_state.chat_history, f, indent=2)

if __name__ == "__main__":
    main() 