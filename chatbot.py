import streamlit as st
from embeddings import DocumentProcessor
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
import openai
import anthropic
import csv
from datetime import datetime
import json
import requests

load_dotenv()

class VAChatbot:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        # Define available models with their tags
        self.available_models = {
            "GPT-4 (Expensive)": {"provider": "openai", "model": "gpt-4", "temperature": 0.7},
            "GPT-3.5 (Cheap)": {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.7},
            "Claude-3-Opus (Expensive)": {"provider": "claude", "model": "claude-3-opus-20240229", "temperature": 0.7},
            "Claude-3.7-Sonnet (Cheap)": {"provider": "claude", "model": "claude-3-7-sonnet-20250219", "temperature": 0.7},
            "Azure GPT-4o-mini": {"provider": "azure", "model": "gpt-4o-mini", "temperature": 0.7, "endpoint": "https://eis-omni-devtest-int-southwest.openai.azure.us/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview"}
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

    def check_moderation(self, text: str) -> tuple[bool, str]:
        try:
            response = self.openai_client.moderations.create(input=text)
            result = response.results[0]
            
            if result.flagged:
                categories = [cat for cat, flagged in result.categories.items() if flagged]
                return False, f"Content flagged for: {', '.join(categories)}"
            return True, ""
        except Exception as e:
            print(f"Error during moderation check: {str(e)}")
            return True, f"Moderation check failed: {str(e)}"

    def extract_claude_text(self, content):
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                elif hasattr(block, "text"):
                    texts.append(str(block.text))
                else:
                    texts.append(str(block))
            return "\n".join(texts)
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        if isinstance(content, str):
            return content
        if hasattr(content, "text"):
            return str(content.text)
        return str(content)

    def get_responses(self, query: str, chat_history: list, model1: str, model2: str) -> tuple[str, str, str]:
        # Check for self-harm/suicide keywords (PRIORITY 1)
        self_harm_keywords = [
            "kill myself", "end my life", "suicide", "self-harm", "unalive",
            "want to die", "harm myself", "take my own life", "don't want to live"
        ]
        if any(keyword in query.lower() for keyword in self_harm_keywords):
            crisis_msg = """It sounds like you are going to a difficult time. Please know that there's support available.\n\nHere are some immediate resources:\n\n*   **Veterans Crisis Line:** Call or text 988, then select 1. You can also chat online at [https://www.veteranscrisisline.net/](https://www.veteranscrisisline.net/)\n*   **National Suicide Prevention Lifeline:** Call or text 988\n*   **Crisis Text Line:** Text HOME to 741741\n\nPlease reach out for help. You are not alone."""
            return crisis_msg, crisis_msg, ""

        # Check if the query is appropriate (PRIORITY 2)
        is_safe, reason = self.check_moderation(query)
        if not is_safe:
            error_msg = f"I apologize, but I cannot process that request. {reason} Please keep your questions focused on VA benefits and services."
            return error_msg, error_msg, ""

        # Check for medical advice requests (PRIORITY 3)
        medical_keywords = [
            "diagnose", "diagnosis", "treatment", "symptom", "condition",
            "illness", "disease", "medication", "prescription", "doctor",
            "healthcare", "medical", "therapy", "cure", "heal"
        ]
        if any(keyword in query.lower() for keyword in medical_keywords):
            medical_msg = """I cannot provide medical advice. For medical questions, please:\n1. Contact your VA healthcare provider\n2. Visit your nearest VA medical center\n3. Use My HealtheVet secure messaging\n4. Call 911 for emergencies\n\nI can help you with VA benefits, services, and administrative questions."""
            return medical_msg, medical_msg, ""

        # Get relevant documents
        relevant_docs = self.doc_processor.query_documents(query)
        
        # Create context from relevant documents
        context = "\n\n".join([
            f"Content: {doc['text']}\nSource: {doc['metadata']['url']}"
            for doc in relevant_docs
        ])
        
        # Prepare messages with system prompt and context
        messages = [
            SystemMessage(content=f"{self.system_prompt}\n\nContext:\n{context}"),
        ]
        
        # Add chat history
        for message in chat_history:
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
        elif model1_config["provider"] == "azure":
            # Azure integration for model 1
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_api_key,
            }
            payload = {
                "messages": [
                    {"role": "system", "content": f"{self.system_prompt}\n\nContext:\n{context}"},
                    {"role": "user", "content": query}
                ],
                "temperature": model1_config["temperature"],
                "max_tokens": 1000
            }
            try:
                response1_raw = requests.post(model1_config["endpoint"], headers=headers, json=payload)
                response1_raw.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                response1_json = response1_raw.json()
                response1 = AIMessage(content=response1_json['choices'][0]['message']['content'])
            except requests.exceptions.RequestException as e:
                st.error(f"Azure Model 1 Request Error: {e}")
                return f"Error from Azure Model 1: {e}", f"Error from Azure Model 1: {e}", ""
            except KeyError as e:
                st.error(f"KeyError from Azure Model 1: {e}. Full response: {response1_json}")
                return f"Error processing Azure Model 1 response: {e}", f"Error processing Azure Model 1 response: {e}", ""
        else:
            # Claude integration for model 1
            client = anthropic.Client(api_key=anthropic.api_key)
            raw_response1 = client.messages.create(
                model=model1_config["model"],
                messages=[
                    {"role": "user", "content": query}
                ],
                system=f"{self.system_prompt}\n\nContext:\n{context}", # Pass system prompt as a separate argument
                temperature=model1_config["temperature"],
                max_tokens=1000
            )
            response1 = AIMessage(content=self.extract_claude_text(raw_response1.content))

        # Model 2
        model2_config = self.available_models[model2]
        if model2_config["provider"] == "openai":
            llm2 = ChatOpenAI(model=model2_config["model"], temperature=model2_config["temperature"])
            response2 = llm2.invoke(messages)
        elif model2_config["provider"] == "azure":
            # Azure integration for model 2
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_api_key,
            }
            payload = {
                "messages": [
                    {"role": "system", "content": f"{self.system_prompt}\n\nContext:\n{context}"},
                    {"role": "user", "content": query}
                ],
                "temperature": model2_config["temperature"],
                "max_tokens": 1000
            }
            try:
                response2_raw = requests.post(model2_config["endpoint"], headers=headers, json=payload)
                response2_raw.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                response2_json = response2_raw.json()
                response2 = AIMessage(content=response2_json['choices'][0]['message']['content'])
            except requests.exceptions.RequestException as e:
                st.error(f"Azure Model 2 Request Error: {e}")
                return f"Error from Azure Model 2: {e}", f"Error from Azure Model 2: {e}", ""
            except KeyError as e:
                st.error(f"KeyError from Azure Model 2: {e}. Full response: {response2_json}")
                return f"Error processing Azure Model 2 response: {e}", f"Error processing Azure Model 2 response: {e}", ""
        else:
            # Claude integration
            client = anthropic.Client(api_key=anthropic.api_key)
            raw_response2 = client.messages.create(
                model=model2_config["model"],
                messages=[
                    {"role": "user", "content": query}
                ],
                system=f"{self.system_prompt}\n\nContext:\n{context}", # Pass system prompt as a separate argument
                temperature=model2_config["temperature"],
                max_tokens=1000
            )
            response2 = AIMessage(content=self.extract_claude_text(raw_response2.content))

        return response1.content, response2.content, context

def main():
    st.title("VA.gov Assistant")
    st.write("Ask me anything about VA benefits and services!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
            try:
                response1, response2, context = st.session_state.chatbot.get_responses(query, st.session_state.chat_history, model1, model2)
                with col1:
                    st.markdown(f"**{model1}**")
                    st.write(response1)
                with col2:
                    st.markdown(f"**{model2}**")
                    st.write(response2)
            except Exception as e:
                st.error(f"Error getting responses: {str(e)}")

        # Update chat history with current turn and save to file
        st.session_state.chat_history.append(
            {"timestamp": datetime.now().isoformat(), "role": "user", "content": query, "context": context}
        )
        st.session_state.chat_history.append(
            {"timestamp": datetime.now().isoformat(), "role": "assistant", "content": f"Model 1 ({model1}): {response1}\nModel 2 ({model2}): {response2}", "context": context}
        )

        with open('chat_history.json', 'w') as f:
            json.dump(st.session_state.chat_history, f, indent=2)

if __name__ == "__main__":
    main() 