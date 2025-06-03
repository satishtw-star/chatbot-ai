import streamlit as st
from embeddings import DocumentProcessor
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
import openai
import anthropic
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
import csv
from datetime import datetime

load_dotenv()

class VAChatbot:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")  # Load Claude API key
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

    def check_moderation(self, text: str) -> tuple[bool, str]:
        """
        Check if the text violates OpenAI's content policy
        Returns: (is_safe, reason)
        """
        try:
            response = self.openai_client.moderations.create(input=text)
            result = response.results[0]
            
            if result.flagged:
                categories = [cat for cat, flagged in result.categories.items() if flagged]
                return False, f"Content flagged for: {', '.join(categories)}"
            return True, ""
        except Exception as e:
            return False, f"Error checking moderation: {str(e)}"

    def extract_claude_text(self, content):
        # If it's a list of blocks/objects, join all text fields or str representations
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
        # If it's a dict with a 'text' field
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        # If it's a string, just return it
        if isinstance(content, str):
            return content
        # If it's an object (e.g., TextBlock), try to get its 'text' attribute
        if hasattr(content, "text"):
            return str(content.text)
        # Fallback: string representation
        return str(content)

    def get_responses(self, query: str, chat_history: list, model1: str, model2: str) -> tuple[str, str]:
        # Check if the query is appropriate
        is_safe, reason = self.check_moderation(query)
        if not is_safe:
            return f"I apologize, but I cannot process that request. {reason} Please keep your questions focused on VA benefits and services.", f"I apologize, but I cannot process that request. {reason} Please keep your questions focused on VA benefits and services."

        # Check for medical advice requests
        medical_keywords = [
            "diagnose", "diagnosis", "treatment", "symptom", "condition",
            "illness", "disease", "medication", "prescription", "doctor",
            "healthcare", "medical", "therapy", "cure", "heal"
        ]
        if any(keyword in query.lower() for keyword in medical_keywords):
            return """I cannot provide medical advice. For medical questions, please:
1. Contact your VA healthcare provider
2. Visit your nearest VA medical center
3. Use My HealtheVet secure messaging
4. Call 911 for emergencies

I can help you with VA benefits, services, and administrative questions.""", """I cannot provide medical advice. For medical questions, please:
1. Contact your VA healthcare provider
2. Visit your nearest VA medical center
3. Use My HealtheVet secure messaging
4. Call 911 for emergencies

I can help you with VA benefits, services, and administrative questions."""

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
        for message in chat_history:
            if message["role"] == "user":
                # Check user messages in chat history
                is_safe, _ = self.check_moderation(message["content"])
                if not is_safe:
                    continue
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
            response1 = AIMessage(content=self.extract_claude_text(raw_response1.content))

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
            response2 = AIMessage(content=self.extract_claude_text(raw_response2.content))

        # Check if the responses are appropriate
        is_safe1, reason1 = self.check_moderation(response1.content)
        is_safe2, reason2 = self.check_moderation(response2.content)

        # --- DeepEval integration ---
        # Faithfulness metric
        faith_test_case1 = LLMTestCase(input=query, actual_output=response1.content, retrieval_context=[context])
        faith_test_case2 = LLMTestCase(input=query, actual_output=response2.content, retrieval_context=[context])
        faithfulness_metric = FaithfulnessMetric()
        faith1 = faithfulness_metric.measure(faith_test_case1)
        faith2 = faithfulness_metric.measure(faith_test_case2)

        # Contextual Precision
        cprec_test_case1 = LLMTestCase(
            input=query,
            actual_output=response1.content,
            expected_output=response1.content,
            retrieval_context=[context]
        )
        cprec_test_case2 = LLMTestCase(
            input=query,
            actual_output=response2.content,
            expected_output=response2.content,
            retrieval_context=[context]
        )
        contextual_precision_metric = ContextualPrecisionMetric()
        cprec1 = contextual_precision_metric.measure(cprec_test_case1)
        cprec2 = contextual_precision_metric.measure(cprec_test_case2)

        # Contextual Recall
        crec_test_case1 = LLMTestCase(
            input=query,
            actual_output=response1.content,
            expected_output=response1.content,
            retrieval_context=[context]
        )
        crec_test_case2 = LLMTestCase(
            input=query,
            actual_output=response2.content,
            expected_output=response2.content,
            retrieval_context=[context]
        )
        contextual_recall_metric = ContextualRecallMetric()
        crec1 = contextual_recall_metric.measure(crec_test_case1)
        crec2 = contextual_recall_metric.measure(crec_test_case2)

        # Contextual Relevancy
        crel_test_case1 = LLMTestCase(
            input=query,
            actual_output=response1.content,
            expected_output=response1.content,
            retrieval_context=[context]
        )
        crel_test_case2 = LLMTestCase(
            input=query,
            actual_output=response2.content,
            expected_output=response2.content,
            retrieval_context=[context]
        )
        contextual_relevancy_metric = ContextualRelevancyMetric()
        crel1 = contextual_relevancy_metric.measure(crel_test_case1)
        crel2 = contextual_relevancy_metric.measure(crel_test_case2)

        # Answer Relevancy (no retrieval_context/expected_output needed)
        test_case1 = LLMTestCase(input=query, actual_output=response1.content)
        test_case2 = LLMTestCase(input=query, actual_output=response2.content)
        relevancy_metric = AnswerRelevancyMetric()
        score1 = relevancy_metric.measure(test_case1)
        score2 = relevancy_metric.measure(test_case2)

        print(f"[DeepEval] Model 1 ({model1}) relevancy: {score1}, faithfulness: {faith1}, contextual precision: {cprec1}, contextual recall: {crec1}, contextual relevancy: {crel1}")
        print(f"[DeepEval] Model 2 ({model2}) relevancy: {score2}, faithfulness: {faith2}, contextual precision: {cprec2}, contextual recall: {crec2}, contextual relevancy: {crel2}")
        # --- End DeepEval integration ---

        # --- Logging to CSV ---
        log_row = {
            'timestamp': datetime.now().isoformat(),
            'prompt': query,
            'model1': model1,
            'model2': model2,
            'response1': response1.content,
            'response2': response2.content,
            'score1': score1,
            'score2': score2,
            'faithfulness1': faith1,
            'faithfulness2': faith2,
            'contextual_precision1': cprec1,
            'contextual_precision2': cprec2,
            'contextual_recall1': crec1,
            'contextual_recall2': crec2,
            'contextual_relevancy1': crel1,
            'contextual_relevancy2': crel2
        }
        log_file = 'chat_eval_log.csv'
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)
        # --- End Logging ---

        if not is_safe1:
            # Show the actual response for debugging
            return response1.content, response2.content
        if not is_safe2:
            return response1.content, response2.content
        
        return response1.content, response2.content

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
            response1, response2 = st.session_state.chatbot.get_responses(query, st.session_state.chat_history, model1, model2)
            with col1:
                st.markdown(f"**{model1}**")
                st.write(response1)
            with col2:
                st.markdown(f"**{model2}**")
                st.write(response2)

        # Update chat history
        st.session_state.chat_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"Model 1 ({model1}): {response1}\nModel 2 ({model2}): {response2}"}
        ])

if __name__ == "__main__":
    main() 