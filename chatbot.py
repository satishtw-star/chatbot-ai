import streamlit as st
from embeddings import DocumentProcessor
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
import openai

load_dotenv()

class VAChatbot:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
        )
        
        self.system_prompt = """You are a helpful VA assistant that answers questions about VA benefits and services using information from VA.gov.
        Always use the provided context from VA.gov to answer questions accurately. If the context contains specific information about VA.gov processes or requirements, use that information.
        
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
            response = openai.Moderation.create(input=text)
            result = response.results[0]
            
            if result.flagged:
                categories = [cat for cat, flagged in result.categories.items() if flagged]
                return False, f"Content flagged for: {', '.join(categories)}"
            return True, ""
        except Exception as e:
            return False, f"Error checking moderation: {str(e)}"

    def get_response(self, query: str, chat_history: list) -> str:
        # Check if the query is appropriate
        is_safe, reason = self.check_moderation(query)
        if not is_safe:
            return f"I apologize, but I cannot process that request. {reason} Please keep your questions focused on VA benefits and services."

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
        
        # Get response
        response = self.llm.invoke(messages)
        
        # Check if the response is appropriate
        is_safe, reason = self.check_moderation(response.content)
        if not is_safe:
            return "I apologize, but I cannot provide that response. Please try rephrasing your question about VA benefits and services."
            
        return response.content

def main():
    st.title("VA.gov Assistant")
    st.write("Ask me anything about VA benefits and services!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = VAChatbot()

    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if query := st.chat_input("Type your question here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(query)

        # Get and display assistant response
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.get_response(
                query, 
                st.session_state.chat_history
            )
            st.write(response)

        # Update chat history
        st.session_state.chat_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

if __name__ == "__main__":
    main() 