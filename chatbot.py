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
        
        self.system_prompt = """You are a helpful VA assistant that answers questions about VA benefits and services. 
        Use the provided context to answer questions accurately. If you're not sure about something, say so.
        Always cite the source URL when providing information."""

    def get_response(self, query: str, chat_history: list) -> str:
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
                messages.append(HumanMessage(content=message["content"]))
            else:
                messages.append(AIMessage(content=message["content"]))
                
        # Add current query
        messages.append(HumanMessage(content=query))
        
        # Get response
        response = self.llm.invoke(messages)
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