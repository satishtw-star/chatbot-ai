import os
from dotenv import load_dotenv
import openai
import anthropic

load_dotenv()

class DeepEvaluator:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

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

    def process_model_responses(self, query: str, response1: str, response2: str, context: str, model1: str, model2: str, chat_history: list) -> tuple[str, str]:
        """
        Process model responses, including moderation checks
        Returns the processed responses
        """
        # Check if the query is appropriate
        is_safe, reason = self.check_moderation(query)
        if not is_safe:
            error_msg = f"I apologize, but I cannot process that request. {reason} Please keep your questions focused on VA benefits and services."
            return error_msg, error_msg

        # Check for medical advice requests
        medical_keywords = [
            "diagnose", "diagnosis", "treatment", "symptom", "condition",
            "illness", "disease", "medication", "prescription", "doctor",
            "healthcare", "medical", "therapy", "cure", "heal"
        ]
        if any(keyword in query.lower() for keyword in medical_keywords):
            medical_msg = """I cannot provide medical advice. For medical questions, please:
1. Contact your VA healthcare provider
2. Visit your nearest VA medical center
3. Use My HealtheVet secure messaging
4. Call 911 for emergencies

I can help you with VA benefits, services, and administrative questions."""
            return medical_msg, medical_msg

        # Check if the responses are appropriate
        is_safe1, _ = self.check_moderation(response1)
        is_safe2, _ = self.check_moderation(response2)

        return response1, response2

    def process_chat_history(self, chat_history: list) -> list:
        """
        Process chat history by checking moderation on user messages
        Returns filtered chat history
        """
        filtered_history = []
        for message in chat_history:
            if message["role"] == "user":
                is_safe, _ = self.check_moderation(message["content"])
                if not is_safe:
                    continue
            filtered_history.append(message)
        return filtered_history

    def process_claude_response(self, content) -> str:
        """
        Process Claude's response and extract text
        """
        return self.extract_claude_text(content) 