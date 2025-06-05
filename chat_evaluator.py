from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ConversationCompletenessMetric
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
import csv
from datetime import datetime
import os

class ChatEvaluator:
    def __init__(self):
        self.relevancy_metric = AnswerRelevancyMetric()
        self.faithfulness_metric = FaithfulnessMetric()
        self.contextual_precision_metric = ContextualPrecisionMetric()
        self.contextual_recall_metric = ContextualRecallMetric()
        self.contextual_relevancy_metric = ContextualRelevancyMetric()
        self.conversation_completeness_metric = ConversationCompletenessMetric(model='gpt-4-turbo', include_reason=True)

    def evaluate_chat_history(self, chat_history: list, context: str = None):
        """
        Evaluate a chat history using DeepEval metrics
        Args:
            chat_history: List of chat messages
            context: Optional context for evaluation (ignored if per-message context is present)
        """
        results = []
        for i in range(0, len(chat_history)-1, 2):  # Process pairs of messages (user query + assistant response)
            if i+1 >= len(chat_history):
                break
                
            query = chat_history[i]["content"]
            response = chat_history[i+1]["content"]
            # Use context from user message if available, else fallback
            turn_context = chat_history[i].get("context") or context or "No context provided"
            # Create test cases with per-turn context
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output=response,  # Using response as expected output for self-consistency
                retrieval_context=[turn_context]
            )
            
            # Run metrics
            relevancy = self.relevancy_metric.measure(test_case)
            faithfulness = self.faithfulness_metric.measure(test_case)
            contextual_precision = self.contextual_precision_metric.measure(test_case)
            contextual_recall = self.contextual_recall_metric.measure(test_case)
            contextual_relevancy = self.contextual_relevancy_metric.measure(test_case)
            
            # Evaluate conversation completeness
            conversation_test_case = ConversationalTestCase(
                turns=[LLMTestCase(input=msg["content"], actual_output=msg["content"]) 
                      for msg in chat_history[:i+2]]
            )
            conversation_completeness = self.conversation_completeness_metric.measure(conversation_test_case)
            
            # Extract scores
            def extract_score(metric_result):
                if metric_result is None:
                    return None
                if hasattr(metric_result, 'score'):
                    try:
                        return float(metric_result.score)
                    except (ValueError, TypeError):
                        return None
                try:
                    return float(metric_result)
                except (ValueError, TypeError):
                    return None
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'relevancy': extract_score(relevancy),
                'faithfulness': extract_score(faithfulness),
                'contextual_precision': extract_score(contextual_precision),
                'contextual_recall': extract_score(contextual_recall),
                'contextual_relevancy': extract_score(contextual_relevancy),
                'conversation_completeness': extract_score(conversation_completeness)
            }
            results.append(result)
            
            # Log to CSV
            self._log_evaluation_result(result)
            
        return results

    def _log_evaluation_result(self, result: dict):
        """Log evaluation result to CSV file"""
        log_file = 'chat_eval_log.csv'
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result) 