from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ConversationCompletenessMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase, LLMTestCaseParams
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
        
        self.call_deflection_metric = GEval(
            name="Call Deflection Effectiveness",
            evaluation_steps=[
                "Does the actual output completely and accurately answer the user's input question, leaving no critical information missing?",
                "Is the actual output clear, concise, and easy for a veteran to understand, without ambiguity or jargon that would require further explanation?",
                "Does the actual output provide all necessary steps, links, or specific instructions for the user to independently resolve their query or access a service without needing to call a VA call center?",
                "If the bot cannot fully resolve the query, does it clearly and correctly direct the user to the *most appropriate* specific VA resource (e.g., a precise VA.gov page, a particular VA department, or a direct contact number) with a clear reason why a call is necessary?",
                "Heavily penalize responses that are vague, incomplete, or suggest calling a general number without attempting to provide specific information or guidance first."
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT
            ],
            threshold=0.7
        )

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
            
            # Create separate test cases for each model's response
            test_case_model1 = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output=response, # Using response for self-consistency
                retrieval_context=[turn_context]
            )
            test_case_model2 = LLMTestCase(
                input=query,
                actual_output=response, # Will be replaced with model2_response later
                expected_output=response, # Using response for self-consistency
                retrieval_context=[turn_context]
            )
            
            # Separate responses from the combined string
            model1_res_str = ""
            model2_res_str = ""
            if "Model 1" in response and "Model 2" in response:
                parts = response.split("Model 2")
                model1_res_str = parts[0].replace("Model 1", "").strip()
                model2_res_str = parts[1].strip()
                test_case_model1.actual_output = model1_res_str
                test_case_model2.actual_output = model2_res_str
            else:
                # Fallback if responses are not in expected combined format
                model1_res_str = response
                model2_res_str = response
                test_case_model1.actual_output = response
                test_case_model2.actual_output = response


            # Run metrics for Model 1
            relevancy_1 = self.relevancy_metric.measure(test_case_model1)
            faithfulness_1 = self.faithfulness_metric.measure(test_case_model1)
            contextual_precision_1 = self.contextual_precision_metric.measure(test_case_model1)
            contextual_recall_1 = self.contextual_recall_metric.measure(test_case_model1)
            contextual_relevancy_1 = self.contextual_relevancy_metric.measure(test_case_model1)
            call_deflection_1 = self.call_deflection_metric.measure(test_case_model1)

            # Run metrics for Model 2
            relevancy_2 = self.relevancy_metric.measure(test_case_model2)
            faithfulness_2 = self.faithfulness_metric.measure(test_case_model2)
            contextual_precision_2 = self.contextual_precision_metric.measure(test_case_model2)
            contextual_recall_2 = self.contextual_recall_metric.measure(test_case_model2)
            contextual_relevancy_2 = self.contextual_relevancy_metric.measure(test_case_model2)
            call_deflection_2 = self.call_deflection_metric.measure(test_case_model2)
            
            # Evaluate conversation completeness (applies to overall conversation, not per model)
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
                'model1_response': model1_res_str, # Store separate responses
                'model2_response': model2_res_str, # Store separate responses
                'relevancy_1': extract_score(relevancy_1),
                'faithfulness_1': extract_score(faithfulness_1),
                'contextual_precision_1': extract_score(contextual_precision_1),
                'contextual_recall_1': extract_score(contextual_recall_1),
                'contextual_relevancy_1': extract_score(contextual_relevancy_1),
                'call_deflection_effectiveness_1': extract_score(call_deflection_1),
                'relevancy_2': extract_score(relevancy_2),
                'faithfulness_2': extract_score(faithfulness_2),
                'contextual_precision_2': extract_score(contextual_precision_2),
                'contextual_recall_2': extract_score(contextual_recall_2),
                'contextual_relevancy_2': extract_score(contextual_relevancy_2),
                'call_deflection_effectiveness_2': extract_score(call_deflection_2),
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