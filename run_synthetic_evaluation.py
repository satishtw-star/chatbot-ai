import json
import csv
import os
from datetime import datetime
from chatbot import VAChatbot
from chat_evaluator import ChatEvaluator
from deepeval.test_case import LLMTestCase, ConversationalTestCase

# Load synthetic dataset
with open('synthetic_va_dataset.json', 'r') as f:
    synthetic_data = json.load(f)

# Initialize chatbot and evaluator
chatbot = VAChatbot()
evaluator = ChatEvaluator()

# Model selection
print("Available models:")
for i, model_name in enumerate(chatbot.available_models.keys()):
    print(f"{i+1}. {model_name}")
model_idx = int(input("Select a model by number: ")) - 1
model_name = list(chatbot.available_models.keys())[model_idx]
print(f"Using model: {model_name}")

results = []

for i, example in enumerate(synthetic_data):
    query = example['input']
    expected_output = example['expected_output']
    context = example['context'][0] if example['context'] else ""
    # Generate model response
    response, _, _ = chatbot.get_responses(query, [], model_name, model_name)
    # Prepare test case for evaluation (model1)
    test_case_model1 = LLMTestCase(
        input=query,
        actual_output=response,
        expected_output=expected_output,
        retrieval_context=[context]
    )
    # Calculate all metrics for model1
    relevancy_1 = evaluator.relevancy_metric.measure(test_case_model1)
    faithfulness_1 = evaluator.faithfulness_metric.measure(test_case_model1)
    contextual_precision_1 = evaluator.contextual_precision_metric.measure(test_case_model1)
    contextual_recall_1 = evaluator.contextual_recall_metric.measure(test_case_model1)
    contextual_relevancy_1 = evaluator.contextual_relevancy_metric.measure(test_case_model1)
    call_deflection_1 = evaluator.call_deflection_metric.measure(test_case_model1)
    # Conversation completeness (single-turn, so just this turn)
    conversation_completeness = None
    try:
        conversation_completeness = evaluator.conversation_completeness_metric.measure(
            ConversationalTestCase(turns=[test_case_model1])
        )
    except Exception:
        conversation_completeness = None
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
        'model1_response': response,
        'model2_response': '',
        'relevancy_1': extract_score(relevancy_1),
        'faithfulness_1': extract_score(faithfulness_1),
        'contextual_precision_1': extract_score(contextual_precision_1),
        'contextual_recall_1': extract_score(contextual_recall_1),
        'contextual_relevancy_1': extract_score(contextual_relevancy_1),
        'call_deflection_effectiveness_1': extract_score(call_deflection_1),
        'relevancy_2': '',
        'faithfulness_2': '',
        'contextual_precision_2': '',
        'contextual_recall_2': '',
        'contextual_relevancy_2': '',
        'call_deflection_effectiveness_2': '',
        'conversation_completeness': extract_score(conversation_completeness)
    }
    results.append(result)
    print(f"[{i+1}/{len(synthetic_data)}] Query: {query}\nModel Response: {response}\nRelevancy: {result['relevancy_1']}\n---")

# Save results to CSV
csv_file = 'synthetic_eval_log.csv'
fieldnames = list(results[0].keys())
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nSaved synthetic evaluation results to {csv_file}") 