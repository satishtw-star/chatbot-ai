from chat_evaluator import ChatEvaluator
import json
import os

def main():
    # Check if chat history exists
    if not os.path.exists('chat_history.json'):
        print("Error: chat_history.json not found. Please run the chatbot first to generate chat history.")
        return

    # Load chat history
    print("Loading chat history...")
    with open('chat_history.json', 'r') as f:
        chat_history = json.load(f)

    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = ChatEvaluator()

    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate_chat_history(chat_history)

    # Print summary
    print("\nEvaluation complete!")
    print(f"Processed {len(results)} conversation pairs")
    print("Results have been saved to chat_eval_log.csv")

    print("\nDetailed Metric Scores:")
    metrics_to_display = [
        'relevancy', 'faithfulness', 'contextual_precision', 
        'contextual_recall', 'contextual_relevancy', 'conversation_completeness',
        'call_deflection_effectiveness'
    ]
    for i, result in enumerate(results):
        print(f"--- Conversation Pair {i+1} ---")
        for metric in metrics_to_display:
            if metric in result:
                print(f"  {metric.replace('_', ' ').title()}: {result[metric]:.4f}")
            else:
                print(f"  {metric.replace('_', ' ').title()}: N/A")
        print("----------------------")

if __name__ == "__main__":
    main() 