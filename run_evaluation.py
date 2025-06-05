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

if __name__ == "__main__":
    main() 