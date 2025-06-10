from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
import json
import os

def load_va_docs():
    """Load VA.gov documents from the embeddings directory"""
    docs = []
    embeddings_dir = "embeddings"
    
    # Load all JSON files in the embeddings directory
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.json'):
            with open(os.path.join(embeddings_dir, filename), 'r') as f:
                data = json.load(f)
                # Extract text and metadata
                for doc in data:
                    docs.append({
                        'text': doc['text'],
                        'metadata': doc['metadata']
                    })
    return docs

def save_docs_for_synthesis(docs):
    """Save documents in a format suitable for synthesis"""
    os.makedirs('synthesis_data', exist_ok=True)
    
    # Save each document as a separate text file
    for i, doc in enumerate(docs):
        filename = f"synthesis_data/doc_{i}.txt"
        with open(filename, 'w') as f:
            # Write metadata as comments
            f.write(f"# Source: {doc['metadata']['url']}\n")
            f.write(f"# Title: {doc['metadata'].get('title', '')}\n\n")
            # Write main content
            f.write(doc['text'])

def main():
    # Load VA.gov documents
    print("Loading VA.gov documents...")
    docs = load_va_docs()
    
    # Save documents for synthesis
    print("Saving documents for synthesis...")
    save_docs_for_synthesis(docs)
    
    # Initialize synthesizer
    print("Initializing synthesizer...")
    synthesizer = Synthesizer()
    
    # Create evaluation dataset
    print("Generating synthetic data...")
    dataset = EvaluationDataset()
    dataset.generate_goldens_from_docs(
        max_goldens_per_context=2,
        document_paths=[f"synthesis_data/doc_{i}.txt" for i in range(len(docs))],
        synthesizer=synthesizer
    )
    
    # Save the generated dataset
    print("Saving generated dataset...")
    dataset.save("synthetic_va_dataset.json")
    
    # Print example
    print("\nExample generated data:")
    print(dataset.goldens[0])

if __name__ == "__main__":
    main() 