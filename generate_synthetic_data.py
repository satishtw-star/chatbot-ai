import os
import json
import glob
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
import chromadb


def load_va_docs():
    docs = []
    try:
        with open("va_content.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        # The first entry is the main page, and its 'links' contain the articles
        if data and "links" in data[0]:
            # Add the main page content itself as a document
            docs.append({"text": data[0]["content"], "url": data[0]["url"], "title": data[0]["title"]})

            # Add content from the linked articles (assuming they are also in data list)
            # We iterate through the data list, skipping the first element which is already added
            for entry in data[1:]:
                docs.append({"text": entry["content"], "url": entry["url"], "title": entry["title"]})

    except FileNotFoundError:
        print("Error: va_content.json not found. Please run the scraper first.")
        exit()
    except json.JSONDecodeError:
        print("Error: Could not decode va_content.json. Ensure it's valid JSON.")
        exit()
    return docs

def save_docs_for_synthesis(docs, directory="synthesis_data"):
    os.makedirs(directory, exist_ok=True)
    for i, doc in enumerate(docs):
        filename = os.path.join(directory, f"doc_{i}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {doc['url']}\n")
            f.write(f"Title: {doc['title']}\n")
            f.write(f"Content:\n{doc['text']}")
    print(f"Saved {len(docs)} documents to '{directory}'.")

def main():
    # Load documents from va_content.json
    va_docs = load_va_docs()

    if not va_docs:
        print("No documents to process. Exiting.")
        return

    # Save documents as individual text files for the synthesizer
    save_docs_for_synthesis(va_docs)

    # Define StylingConfig
    styling_config = StylingConfig(
        expected_output_format="Ensure the output resembles a VA chatbot tasked with providing information on VA benefits and services. It should pose additional questions if the details are inadequate or provide relevant information/direct to resources when the input is sufficiently detailed.",
        input_format="Mimic the kind of queries or statements a Veteran or their family might share with a VA chatbot when seeking information or assistance with VA benefits and services.",
        task="The chatbot acts as a specialist in VA benefits and services, integrated with relevant VA systems. It manages tasks in a sequence to ensure precise and effective information delivery and support processing.",
        scenario="Veterans or their family members describing their needs to seek information or assistance with VA benefits and services."
    )

    # Initialize Synthesizer with StylingConfig
    synthesizer = Synthesizer(model="gpt-4o-mini", styling_config=styling_config)

    # Generate synthetic data
    print("Generating synthetic data...")
    dataset = EvaluationDataset()

    # Collect paths of the saved documents
    document_paths = glob.glob("synthesis_data/*.txt")

    dataset.generate_goldens_from_docs(
        max_goldens_per_context=1,
        document_paths=document_paths,
        synthesizer=synthesizer
    )

    # Save the dataset
    with open("synthetic_va_dataset.json", "w", encoding="utf-8") as f:
        json.dump([golden.model_dump() for golden in dataset.goldens], f, ensure_ascii=False, indent=2)
    print(f"Successfully generated {len(dataset.goldens)} goldens and saved to synthetic_va_dataset.json")

if __name__ == "__main__":
    main() 