import json
from typing import List, Dict
import chromadb
import os
from dotenv import load_dotenv
import openai

load_dotenv()

class CustomOpenAIEmbeddingFunction:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "text-embedding-3-small"
        self.client = openai.OpenAI(api_key=self.api_key)

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Handle empty input
        if not input:
            return []
        # Get embeddings from OpenAI
        response = self.client.embeddings.create(
            model=self.model,
            input=input
        )
        # Extract embeddings from response
        return [item.embedding for item in response.data]

class DocumentProcessor:
    def __init__(self, collection_name: str = "va_docs"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = CustomOpenAIEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        # Check if the collection is empty and process documents if it is
        if self.collection.count() == 0:
            print(f"Collection '{collection_name}' is empty or newly created. Processing documents...")
            self.process_documents("va_content.json")
        else:
            print(f"Collection '{collection_name}' already exists and contains {self.collection.count()} documents. Skipping document processing.")

    def process_documents(self, json_file: str, chunk_size: int = 300):
        """
        Process documents from JSON file and add to vector store
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        for idx, doc in enumerate(documents):
            # Create chunks from the content
            content = doc['content']
            chunks = self._create_chunks(content, chunk_size)
            
            # Add chunks to the collection
            for chunk_idx, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    metadatas=[{
                        "url": doc['url'],
                        "title": doc['title'],
                        "chunk_idx": chunk_idx
                    }],
                    ids=[f"doc_{idx}_chunk_{chunk_idx}"]
                )

    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately equal size
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def query_documents(self, query: str, n_results: int = 8) -> List[Dict]:
        """
        Query the vector store for relevant documents
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [{
            "text": doc,
            "metadata": metadata
        } for doc, metadata in zip(results['documents'][0], results['metadatas'][0])]

if __name__ == "__main__":
    # Initialize with PersistentClient for direct script execution
    processor = DocumentProcessor()
    # No need to call process_documents here, it's handled in __init__ now
    # Test query
    results = processor.query_documents("How do I apply for VA health care?")
    for result in results:
        print(f"\nDocument: {result['text'][:200]}...")
        print(f"Source: {result['metadata']['url']}") 