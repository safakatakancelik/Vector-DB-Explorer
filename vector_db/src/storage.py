import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class VectorStorage:
    def __init__(self, collection_name: str = "my_vectors", persist_directory: str = "./chroma_db"):
        """
        Initializes the ChromaDB client and collection.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def add_vectors(self, vectors: List[List[float]], documents: List[str], metadatas: Optional[List[Dict]]):
        """
        Adds vectors to the collection.
        """
        ids = [str(uuid.uuid4()) for _ in vectors]
        self.collection.add(
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def query_vectors(self, query_vector: List[float], n_results: int = 5):
        """
        Searches for nearest neighbors.
        """
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )

    def get_all_vectors(self):
        """
        Retrieves all vectors for visualization.
        """
        return self.collection.get(include=['embeddings', 'documents', 'metadatas'])

    def delete_collection(self):
        """
        Deletes the entire collection.
        """
        self.client.delete_collection(self.collection_name)
        # Re-create empty so the object is still usable
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def delete_item(self, doc_text: str):
        """
        Deletes items where document content matches exactly.
        Useful for testing cleanup.
        """
        # ChromaDB doesn't support $eq for documents, only $contains.
        # We fetch candidates and filter for exact match in Python.
        result = self.collection.get(where_document={"$contains": doc_text})
        
        ids_to_delete = []
        if result and result.get('ids'):
            for i, doc in enumerate(result['documents']):
                if doc == doc_text:
                    ids_to_delete.append(result['ids'][i])
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
