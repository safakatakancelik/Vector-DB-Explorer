from sentence_transformers import SentenceTransformer
from typing import List
import os

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the sentence transformer model lazily.
        """
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """
        Loads the model if it hasn't been loaded yet.
        """
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of texts into vectors.
        
        Args:
            texts (List[str]): List of strings to encode.
            
        Returns:
            List[List[float]]: List of vector embeddings.
        """
        if not texts:
            return []
        
        self._load_model()
        
        embeddings = self.model.encode(texts)
        # Convert numpy arrays to strict lists for JSON/Database compatibility
        return embeddings.tolist()
