from .loaders import FileLoader
from typing import List, Optional, Dict
from .chunking import TextChunker
from .embeddings import EmbeddingModel
from .storage import VectorStorage
# Visualizer removed from backend to separate concerns

class VectorDBClient:
    def __init__(self, chunk_size=500, chunk_overlap=50, persist_directory="./chroma_db"):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.encoder = EmbeddingModel()
        self.storage = VectorStorage(persist_directory=persist_directory)

    def get_all_vectors(self):
        """
        Retrieves all vectors and metadata for external visualization.
        """
        return self.storage.get_all_vectors()

    def get_documents(self, limit: int = 10, offset: int = 0, filter_text: Optional[str] = None):
        """
        Retrieves documents with pagination and optional text search.
        Returns a tuple: (data_dict, total_count)
        """
        if not filter_text:
            # Efficient path for no filter
            count = self.storage.collection.count()
            data = self.storage.collection.get(
                limit=limit,
                offset=offset,
                include=['documents', 'metadatas']
            )
            return data, count

        # Hybrid Search Path (Content OR Source)
        # Fetch all to filter in Python
        # Note: This scales poorly with millions of vectors but is fine for thousands.
        results = self.storage.collection.get(include=['documents', 'metadatas'])
        
        filtered_indices = []
        term = filter_text.lower()
        
        for i, doc in enumerate(results['documents']):
             meta = results['metadatas'][i]
             source = meta.get('source', '') if meta else ''
             
             # Safe content extraction
             content = doc if doc else ""
             
             # Case-insensitive check on Content OR Source
             if (term in content.lower()) or (term in source.lower()):
                 filtered_indices.append(i)
        
        total = len(filtered_indices)
        
        # Paginate
        start = offset
        end = min(offset + limit, total)
        slice_indices = filtered_indices[start:end]
        
        sliced_data = {
            'ids': [results['ids'][i] for i in slice_indices],
            'documents': [results['documents'][i] for i in slice_indices],
            'metadatas': [results['metadatas'][i] for i in slice_indices],
        }
        
        return sliced_data, total

    def add_from_files(self, file_paths: List[str]):
        """
        Loads content from files and adds them to DB.
        """
        documents = []
        metadatas = []
        
        for path in file_paths:
            try:
                text = FileLoader.load_file(path)
                documents.append(text)
                metadatas.append({"source": path})
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        if documents:
            self.add_documents(documents, metadatas)

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Process and store documents.
        """
        all_chunks = []
        all_metas = []

        # If metadatas is provided, we must align it with chunks
        # If not provided, we pass None to storage
        
        if metadatas:
             iterator = zip(documents, metadatas)
        else:
             iterator = zip(documents, [None]*len(documents))

        for doc, meta in iterator:
            chunks = self.chunker.split_text(doc)
            for chunk in chunks:
                all_chunks.append(chunk)
                if metadatas:
                    all_metas.append(meta)

        if not all_chunks:
            return

        print(f"Embedding {len(all_chunks)} chunks...")
        vectors = self.encoder.encode(all_chunks)
        
        print(f"Storing in DB...")
        self.storage.add_vectors(vectors, all_chunks, all_metas if metadatas else None)

    def similarity_search(self, query_text: str, top_k: int = 5):
        """
        Search for documents similar to query.
        """
        query_vector = self.encoder.encode([query_text])[0]
        results = self.storage.query_vectors(query_vector, top_k)
        return results

    # visualize method removed. Use external logic with get_all_vectors()

    def delete_collection(self):
        self.storage.delete_collection()

    def delete_document(self, text_content: str):
        self.storage.delete_item(text_content)
