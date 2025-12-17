import pytest
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.client import VectorDBClient

def test_apple_search_logic():
    client = VectorDBClient()
    
    # 1. Clean start
    client.delete_collection()
    
    # 2. Add target document
    target_doc = "The fruit apple is red and delicious."
    distractor_doc = "Carrots are orange vegetables."
    client.add_documents([target_doc, distractor_doc])
    
    # 3. Search
    print("Searching for 'apple'...")
    results = client.similarity_search("apple", top_k=1)
    
    # 4. Assert
    top_doc = results['documents'][0][0]
    print(f"Top result: {top_doc}")
    assert "apple" in top_doc
    assert top_doc == target_doc

    # 5. Cleanup / Delete
    client.delete_document(target_doc)
    
    # 6. Verify deletion
    results_after = client.similarity_search("apple", top_k=1)
    best_doc_after = results_after['documents'][0][0] if results_after['documents'][0] else ""
    # Should likely match carrots now or be empty if thresholded, 
    # but strictly it shouldn't be the deleted one.
    assert best_doc_after != target_doc
    
    print("Test passed: Apple added, found, and removed.")

if __name__ == "__main__":
    test_apple_search_logic()
