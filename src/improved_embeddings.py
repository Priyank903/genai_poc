"""
Enhanced Embedding Creation Script for Nephrology RAG System

This script processes the nephro_chunks.json file and creates FAISS embeddings
for efficient similarity search in the RAG system.
"""

import json
import os
import pickle
import argparse
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

class EmbeddingCreator:
    """Creates and manages FAISS embeddings from nephrology text chunks"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the embedding creator"""
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        logger.info(f"Initializing EmbeddingCreator with model: {model_name}")
        
    def load_model(self):
        """Load the SentenceTransformer model"""
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully")
        
    def load_chunks(self, chunks_file: str):
        """
        Load text chunks from JSON file
        
        Args:
            chunks_file: Path to nephro_chunks.json file
        """
        logger.info(f"Loading chunks from: {chunks_file}")
        
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
            
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process chunks from the JSON structure
        chunk_id = 0
        for page_data in data:
            page_number = page_data.get('page_number', 0)
            paragraphs = page_data.get('paragraphs', [])
            
            for paragraph in paragraphs:
                # Skip empty paragraphs or very short ones
                if not paragraph or len(paragraph.strip()) < 10:
                    continue
                    
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'text': paragraph.strip(),
                    'page': page_number,
                    'source': 'comprehensive-clinical-nephrology',
                    'chunk_type': 'paragraph'
                }
                
                self.chunks.append(chunk_metadata)
                chunk_id += 1
                
        logger.info(f"Loaded {len(self.chunks)} text chunks from {len(data)} pages")
        
    def create_embeddings(self, batch_size: int = 32):
        """
        Create embeddings for all chunks
        
        Args:
            batch_size: Number of chunks to process at once
        """
        if not self.model:
            self.load_model()
            
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")
            
        logger.info(f"Creating embeddings for {len(self.chunks)} chunks...")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Create embeddings in batches with progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
            
        # Combine all embeddings
        self.embeddings = np.vstack(all_embeddings)
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
        
    def create_faiss_index(self):
        """Create FAISS index from embeddings"""
        if self.embeddings is None:
            raise ValueError("No embeddings created. Call create_embeddings() first.")
            
        logger.info("Creating FAISS index...")
        
        # Create FAISS index for cosine similarity (Inner Product after normalization)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
        
    def save_index(self, output_dir: str):
        """
        Save FAISS index and metadata to disk
        
        Args:
            output_dir: Directory to save the index files
        """
        if not self.index:
            raise ValueError("No FAISS index created. Call create_faiss_index() first.")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "nephro_index.faiss")
        faiss.write_index(self.index, index_path)
        logger.info(f"FAISS index saved to: {index_path}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "nephro_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Metadata saved to: {metadata_path}")
        
        # Save index info
        info_path = os.path.join(output_dir, "index_info.json")
        index_info = {
            "model_name": self.model_name,
            "total_chunks": len(self.chunks),
            "embedding_dim": EMBEDDING_DIM,
            "created_at": datetime.now().isoformat(),
            "index_type": "IndexFlatIP (cosine similarity)"
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(index_info, f, indent=2)
        logger.info(f"Index info saved to: {info_path}")
        
    def test_search(self, query: str, top_k: int = 3):
        """
        Test the created index with a sample query
        
        Args:
            query: Test query string
            top_k: Number of results to return
        """
        if not self.index or not self.model:
            raise ValueError("Index and model must be created first")
            
        logger.info(f"Testing search with query: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        logger.info("Search results:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                logger.info(f"  {i+1}. Score: {score:.3f}, Page: {chunk['page']}")
                logger.info(f"     Text: {chunk['text'][:100]}...")
                
def main():
    """Main function to create embeddings from command line"""
    parser = argparse.ArgumentParser(description="Create FAISS embeddings from nephrology chunks")
    parser.add_argument("--chunks", default="../data/nephro_chunks.json", 
                       help="Path to nephro_chunks.json file")
    parser.add_argument("--output", default="../embeddings/nephro_faiss", 
                       help="Output directory for FAISS index")
    parser.add_argument("--model", default=MODEL_NAME, 
                       help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for embedding creation")
    parser.add_argument("--test-query", default="What is chronic kidney disease?", 
                       help="Test query for validation")
    
    args = parser.parse_args()
    
    try:
        # Initialize creator
        creator = EmbeddingCreator(model_name=args.model)
        
        # Load chunks
        creator.load_chunks(args.chunks)
        
        # Create embeddings
        creator.create_embeddings(batch_size=args.batch_size)
        
        # Create FAISS index
        creator.create_faiss_index()
        
        # Save to disk
        creator.save_index(args.output)
        
        # Test the index
        creator.test_search(args.test_query)
        
        logger.info("✅ Embedding creation completed successfully!")
        logger.info(f"📁 Files saved to: {args.output}")
        logger.info(f"📊 Total chunks: {len(creator.chunks)}")
        logger.info(f"🔍 Index size: {creator.index.ntotal} vectors")
        
    except Exception as e:
        logger.error(f"❌ Error creating embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    main()
