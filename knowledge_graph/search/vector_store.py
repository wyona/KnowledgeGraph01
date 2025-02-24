"""FAISS-based vector store for entity and relationship embeddings."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import faiss
import pickle
from datetime import datetime

class VectorStore:
    """FAISS-powered vector store for similarity search."""
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "IVFFlat",
        metric: str = "cosine",
        nlist: int = 100
    ):
        """Initialize the vector store.
        
        Args:
            dimension: Dimensionality of vectors
            index_type: FAISS index type (IVFFlat, Flat, etc.)
            metric: Distance metric (cosine, l2, etc.)
            nlist: Number of clusters for IVF indices
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Map FAISS ids to node ids
        self.id_map: Dict[int, str] = {}
        self.reverse_map: Dict[str, int] = {}
        
        # Track index statistics
        self.total_vectors = 0
        self.last_updated = None

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration.
        
        Returns:
            Initialized FAISS index
        """
        # Configure metric
        if self.metric == "cosine":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2
            
        # Create base index
        if self.index_type == "Flat":
            return faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" \
                else faiss.IndexFlatL2(self.dimension)
                
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(
                quantizer, 
                self.dimension,
                self.nlist,
                metric
            )
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def add_vectors(
        self,
        vectors: np.ndarray,
        node_ids: List[str],
        train: bool = False
    ):
        """Add vectors to the store with their corresponding node IDs.
        
        Args:
            vectors: Array of vectors to add (n_vectors x dimension)
            node_ids: List of node IDs corresponding to vectors
            train: Whether to train index before adding (for IVF indices)
        """
        if len(vectors) != len(node_ids):
            raise ValueError("Number of vectors and IDs must match")
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {vectors.shape[1]}")
            
        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
            
        # Train index if needed
        if train and isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            self.index.train(vectors)
            
        # Add vectors to index
        faiss.normalize_L2(vectors)
        start_id = self.total_vectors
        self.index.add(vectors)
        
        # Update ID mappings
        for i, node_id in enumerate(node_ids):
            faiss_id = start_id + i
            self.id_map[faiss_id] = node_id
            self.reverse_map[node_id] = faiss_id
            
        # Update stats
        self.total_vectors += len(vectors)
        self.last_updated = datetime.now()

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            nprobe: Number of clusters to probe (for IVF indices)
            
        Returns:
            Tuple of (distances, indices, node_ids)
        """
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {query_vector.shape[1]}")
            
        # Normalize query vector for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)
            
        # Set search parameters
        if nprobe and isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = nprobe
            
        # Perform search
        distances, indices = self.index.search(query_vector, k)
        
        # Map FAISS indices to node IDs
        node_ids = [self.id_map.get(int(i)) for i in indices[0]]
        
        return distances[0], indices[0], node_ids

    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        """Retrieve vector for a given node ID.
        
        Args:
            node_id: Node ID to look up
            
        Returns:
            Vector if found, None otherwise
        """
        if node_id not in self.reverse_map:
            return None
            
        faiss_id = self.reverse_map[node_id]
        reconstructed = np.zeros((1, self.dimension), dtype=np.float32)
        self.index.reconstruct(faiss_id, reconstructed[0])
        return reconstructed[0]

    def remove_vectors(self, node_ids: List[str]):
        """Remove vectors for given node IDs.
        
        Args:
            node_ids: List of node IDs to remove
        """
        if not isinstance(self.index, faiss.IndexFlat):
            raise NotImplementedError(
                "Vector removal only supported for Flat indices"
            )
            
        # Get FAISS IDs to remove
        faiss_ids = []
        for node_id in node_ids:
            if node_id in self.reverse_map:
                faiss_ids.append(self.reverse_map[node_id])
                del self.reverse_map[node_id]
                
        # Remove from index
        self.index.remove_ids(np.array(faiss_ids))
        
        # Update ID map
        for faiss_id in faiss_ids:
            if faiss_id in self.id_map:
                del self.id_map[faiss_id]
                
        self.total_vectors -= len(faiss_ids)
        self.last_updated = datetime.now()

    def save(self, directory: str):
        """Save vector store to disk.
        
        Args:
            directory: Directory to save files in
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(
            self.index,
            os.path.join(directory, "vectors.index")
        )
        
        # Save ID mappings and metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "total_vectors": self.total_vectors,
            "last_updated": self.last_updated,
            "id_map": self.id_map,
            "reverse_map": self.reverse_map
        }
        
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load vector store from disk.
        
        Args:
            directory: Directory containing saved files
            
        Returns:
            Loaded VectorStore instance
        """
        # Load metadata
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            
        # Create instance
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
            nlist=metadata["nlist"]
        )
        
        # Load FAISS index
        store.index = faiss.read_index(
            os.path.join(directory, "vectors.index")
        )
        
        # Restore metadata
        store.id_map = metadata["id_map"]
        store.reverse_map = metadata["reverse_map"]
        store.total_vectors = metadata["total_vectors"]
        store.last_updated = metadata["last_updated"]
        
        return store
