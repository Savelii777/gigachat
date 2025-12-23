"""Qdrant vector database integration for knowledge base RAG."""
import logging
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..utils.config import QdrantConfig

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDocument:
    """A document in the knowledge base."""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0


class QdrantKnowledgeBase:
    """Knowledge base using Qdrant vector database for RAG."""
    
    def __init__(self, config: QdrantConfig):
        """Initialize Qdrant knowledge base client.
        
        Args:
            config: Qdrant configuration object
        """
        self.config = config
        self._client = None
        self._initialized = False
        
    def _initialize(self):
        """Initialize the Qdrant client and ensure collection exists."""
        if self._initialized:
            return
            
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # Connect to Qdrant
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
            )
            
            # Check if collection exists, create if not
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.config.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    f"Created collection: {self.config.collection_name}"
                )
            
            self._initialized = True
            logger.info("Qdrant client initialized successfully")
            
        except ImportError:
            logger.warning(
                "qdrant-client package not installed. "
                "Install with: pip install qdrant-client"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using GigaChat.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            from gigachat import GigaChat
            
            # Use GigaChat for embeddings
            # Note: You may need to configure this with your credentials
            with GigaChat() as giga:
                embeddings = []
                for text in texts:
                    result = giga.embeddings(input=[text])
                    if result.data:
                        embeddings.append(result.data[0].embedding)
                    else:
                        # Return zero vector if embedding fails
                        embeddings.append([0.0] * self.config.vector_size)
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.config.vector_size for _ in texts]
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> bool:
        """Add documents to the knowledge base.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            
        Returns:
            True if successful
        """
        self._initialize()
        
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return False
        
        try:
            from qdrant_client.models import PointStruct
            
            # Extract contents and generate embeddings
            contents = [doc.get("content", "") for doc in documents]
            embeddings = self._get_embeddings(contents)
            
            # Create points
            points = []
            for i, doc in enumerate(documents):
                point_id = doc.get("id", str(uuid.uuid4()))
                payload = {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                }
                
                points.append(PointStruct(
                    id=point_id if isinstance(point_id, int) else hash(point_id) % (10**9),
                    vector=embeddings[i],
                    payload=payload,
                ))
            
            # Upload points
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5,
    ) -> List[KnowledgeDocument]:
        """Search the knowledge base for relevant documents.
        
        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching documents
        """
        self._initialize()
        
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embeddings([query])[0]
            
            # Search
            results = self._client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )
            
            # Convert to KnowledgeDocument objects
            documents = []
            for result in results:
                doc = KnowledgeDocument(
                    id=str(result.id),
                    content=result.payload.get("content", ""),
                    metadata=result.payload.get("metadata", {}),
                    score=result.score,
                )
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_context_for_query(
        self,
        query: str,
        max_context_length: int = 2000,
    ) -> str:
        """Get relevant context from knowledge base for a query.
        
        Args:
            query: The query to find context for
            max_context_length: Maximum length of combined context
            
        Returns:
            Concatenated relevant context string
        """
        documents = self.search(query, limit=5)
        
        if not documents:
            return ""
        
        # Combine document contents
        context_parts = []
        current_length = 0
        
        for doc in documents:
            content = doc.content
            if current_length + len(content) > max_context_length:
                # Truncate to fit
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if meaningful amount remains
                    context_parts.append(content[:remaining] + "...")
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n\n".join(context_parts)
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the knowledge base.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        self._initialize()
        
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return False
        
        try:
            from qdrant_client.models import PointIdsList
            
            # Convert string IDs to integer hashes
            point_ids = [
                hash(doc_id) % (10**9) if isinstance(doc_id, str) else doc_id
                for doc_id in document_ids
            ]
            
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
            
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        self._initialize()
        
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return False
        
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Delete and recreate collection
            self._client.delete_collection(
                collection_name=self.config.collection_name
            )
            
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            
            logger.info(f"Cleared collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def close(self):
        """Close the Qdrant client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        self._initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
