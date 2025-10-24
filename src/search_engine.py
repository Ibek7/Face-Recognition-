# Full-Text Search Engine

import threading
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import math

class SearchMode(Enum):
    """Search modes."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    PHRASE = "phrase"
    BOOLEAN = "boolean"
    RANGE = "range"

@dataclass
class Document:
    """Searchable document."""
    doc_id: str
    title: str
    content: str
    metadata: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score
        }

class InvertedIndex:
    """Inverted index for full-text search."""
    
    def __init__(self):
        self.index: Dict[str, Set[str]] = {}
        self.doc_frequency: Dict[str, Dict[str, int]] = {}
        self.lock = threading.RLock()
    
    def index_document(self, doc_id: str, text: str) -> None:
        """Index document text."""
        # Tokenize
        tokens = self._tokenize(text)
        
        with self.lock:
            for token in set(tokens):
                if token not in self.index:
                    self.index[token] = set()
                self.index[token].add(doc_id)
                
                if token not in self.doc_frequency:
                    self.doc_frequency[token] = {}
                self.doc_frequency[token][doc_id] = tokens.count(token)
    
    def remove_document(self, doc_id: str) -> None:
        """Remove document from index."""
        with self.lock:
            for token in list(self.index.keys()):
                if doc_id in self.index[token]:
                    self.index[token].discard(doc_id)
                    if not self.index[token]:
                        del self.index[token]
                
                if token in self.doc_frequency and doc_id in self.doc_frequency[token]:
                    del self.doc_frequency[token][doc_id]
    
    def search(self, token: str) -> Set[str]:
        """Search for token."""
        with self.lock:
            return self.index.get(token.lower(), set())
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        text = text.lower()
        # Simple tokenization
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum():
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
        
        if current_token:
            tokens.append(current_token)
        
        return tokens

class BM25Ranker:
    """BM25 ranking algorithm."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.avgdl = 0  # Average document length
        self.N = 0      # Number of documents
    
    def score(self, term_frequency: int, doc_length: int,
             document_frequency: int) -> float:
        """Calculate BM25 score."""
        if self.N == 0 or self.avgdl == 0:
            return 0.0
        
        idf = math.log((self.N - document_frequency + 0.5) /
                      (document_frequency + 0.5) + 1)
        
        numerator = term_frequency * (self.k1 + 1)
        denominator = term_frequency + self.k1 * (
            1 - self.b + self.b * (doc_length / self.avgdl)
        )
        
        return idf * (numerator / denominator)

class FullTextSearchEngine:
    """Full-text search engine."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.index = InvertedIndex()
        self.ranker = BM25Ranker()
        self.lock = threading.RLock()
    
    def add_document(self, doc_id: str, title: str, content: str,
                    metadata: Dict = None) -> None:
        """Add document to index."""
        metadata = metadata or {}
        
        with self.lock:
            doc = Document(doc_id, title, content, metadata)
            self.documents[doc_id] = doc
            
            # Index title and content
            self.index.index_document(doc_id, title + " " + content)
            
            # Update BM25 parameters
            self.ranker.N = len(self.documents)
            total_length = sum(len(d.content.split()) for d in self.documents.values())
            self.ranker.avgdl = total_length / self.ranker.N if self.ranker.N > 0 else 0
    
    def remove_document(self, doc_id: str) -> None:
        """Remove document from index."""
        with self.lock:
            if doc_id in self.documents:
                del self.documents[doc_id]
                self.index.remove_document(doc_id)
    
    def search(self, query: str, mode: SearchMode = SearchMode.FUZZY,
              limit: int = 10) -> List[Document]:
        """Search documents."""
        if mode == SearchMode.EXACT:
            return self._search_exact(query, limit)
        elif mode == SearchMode.PHRASE:
            return self._search_phrase(query, limit)
        elif mode == SearchMode.FUZZY:
            return self._search_fuzzy(query, limit)
        elif mode == SearchMode.BOOLEAN:
            return self._search_boolean(query, limit)
        
        return []
    
    def _search_exact(self, query: str, limit: int) -> List[Document]:
        """Exact phrase search."""
        query_lower = query.lower()
        results = []
        
        with self.lock:
            for doc in self.documents.values():
                if query_lower in doc.content.lower():
                    doc.score = 10.0
                    results.append(doc)
        
        return sorted(results, key=lambda d: d.score, reverse=True)[:limit]
    
    def _search_phrase(self, query: str, limit: int) -> List[Document]:
        """Phrase search."""
        tokens = query.lower().split()
        results: Dict[str, float] = {}
        
        with self.lock:
            for token in tokens:
                doc_ids = self.index.search(token)
                for doc_id in doc_ids:
                    if doc_id not in results:
                        results[doc_id] = 0
                    results[doc_id] += 1
        
        # Filter to docs with all tokens
        matching_docs = [
            (self.documents[doc_id], score)
            for doc_id, score in results.items()
            if score == len(tokens)
        ]
        
        return [doc for doc, _ in sorted(matching_docs, key=lambda x: x[1], reverse=True)][:limit]
    
    def _search_fuzzy(self, query: str, limit: int) -> List[Document]:
        """Fuzzy search."""
        tokens = query.lower().split()
        scored_docs: Dict[str, float] = {}
        
        with self.lock:
            for token in tokens:
                doc_ids = self.index.search(token)
                
                for doc_id in doc_ids:
                    if doc_id not in scored_docs:
                        scored_docs[doc_id] = 0.0
                    
                    doc = self.documents[doc_id]
                    doc_length = len(doc.content.split())
                    term_freq = self.index.doc_frequency.get(token, {}).get(doc_id, 0)
                    doc_freq = len(self.index.search(token))
                    
                    score = self.ranker.score(term_freq, doc_length, doc_freq)
                    scored_docs[doc_id] += score
        
        # Sort by score
        ranked = sorted(
            [(self.documents[doc_id], score) for doc_id, score in scored_docs.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, _ in ranked[:limit]]
    
    def _search_boolean(self, query: str, limit: int) -> List[Document]:
        """Boolean search."""
        # Simple AND/OR/NOT parsing
        must_terms = [t for t in query.split() if not t.startswith("-")]
        not_terms = [t[1:] for t in query.split() if t.startswith("-")]
        
        results = set()
        
        with self.lock:
            # Find docs with all must terms
            for term in must_terms:
                doc_ids = self.index.search(term)
                if not results:
                    results = doc_ids
                else:
                    results = results.intersection(doc_ids)
            
            # Remove docs with not terms
            for term in not_terms:
                excluded = self.index.search(term)
                results = results - excluded
        
        return [self.documents[doc_id] for doc_id in results][:limit]
    
    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        with self.lock:
            return {
                'documents': len(self.documents),
                'terms': len(self.index.index),
                'total_indexed': sum(len(docs) for docs in self.index.index.values())
            }

class SearchAnalytics:
    """Analytics for search queries."""
    
    def __init__(self):
        self.queries: List[Dict] = []
        self.lock = threading.RLock()
    
    def log_query(self, query: str, results_count: int, elapsed_ms: float) -> None:
        """Log search query."""
        with self.lock:
            self.queries.append({
                'query': query,
                'results': results_count,
                'elapsed_ms': elapsed_ms,
                'timestamp': time.time()
            })
    
    def get_top_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top search queries."""
        with self.lock:
            query_counts: Dict[str, int] = {}
            for q in self.queries:
                query = q['query']
                query_counts[query] = query_counts.get(query, 0) + 1
            
            return sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_avg_response_time(self) -> float:
        """Get average query response time."""
        with self.lock:
            if not self.queries:
                return 0.0
            
            total = sum(q['elapsed_ms'] for q in self.queries)
            return total / len(self.queries)

# Example usage
if __name__ == "__main__":
    engine = FullTextSearchEngine()
    
    # Add documents
    docs = [
        ("doc1", "Face Recognition", "Advanced face recognition technology for security"),
        ("doc2", "Deep Learning", "Deep learning models for computer vision"),
        ("doc3", "Neural Networks", "Neural networks and artificial intelligence"),
    ]
    
    for doc_id, title, content in docs:
        engine.add_document(doc_id, title, content)
    
    # Search
    results = engine.search("face recognition")
    print(f"Search results for 'face recognition': {len(results)} documents")
    for doc in results:
        print(f"  - {doc.title}: {doc.score:.2f}")
    
    # Get stats
    stats = engine.get_index_stats()
    print(f"\nIndex Stats: {json.dumps(stats, indent=2)}")
