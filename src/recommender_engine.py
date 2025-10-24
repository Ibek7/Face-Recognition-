# Recommendation Engine

import threading
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import math

class RecommendationType(Enum):
    """Recommendation types."""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    POPULARITY = "popularity"

@dataclass
class UserProfile:
    """User profile for recommendations."""
    user_id: str
    preferences: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

@dataclass
class Item:
    """Item for recommendation."""
    item_id: str
    title: str
    features: Dict[str, float] = field(default_factory=dict)
    popularity: int = 0
    created_at: float = field(default_factory=time.time)

@dataclass
class UserItemInteraction:
    """User-item interaction."""
    user_id: str
    item_id: str
    interaction_type: str
    rating: float = 0.0
    timestamp: float = field(default_factory=time.time)

class CollaborativeFilteringEngine:
    """Collaborative filtering for recommendations."""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.items: Dict[str, Item] = {}
        self.interactions: List[UserItemInteraction] = []
        self.user_similarity: Dict[Tuple[str, str], float] = {}
        self.lock = threading.RLock()
    
    def add_user(self, user_id: str) -> None:
        """Add user profile."""
        with self.lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id)
    
    def add_item(self, item: Item) -> None:
        """Add item."""
        with self.lock:
            self.items[item.item_id] = item
    
    def record_interaction(self, interaction: UserItemInteraction) -> None:
        """Record user-item interaction."""
        with self.lock:
            self.interactions.append(interaction)
            
            if interaction.user_id in self.user_profiles:
                self.user_profiles[interaction.user_id].interaction_history.append(
                    interaction.item_id
                )
            
            if interaction.item_id in self.items:
                self.items[interaction.item_id].popularity += 1
    
    def calculate_user_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between users using Pearson correlation."""
        with self.lock:
            items1 = set(self.user_profiles.get(user1, UserProfile(user1)).interaction_history)
            items2 = set(self.user_profiles.get(user2, UserProfile(user2)).interaction_history)
            
            common_items = items1.intersection(items2)
            
            if not common_items:
                return 0.0
            
            # Simple Jaccard similarity
            union = items1.union(items2)
            return len(common_items) / len(union) if union else 0.0
    
    def get_similar_users(self, user_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get similar users."""
        similarities = []
        
        with self.lock:
            for other_user in self.user_profiles.keys():
                if other_user != user_id:
                    sim = self.calculate_user_similarity(user_id, other_user)
                    if sim > 0:
                        similarities.append((other_user, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]
    
    def recommend_items(self, user_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Recommend items using collaborative filtering."""
        if user_id not in self.user_profiles:
            return []
        
        user_items = set(self.user_profiles[user_id].interaction_history)
        recommendations: Dict[str, float] = {}
        
        with self.lock:
            # Get similar users
            similar_users = self.get_similar_users(user_id, 5)
            
            for similar_user, similarity in similar_users:
                similar_items = self.user_profiles[similar_user].interaction_history
                
                for item_id in similar_items:
                    if item_id not in user_items:
                        if item_id not in recommendations:
                            recommendations[item_id] = 0.0
                        recommendations[item_id] += similarity
        
        # Sort by score
        ranked = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

class ContentBasedEngine:
    """Content-based recommendation engine."""
    
    def __init__(self):
        self.items: Dict[str, Item] = {}
        self.user_preferences: Dict[str, Dict[str, float]] = {}
        self.lock = threading.RLock()
    
    def add_item(self, item: Item) -> None:
        """Add item."""
        with self.lock:
            self.items[item.item_id] = item
    
    def update_user_preferences(self, user_id: str, item_id: str, rating: float) -> None:
        """Update user preferences based on item features."""
        with self.lock:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
            
            if item_id in self.items:
                item = self.items[item_id]
                
                for feature, value in item.features.items():
                    if feature not in self.user_preferences[user_id]:
                        self.user_preferences[user_id][feature] = 0.0
                    
                    self.user_preferences[user_id][feature] += rating * value
    
    def calculate_item_score(self, user_id: str, item_id: str) -> float:
        """Calculate item score for user."""
        if user_id not in self.user_preferences or item_id not in self.items:
            return 0.0
        
        user_prefs = self.user_preferences[user_id]
        item = self.items[item_id]
        
        score = 0.0
        for feature, pref_value in user_prefs.items():
            if feature in item.features:
                score += pref_value * item.features[feature]
        
        return score
    
    def recommend_items(self, user_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Recommend items based on content."""
        if user_id not in self.user_preferences:
            return []
        
        scores = []
        
        with self.lock:
            for item_id in self.items.keys():
                score = self.calculate_item_score(user_id, item_id)
                if score > 0:
                    scores.append((item_id, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:limit]

class HybridRecommendationEngine:
    """Hybrid recommendation engine combining multiple strategies."""
    
    def __init__(self):
        self.collaborative = CollaborativeFilteringEngine()
        self.content_based = ContentBasedEngine()
        self.popularity_scores: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def add_user(self, user_id: str) -> None:
        """Add user."""
        self.collaborative.add_user(user_id)
    
    def add_item(self, item: Item) -> None:
        """Add item."""
        self.collaborative.add_item(item)
        self.content_based.add_item(item)
        
        with self.lock:
            self.popularity_scores[item.item_id] = item.popularity
    
    def record_interaction(self, interaction: UserItemInteraction) -> None:
        """Record interaction."""
        self.collaborative.record_interaction(interaction)
        self.content_based.update_user_preferences(
            interaction.user_id,
            interaction.item_id,
            interaction.rating
        )
    
    def recommend_items(self, user_id: str, limit: int = 10,
                       weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """Get hybrid recommendations."""
        weights = weights or {
            'collaborative': 0.4,
            'content_based': 0.4,
            'popularity': 0.2
        }
        
        scores: Dict[str, float] = {}
        
        # Collaborative filtering
        collab_recs = self.collaborative.recommend_items(user_id, limit * 2)
        for item_id, score in collab_recs:
            scores[item_id] = scores.get(item_id, 0.0) + score * weights['collaborative']
        
        # Content-based
        content_recs = self.content_based.recommend_items(user_id, limit * 2)
        for item_id, score in content_recs:
            scores[item_id] = scores.get(item_id, 0.0) + score * weights['content_based']
        
        # Popularity
        with self.lock:
            max_popularity = max(self.popularity_scores.values()) if self.popularity_scores else 1
            for item_id, popularity in self.popularity_scores.items():
                norm_popularity = popularity / max_popularity
                scores[item_id] = scores.get(item_id, 0.0) + norm_popularity * weights['popularity']
        
        # Sort and return
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

class ABTestingEngine:
    """A/B testing for recommendations."""
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.results: List[Dict] = []
        self.lock = threading.RLock()
    
    def create_experiment(self, experiment_id: str, control_strategy: str,
                         test_strategy: str) -> None:
        """Create A/B test."""
        with self.lock:
            self.experiments[experiment_id] = {
                'control': control_strategy,
                'test': test_strategy,
                'created_at': time.time(),
                'conversions_control': 0,
                'conversions_test': 0,
                'impressions_control': 0,
                'impressions_test': 0
            }
    
    def log_result(self, experiment_id: str, variant: str, converted: bool) -> None:
        """Log experiment result."""
        with self.lock:
            if experiment_id in self.experiments:
                key = f'impressions_{variant}'
                self.experiments[experiment_id][key] += 1
                
                if converted:
                    key = f'conversions_{variant}'
                    self.experiments[experiment_id][key] += 1
    
    def get_results(self, experiment_id: str) -> Dict:
        """Get experiment results."""
        with self.lock:
            if experiment_id not in self.experiments:
                return {}
            
            exp = self.experiments[experiment_id]
            
            control_ctr = (exp['conversions_control'] / exp['impressions_control']
                          if exp['impressions_control'] > 0 else 0)
            test_ctr = (exp['conversions_test'] / exp['impressions_test']
                       if exp['impressions_test'] > 0 else 0)
            
            return {
                'experiment_id': experiment_id,
                'control_ctr': control_ctr,
                'test_ctr': test_ctr,
                'improvement': (test_ctr - control_ctr) / control_ctr if control_ctr > 0 else 0
            }

# Example usage
if __name__ == "__main__":
    engine = HybridRecommendationEngine()
    
    # Add users
    engine.add_user("user-1")
    engine.add_user("user-2")
    
    # Add items
    items = [
        Item("item-1", "Face Detection", {"accuracy": 0.8, "speed": 0.9}),
        Item("item-2", "Face Recognition", {"accuracy": 0.95, "speed": 0.7}),
        Item("item-3", "Facial Expression", {"accuracy": 0.85, "speed": 0.8}),
    ]
    
    for item in items:
        engine.add_item(item)
    
    # Record interactions
    engine.record_interaction(UserItemInteraction("user-1", "item-1", "view", 4.5))
    engine.record_interaction(UserItemInteraction("user-1", "item-2", "view", 5.0))
    engine.record_interaction(UserItemInteraction("user-2", "item-1", "view", 3.5))
    
    # Get recommendations
    recs = engine.recommend_items("user-1", 5)
    print("Recommendations for user-1:")
    for item_id, score in recs:
        print(f"  - {item_id}: {score:.2f}")
