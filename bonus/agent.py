from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import Embedder  # noqa: E402


COLLECTION = "bonus_hybrid_memory"
RRF_K = 60


@dataclass
class UserProfile:
    preferred_language: str = "vi/en mix"
    topic_affinity: str = "cloud"
    reading_speed_wpm: int = 220
    active_hours: str = "20:00-23:00"


@dataclass
class RecentActivity:
    queries_last_hour: int = 0
    recent_topics: list[str] = field(default_factory=list)
    last_query: str = ""


class MiniFeatureStore:
    """Small in-memory stand-in for Feast online features used by the POC."""

    def __init__(self) -> None:
        self.profiles: dict[str, UserProfile] = {
            "u_001": UserProfile(),
        }
        self.activity: dict[str, RecentActivity] = {
            "u_001": RecentActivity(recent_topics=["cloud", "security"]),
        }

    def get_profile(self, user_id: str) -> UserProfile:
        return self.profiles.setdefault(user_id, UserProfile())

    def get_activity(self, user_id: str) -> RecentActivity:
        return self.activity.setdefault(user_id, RecentActivity())

    def record_query(self, user_id: str, query: str) -> None:
        activity = self.get_activity(user_id)
        activity.queries_last_hour += 1
        activity.last_query = query
        for topic in ("cloud", "security", "kubernetes", "ai", "data"):
            if topic.lower() in query.lower() and topic not in activity.recent_topics:
                activity.recent_topics.append(topic)
        activity.recent_topics = activity.recent_topics[-5:]


@dataclass
class Memory:
    memory_id: int
    user_id: str
    text: str
    source: str = "user_saved"


class HybridMemoryAgent:
    """Minimal hybrid memory agent: remember text, then recall assembled context."""

    def __init__(self, feature_store: MiniFeatureStore | None = None) -> None:
        self.embedder = Embedder("fastembed")
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=self.embedder.dim,
                distance=models.Distance.COSINE,
            ),
        )
        self.feature_store = feature_store or MiniFeatureStore()
        self.memories: list[Memory] = []
        self._bm25: BM25Okapi | None = None

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        chunks = self._chunk(text)
        if not chunks:
            return
        vectors = list(self.embedder.embed(chunks))
        points = []
        for chunk, vector in zip(chunks, vectors):
            memory = Memory(memory_id=len(self.memories), user_id=user_id, text=chunk)
            self.memories.append(memory)
            points.append(
                models.PointStruct(
                    id=memory.memory_id,
                    vector=np.asarray(vector, dtype=np.float32).tolist(),
                    payload={
                        "memory_id": memory.memory_id,
                        "user_id": user_id,
                        "text": chunk,
                        "source": memory.source,
                    },
                )
            )
        self.client.upsert(collection_name=COLLECTION, points=points)
        self._rebuild_bm25()

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top memories plus user features, then return assembled context."""
        self.feature_store.record_query(user_id, query)
        profile = self.feature_store.get_profile(user_id)
        activity = self.feature_store.get_activity(user_id)
        hits = self._hybrid_search(query, user_id=user_id, top_k=3)

        memory_lines = [
            f"{i}. {memory.text}"
            for i, memory in enumerate(hits, start=1)
        ] or ["No matching memories found."]

        return "\n".join(
            [
                f"User: {user_id}",
                "Profile features:",
                f"- preferred_language: {profile.preferred_language}",
                f"- topic_affinity: {profile.topic_affinity}",
                f"- reading_speed_wpm: {profile.reading_speed_wpm}",
                f"- active_hours: {profile.active_hours}",
                "Recent activity features:",
                f"- queries_last_hour: {activity.queries_last_hour}",
                f"- recent_topics: {', '.join(activity.recent_topics) or 'none'}",
                f"- last_query: {activity.last_query}",
                "Top episodic memories:",
                *memory_lines,
            ]
        )

    @staticmethod
    def _chunk(text: str, max_chars: int = 420) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text.strip()] if text.strip() else []
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _rebuild_bm25(self) -> None:
        tokenized = [self._tokenize(m.text) for m in self.memories]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _keyword_ids(self, query: str, user_id: str, depth: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in ranked if self.memories[i].user_id == user_id][:depth]

    def _vector_ids(self, query: str, user_id: str, depth: int) -> list[int]:
        q_vec = np.asarray(next(self.embedder.embed([query])), dtype=np.float32).tolist()
        q_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            ]
        )
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            query_filter=q_filter,
            limit=depth,
        ).points
        return [int(p.payload["memory_id"]) for p in hits]

    def _hybrid_search(self, query: str, user_id: str, top_k: int = 3) -> list[Memory]:
        depth = max(top_k * 3, 8)
        ranked_lists = [
            self._keyword_ids(query, user_id, depth),
            self._vector_ids(query, user_id, depth),
        ]
        scores: dict[int, float] = {}
        for ids in ranked_lists:
            for rank, memory_id in enumerate(ids, start=1):
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(scores, key=lambda memory_id: -scores[memory_id])[:top_k]
        return [self.memories[i] for i in ordered]
