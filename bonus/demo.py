from __future__ import annotations

import sys

from agent import HybridMemoryAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def seed(agent: HybridMemoryAgent) -> None:
    agent.remember(
        "Kubernetes notes: I read about autoscaling a cloud service by combining "
        "Horizontal Pod Autoscaler with CPU metrics and request latency. The key "
        "idea was to keep enough replicas during traffic spikes without wasting "
        "money during quiet hours.",
    )
    agent.remember(
        "Cloud security note: JWT should be short lived, refresh tokens need safer "
        "storage, and sensitive APIs should combine OAuth scopes with audit logs. "
        "For Vietnamese users, account recovery must be clear because support chats "
        "often mix Vietnamese explanations with English product terms.",
    )
    agent.remember(
        "Reading plan: I want to continue with cloud infrastructure topics, especially "
        "Kubernetes networking, autoscaling, and secure API gateways. I read slowly "
        "when documents are all English, so Vietnamese summaries help.",
    )
    agent.remember(
        "RAG and vector search note: hybrid retrieval works better than pure keyword "
        "when the query is a Vietnamese paraphrase, but exact terms like Kubernetes "
        "and JWT should still be preserved for BM25.",
    )


def main() -> int:
    agent = HybridMemoryAgent()
    seed(agent)

    queries = [
        "Tôi đã đọc gì về Kubernetes?",
        "Recommend đọc gì tiếp",
        "Tôi đang quan tâm gì gần đây?",
        "Tài liệu về tự động mở rộng hạ tầng?",
        "Cho tôi summary cloud security",
    ]

    for i, query in enumerate(queries, start=1):
        print("=" * 80)
        print(f"Query {i}: {query}")
        print(agent.recall(query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
