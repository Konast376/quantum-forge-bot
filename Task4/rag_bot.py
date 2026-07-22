from __future__ import annotations

import os
import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import Client

INDEX_DIR = Path(os.getenv("INDEX_DIR", "../faiss_index"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "8"))
CONTEXT_K = int(os.getenv("CONTEXT_K", "4"))
MAX_L2_DISTANCE = float(os.getenv("MAX_L2_DISTANCE", "1.05"))
UNKNOWN = "I don't know."

SYSTEM = """Answer only from CURRENT_CONTEXT in the last user message.
Earlier messages are format examples only and must not supply facts.
If the answer is absent, return answer = "I don't know." and evidence = "NONE".
Evidence must be an exact quotation from CURRENT_CONTEXT.
Use no more than two brief grounded steps."""

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        },
        "answer": {"type": "string"},
    },
    "required": ["evidence", "steps", "answer"],
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" \"'")


def load_store() -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


def retrieve(store: FAISS, query: str) -> list[tuple]:
    pairs = store.similarity_search_with_score(query, k=RETRIEVE_K)
    return [(doc, float(distance)) for doc, distance in pairs
            if float(distance) <= MAX_L2_DISTANCE][:CONTEXT_K]


def make_context(hits: list[tuple]) -> str:
    blocks = []
    for number, (doc, distance) in enumerate(hits, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", number)
        blocks.append(
            f"[S{number}] source={source}; chunk={chunk_id}; "
            f"distance={distance:.3f}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def answer(client: Client, store: FAISS, query: str) -> str:
    hits = retrieve(store, query)
    if not hits:
        return UNKNOWN

    context = make_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "FORMAT EXAMPLE ONLY\nCURRENT_CONTEXT:\n[S1] Kaelen Starfire "
                "is a young farmer from Dusthold.\nQUESTION: Who is Kaelen Starfire?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                '{"evidence":"Kaelen Starfire is a young farmer from Dusthold.",'
                '"steps":["The passage identifies Kaelen Starfire.",'
                '"The answer is restated from the passage."],'
                '"answer":"Kaelen Starfire is a young farmer from Dusthold. [S1]"}'
            ),
        },
        {
            "role": "user",
            "content": f"CURRENT TASK\nCURRENT_CONTEXT:\n{context}\n\nQUESTION:\n{query}",
        },
    ]

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        format=FORMAT_SCHEMA,
        options={"temperature": 0, "seed": 42, "num_predict": 180},
    )

    import json

    try:
        data = json.loads(response.message.content)
    except (json.JSONDecodeError, AttributeError):
        return UNKNOWN

    evidence = str(data.get("evidence", "")).strip()
    final_answer = str(data.get("answer", "")).strip()
    steps = data.get("steps", [])

    if not final_answer or "don't know" in final_answer.lower():
        return UNKNOWN
    if not evidence or evidence.upper() == "NONE":
        return UNKNOWN
    if normalise(evidence) not in normalise(context):
        return UNKNOWN

    source_lines = []
    for number, (doc, _) in enumerate(hits, start=1):
        source_lines.append(f"[S{number}] {doc.metadata.get('source', 'unknown')}")

    reasoning = "\n".join(
        f"{i}. {step}" for i, step in enumerate(steps[:2], start=1)
    )
    return (
        f"Evidence: {evidence}\nReasoning:\n{reasoning}\n"
        f"Answer: {final_answer}\nSources:\n" + "\n".join(source_lines)
    )


def main() -> None:
    store = load_store()
    client = Client(host=OLLAMA_HOST)
    print(f"RAG bot via Ollama model {OLLAMA_MODEL}. Type exit to stop.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        started = time.perf_counter()
        print(f"\n{answer(client, store, query)}")
        print(f"Time: {time.perf_counter() - started:.2f}s\n{'-' * 72}")


if __name__ == "__main__":
    main()
