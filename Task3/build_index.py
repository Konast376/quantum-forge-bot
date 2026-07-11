from __future__ import annotations

import time
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

DATA_DIR = Path("../knowledge_base")
INDEX_DIR = Path("../faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 30


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Knowledge-base folder not found: {DATA_DIR.resolve()}")

    files = sorted([*DATA_DIR.rglob("*.txt"), *DATA_DIR.rglob("*.md")])
    if not files:
        raise FileNotFoundError("No .txt or .md documents were found.")

    documents = []
    for path in files:
        loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            document.metadata.update(
                {
                    "source": str(path),
                    "title": path.stem,
                    "document_id": path.relative_to(DATA_DIR).as_posix(),
                }
            )
        documents.extend(loaded)

    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedding_tokenizer,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
    )
    chunks = splitter.split_documents(documents)

    per_document_counter: dict[str, int] = {}
    for global_id, chunk in enumerate(chunks):
        document_id = str(chunk.metadata.get("document_id", "unknown"))
        local_id = per_document_counter.get(document_id, 0)
        per_document_counter[document_id] = local_id + 1
        chunk.metadata["chunk_id"] = f"{global_id}"
        chunk.metadata["chunk_in_document"] = local_id

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    started = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    elapsed = time.perf_counter() - started

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("Embedding dimension: 384")
    print(f"Indexing time: {elapsed:.2f}s")
    print(f"Saved to: {INDEX_DIR.resolve()}")

    test_query = "Who is Kaelen Starfire?"
    print(f"\nTest query: {test_query}")
    for rank, (doc, distance) in enumerate(
        vectorstore.similarity_search_with_score(test_query, k=3), start=1
    ):
        print(
            f"\n#{rank} distance={float(distance):.3f} "
            f"source={doc.metadata.get('source')} "
            f"chunk={doc.metadata.get('chunk_id')}\n"
            f"{doc.page_content[:500]}"
        )


if __name__ == "__main__":
    main()
