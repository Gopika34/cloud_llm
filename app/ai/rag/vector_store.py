import logging
import numpy as np
from app.db.mongodb import get_collection
from app.utils.ids import generate_uuid

logger = logging.getLogger("uvicorn.error")


class VectorStore:
    def __init__(self):
        self._doc_collection_name = "documents"
        self._chunk_collection_name = "chunks"

    @property
    def documents_collection(self):
        return get_collection(self._doc_collection_name)

    @property
    def chunks_collection(self):
        return get_collection(self._chunk_collection_name)

    async def get_document_by_hash(self, file_hash: str) -> dict | None:
        """Finds an existing document record by its SHA-256 file hash."""
        doc = await self.documents_collection.find_one({"file_hash": file_hash})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
        return None

    async def get_document_by_id(self, document_id: str) -> dict | None:
        """Finds an existing document record by document_id."""
        doc = await self.documents_collection.find_one({"document_id": document_id})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
        return None

    async def insert(
        self,
        document_record: dict,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:
        """Stores document metadata and chunks with their embeddings into MongoDB."""
        # 1. Upsert document record
        await self.documents_collection.update_one(
            {"document_id": document_record["document_id"]},
            {"$set": document_record},
            upsert=True
        )

        # 2. Insert Chunks with embeddings and metadata
        if chunks and embeddings:
            chunk_docs = []
            for chunk, emb in zip(chunks, embeddings):
                chunk_id = chunk.get("id") or generate_uuid()
                chunk_docs.append({
                    "chunk_id": chunk_id,
                    "document_id": document_record["document_id"],
                    "filename": document_record["filename"],
                    "text": chunk["text"],
                    "embedding": emb,
                    "metadata": {
                        "document_id": document_record["document_id"],
                        "filename": document_record["filename"],
                        "chunk_index": int(chunk["chunk_index"]),
                        "page_number": int(chunk["page_number"])
                    }
                })

            if chunk_docs:
                await self.chunks_collection.insert_many(chunk_docs)
                logger.info(
                    f"Stored {len(chunk_docs)} chunks for document '{document_record['filename']}' in MongoDB."
                )

    async def retrieve(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> list[dict]:
        """
        Retrieves top similar chunks for a query embedding using MongoDB Atlas $vectorSearch,
        falling back to in-memory cosine similarity if the Atlas Vector Search index is not yet configured.
        """
        # Attempt MongoDB Atlas $vectorSearch
        try:
            vector_search_stage = {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(n_results * 10, 50),
                    "limit": n_results
                }
            }

            if document_id:
                vector_search_stage["$vectorSearch"]["filter"] = {"document_id": document_id}

            pipeline = [
                vector_search_stage,
                {
                    "$project": {
                        "_id": 1,
                        "chunk_id": 1,
                        "text": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            cursor = self.chunks_collection.aggregate(pipeline)
            raw_results = await cursor.to_list(length=n_results)

            if raw_results:
                formatted_results = []
                for item in raw_results:
                    formatted_results.append({
                        "id": item.get("chunk_id") or str(item.get("_id")),
                        "document": item.get("text", ""),
                        "metadata": item.get("metadata", {}),
                        "distance": 1.0 - float(item.get("score", 1.0))
                    })
                return formatted_results

        except Exception as e:
            logger.warning(
                f"Atlas $vectorSearch failed or index not found: {e}. "
                f"Falling back to in-memory cosine similarity search."
            )

        # Fallback: In-memory cosine similarity search across chunks in MongoDB
        query = {"document_id": document_id} if document_id else {}
        cursor = self.chunks_collection.find(query)
        all_chunks = await cursor.to_list(length=None)

        if not all_chunks:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)

        scored_chunks = []
        for chunk in all_chunks:
            emb = chunk.get("embedding")
            if not emb:
                continue
            c_vec = np.array(emb, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if q_norm == 0 or c_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            scored_chunks.append((sim, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:n_results]

        formatted_results = []
        for sim, chunk in top_chunks:
            formatted_results.append({
                "id": chunk.get("chunk_id") or str(chunk.get("_id")),
                "document": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "distance": 1.0 - sim
            })

        return formatted_results


vector_store = VectorStore()
