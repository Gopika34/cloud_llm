import asyncio
from sentence_transformers import SentenceTransformer


class Embedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    async def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
            
        embeddings = await asyncio.to_thread(
            self.model.encode,
            chunks,
            batch_size=10,
            convert_to_numpy=True,
            show_progress_bar=False
        )        
        return embeddings.tolist()

    async def embed_query(self, prompt: str) -> list[float]:
        embedding = await asyncio.to_thread(
            self.model.encode,
            prompt,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.tolist()


embedding = Embedding()

