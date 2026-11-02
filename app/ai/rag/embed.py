# import asyncio
# from sentence_transformers import SentenceTransformer


# class Embedding:
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         self.model = SentenceTransformer(model_name)

#     async def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
#         if not chunks:
#             return []
            
#         embeddings = await asyncio.to_thread(
#             self.model.encode,
#             chunks,
#             batch_size=10,
#             convert_to_numpy=True,
#             show_progress_bar=False
#         )        
#         return embeddings.tolist()

#     async def embed_query(self, prompt: str) -> list[float]:
#         embedding = await asyncio.to_thread(
#             self.model.encode,
#             prompt,
#             convert_to_numpy=True,
#             show_progress_bar=False
#         )
#         return embedding.tolist()


# embedding = Embedding()


import asyncio
import logging
import time

from google import genai
from google.genai import types
from app.core.config import settings


logger = logging.getLogger("uvicorn.error")


class Embedding:

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        output_dimensions: int = 768,
        request_interval: float = 5.0
    ):
        self.model_name = model_name
        self.output_dimensions = output_dimensions
        self.request_interval = request_interval

        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

        self._request_lock = asyncio.Lock()
        self._last_request_time = 0.0

    async def _embed(
        self,
        texts: list[str],
        task_type: str
    ) -> list[list[float]]:

        if not texts:
            return []

        async with self._request_lock:

            elapsed = time.monotonic() - self._last_request_time
            wait_time = self.request_interval - elapsed

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            try:
                response = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.output_dimensions
                    )
                )

                self._last_request_time = time.monotonic()

                embeddings = [
                    item.values for item in response.embeddings
                ]

                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} embeddings, "
                        f"received {len(embeddings)}."
                    )

                return embeddings

            except Exception:
                self._last_request_time = time.monotonic()
                logger.exception("Gemini embedding request failed.")
                raise

    async def embed_chunks(
        self,
        chunks: list[str]
    ) -> list[list[float]]:

        return await self._embed(
            chunks,
            "RETRIEVAL_DOCUMENT"
        )

    async def embed_query(self, prompt: str) -> list[float]:

        embeddings = await self._embed(
            [prompt],
            "RETRIEVAL_QUERY"
        )

        return embeddings[0]


embedding = Embedding()