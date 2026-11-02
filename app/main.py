
import asyncio
import logging
import os

from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.rag.pipeline import rag_pipeline
from app.db.mongodb import close_mongo, connect_to_mongo
from app.routes import chat, file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("uvicorn.error")

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"


async def run_initial_ingestion():
    try:
        logger.info("Starting background document ingestion...")

        result = await rag_pipeline.data_ingestion(UPLOADS_DIR)

        logger.info(
            "Initial ingestion completed: "
            f"{result['ingested']} ingested, "
            f"{result['skipped']} skipped, "
            f"{result['failed']} failed."
        )

    except Exception:
        logger.exception("Background document ingestion failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):

    await connect_to_mongo()

    ingestion_task = asyncio.create_task(
        run_initial_ingestion()
    )

    app.state.ingestion_task = ingestion_task

    logger.info("Application startup completed.")

    try:
        yield

    finally:
        if not ingestion_task.done():
            ingestion_task.cancel()

            with suppress(asyncio.CancelledError):
                await ingestion_task

        await close_mongo()

        logger.info("Application shutdown completed.")


app = FastAPI(
    title="Cloud LLM API",
    description="Document-based RAG application with Gemini and Groq.",
    version="1.0.0",
    lifespan=lifespan,
)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def landing_page():
    return {
        "message": "Cloud LLM Backend is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "cloud-llm",
    }


app.include_router(chat.router)
app.include_router(file.router)
