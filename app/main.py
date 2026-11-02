from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat
from contextlib import asynccontextmanager
from app.ai.rag.pipeline import rag_pipeline
import logging
from pathlib import Path
from app.db.mongodb import connect_to_mongo, close_mongo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("uvicorn.error")

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()

    logger.info("App startup: data ingestion is running...")
    await rag_pipeline.data_ingestion(UPLOADS_DIR)
    
    yield

    await close_mongo()
    logger.info("App shutdown")



app = FastAPI(lifespan=lifespan)

app.add_middleware(
CORSMiddleware,
allow_origins=["http://localhost:5173"],
allow_credentials=True,
allow_methods=["GET", "POST"],
allow_headers=["Content-Type", "Authorization"],
)

# @app.get("/api/profile")
# def profile():
# return {"name": "Ava", "role": "Developer"}


@app.get("/")
def landing_page():
    return {"message":"Hi this is your backend"}

@app.get("/health")
def health():
    return {" status": "ok"}

app.include_router(chat.router)

