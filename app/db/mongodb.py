import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

logger = logging.getLogger("uvicorn.error")


class Database:
    client: AsyncIOMotorClient | None = None


db_instance = Database()


def get_database():
    if db_instance.client is None:
        raise RuntimeError(
            "MongoDB is not connected. Call connect_to_mongo() first."
        )

    return db_instance.client[settings.DB_NAME]


def get_collection(name: str):
    return get_database()[name]

def db_client():
    return get_database()


def vector_collection():
    return get_collection("chunks")

async def connect_to_mongo():
    if not settings.MONGO_URL:
        raise RuntimeError(
            "MONGO_URL is missing from environment variables."
        )

    try:
        logger.info("Connecting to MongoDB...")

        db_instance.client = AsyncIOMotorClient(
            settings.MONGO_URL,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )

        await db_instance.client.admin.command("ping")

        logger.info("MongoDB connected successfully.")

    except Exception:
        logger.exception("MongoDB connection failed.")

        if db_instance.client:
            db_instance.client.close()
            db_instance.client = None

        raise


async def close_mongo():
    if db_instance.client:
        db_instance.client.close()
        db_instance.client = None

        logger.info("MongoDB connection closed.")