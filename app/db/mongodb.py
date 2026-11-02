from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

logger = logging.getLogger("uvicorn")

class Database:
    client: AsyncIOMotorClient | None = None

db_instance = Database()

def get_database():
    if db_instance.client is None:
        raise RuntimeError("MongoDB is not connected. Call connect_to_mongo() first.")
    return db_instance.client[settings.DB_NAME]

def get_collection(name: str):
    return get_database()[name]

def db_client():
    return get_database()

def vector_collection():
    return get_collection("chunks")


async def connect_to_mongo():
    try:
        logger.info("⏳ Connecting to MongoDB...")
        print("MONGO_URL loaded:", settings.MONGO_URL[:20] + "..." if settings.MONGO_URL else "MISSING")
        db_instance.client = AsyncIOMotorClient(settings.MONGO_URL)
        
        # THE PING TEST (Crucial for Cloud)
        await db_instance.client.admin.command('ping')
        logger.info("✅ MongoDB Connected Successfully!")
        
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")
        raise e


async def close_mongo():
    if db_instance.client:
        db_instance.client.close()        
        logger.info("🔒 MongoDB connection closed.")