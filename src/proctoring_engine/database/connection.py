"""Database connection configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from proctoring_engine.models.database import Base
from proctoring_engine.utils.logger import get_logger

logger = get_logger("database")

# Fix for Windows + asyncpg compatibility
if sys.platform == 'win32':
    import asyncio
    # Use selector event loop on Windows for asyncpg compatibility
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logger.info("Set Windows event loop policy for asyncpg compatibility")

# Load .env file from project root
_ENV_FILE = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
    logger.info(f"Loaded environment variables from {_ENV_FILE}")
else:
    logger.warning(f".env file not found at {_ENV_FILE}")

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Global variables for engine and session factory (initialized lazily)
engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker | None = None  # ✅ Changed type hint


def _get_engine() -> AsyncEngine:
    """Get or create the database engine (lazy initialization)."""
    global engine
    if engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        
        engine = create_async_engine(
            DATABASE_URL,
            echo=False,  # Set to True for SQL query logging
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # ✅ Added: Verify connections before using
        )
        logger.info("Database engine created")
    return engine


def _get_session_factory() -> async_sessionmaker:  # ✅ Changed return type
    """Get or create the session factory (lazy initialization)."""
    global AsyncSessionLocal
    if AsyncSessionLocal is None:
        AsyncSessionLocal = async_sessionmaker(  # ✅ Changed to async_sessionmaker
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,  # ✅ Added
            autocommit=False,  # ✅ Added
        )
        logger.info("Database session factory created")
    return AsyncSessionLocal


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection.
    
    Yields:
        AsyncSession: Database session
    
    Raises:
        RuntimeError: If DATABASE_URL is not configured
    
    Example:
        @router.get("/users")
        async def get_users(session: AsyncSession = Depends(get_db_session)):
            # Use session here
            pass
    """
    if not DATABASE_URL:
        raise RuntimeError("Database is not configured. Set DATABASE_URL environment variable.")
    
    session_factory = _get_session_factory()
    async with session_factory() as session:
        yield session
        # ✅ That's it! Context manager handles commit/rollback/close


async def init_db() -> None:
    """Initialize database tables."""
    if not DATABASE_URL:
        logger.info("Database not configured - skipping table creation")
        return
    
    db_engine = _get_engine()
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def close_db() -> None:
    """Close database connections."""
    global engine
    if engine is None:
        return
    
    await engine.dispose()
    logger.info("Database connections closed")
    engine = None
