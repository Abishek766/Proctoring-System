"""Database models using SQLAlchemy ORM."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class User(Base):
    """User database model."""
    
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    """User unique identifier (UUID)"""

    username: Mapped[str] = mapped_column(
        String(50), 
        unique=True, 
        nullable=False, 
        index=True
    )
    """Unique username"""

    email: Mapped[str] = mapped_column(
        String(100), 
        unique=True, 
        nullable=False, 
        index=True
    )
    """Unique email address"""

    hashed_password: Mapped[str] = mapped_column(
        String(255), 
        nullable=False
    )
    """Bcrypt hashed password"""

    profile_photo: Mapped[bytes] = mapped_column(
        LargeBinary, 
        nullable=False
    )
    """Binary profile photo (BYTEA)"""

    face_embedding: Mapped[list[float]] = mapped_column(
        Vector(512), 
        nullable=False
    )
    """Face embedding vector (512 dimensions using pgvector)"""

    background_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(768),
        nullable=True
    )
    """Background embedding vector (768 dimensions using pgvector, from DINOv2)"""

    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, 
        nullable=False
    )
    """Timestamp when record was created"""

    created_by: Mapped[str] = mapped_column(
        String(100), 
        nullable=False
    )
    """User who created this record (email or system)"""

    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, 
        onupdate=datetime.utcnow, 
        nullable=False
    )
    """Timestamp when record was last updated"""

    updated_by: Mapped[str] = mapped_column(
        String(100), 
        nullable=False
    )
    """User who last updated this record (email or system)"""

    def __repr__(self) -> str:
        """String representation of User."""
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class BackgroundEmbedding(Base):
    """Background embedding database model for tracking multiple background registrations per user."""
    
    __tablename__ = "background_embeddings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    """Background embedding unique identifier (UUID)"""

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    """Foreign key to users table"""

    embedding: Mapped[list[float]] = mapped_column(
        Vector(768),
        nullable=False
    )
    """Background embedding vector (768 dimensions using pgvector, from DINOv2)"""

    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    """Timestamp when background was registered"""

    created_by: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    """User who created this record (email or system)"""

    def __repr__(self) -> str:
        """String representation of BackgroundEmbedding."""
        return f"<BackgroundEmbedding(id={self.id}, user_id={self.user_id}, created_at={self.created_at})>"
