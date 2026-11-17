"""User repository for database operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from proctoring_engine.models.database import BackgroundEmbedding, User
from proctoring_engine.utils.exceptions import DatabaseException
from proctoring_engine.utils.exceptions.error_codes import DATABASE_ERROR, USER_ALREADY_EXISTS
from proctoring_engine.utils.logger import get_logger

logger = get_logger("user_repository")


class UserRepository:
    """Repository for user data access operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self.session = session

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email address.

        Args:
            email: User's email address

        Returns:
            User object if found, None otherwise

        Raises:
            DatabaseException: If database query fails
        """
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.info(f"User found by email: user_id={user.id}, email={email}")
            else:
                logger.info(f"No user found with email: {email}")
            
            return user
            
        except SQLAlchemyError as e:
            logger.error(
                f"Database error while fetching user by email: email={email}",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to fetch user due to database error",
                error_code=DATABASE_ERROR,
                details={"email": email},
            ) from e

    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username.

        Args:
            username: User's username

        Returns:
            User object if found, None otherwise

        Raises:
            DatabaseException: If database query fails
        """
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.info(f"User found by username: user_id={user.id}, username={username}")
            else:
                logger.info(f"No user found with username: {username}")
            
            return user
            
        except SQLAlchemyError as e:
            logger.error(
                f"Database error while fetching user by username: username={username}",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to fetch user due to database error",
                error_code=DATABASE_ERROR,
                details={"username": username},
            ) from e

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID.

        Args:
            user_id: User's unique identifier

        Returns:
            User object if found, None otherwise

        Raises:
            DatabaseException: If database query fails
        """
        try:
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.info(f"User found by ID: user_id={user_id}")
            else:
                logger.info(f"No user found with ID: {user_id}")
            
            return user
            
        except SQLAlchemyError as e:
            logger.error(
                f"Database error while fetching user by ID: user_id={user_id}",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to fetch user due to database error",
                error_code=DATABASE_ERROR,
                details={"user_id": user_id},
            ) from e

    async def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        profile_photo: bytes,
        face_embedding: list[float],
    ) -> User:
        """Create a new user in the database.

        Args:
            username: User's username
            email: User's email address
            hashed_password: Bcrypt hashed password
            profile_photo: Binary profile photo (BYTEA)
            face_embedding: Face embedding vector (512 dimensions)

        Returns:
            Created User object

        Raises:
            DatabaseException: If user creation fails
        """
        try:
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                profile_photo=profile_photo,
                face_embedding=face_embedding,
                created_by=email,
                updated_by=email,
            )
            
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            
            logger.info(
                f"User created successfully: user_id={user.id}, username={username}, email={email}"
            )
            return user
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(
                f"User creation failed due to integrity error: username={username}, email={email}",
                exc_info=True
            )
            raise DatabaseException(
                message=f"User with username '{username}' or email '{email}' already exists",
                error_code=USER_ALREADY_EXISTS,
                details={"username": username, "email": email},
            ) from e
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                f"Database error during user creation: username={username}, email={email}",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to create user due to database error",
                error_code=DATABASE_ERROR,
                details={"username": username, "email": email},
            ) from e


    async def calculate_cosine_similarity(
        self,
        stored_embedding: list[float],
        current_embedding: list[float],
    ) -> float:
        """Calculate cosine similarity using NumPy.

        Uses NumPy to calculate cosine similarity directly.
        Formula: similarity = (A · B) / (||A|| * ||B||)
        Range: [-1, 1] where 1 = identical, 0 = unrelated

        Args:
            stored_embedding: Stored face embedding from database (512D)
            current_embedding: Current face embedding to compare (512D)

        Returns:
            Cosine similarity score (1 = identical, 0 = unrelated)

        Raises:
            DatabaseException: If similarity calculation fails
        """
        try:
            import numpy as np
            
            logger.info("Calculating cosine similarity using NumPy")
            
            # Convert to numpy arrays
            query_vector = np.array(current_embedding, dtype=np.float64)
            embedding_vector = np.array(stored_embedding, dtype=np.float64)
            
            # Calculate cosine similarity directly
            # Formula: similarity = (A · B) / (||A|| * ||B||)
            similarity_score = np.dot(query_vector, embedding_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(embedding_vector)
            )
            
            # Convert to Python float
            similarity_score = float(similarity_score)
            
            logger.info(f"Cosine similarity score: {similarity_score:.4f}")
            
            return similarity_score
            
        except Exception as e:
            logger.error(
                "Error calculating cosine similarity",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to calculate similarity due to calculation error",
                error_code=DATABASE_ERROR,
                details={"error": str(e)},
            ) from e

    async def update_background_embedding(
        self,
        user_id: str,
        background_embedding: list[float],
    ) -> str:
        """Create a new background embedding record for the user.

        This creates a new record in the background_embeddings table instead of
        overwriting the old one, allowing tracking of multiple background registrations.

        Args:
            user_id: User's unique identifier
            background_embedding: Background embedding vector (768 dimensions)

        Returns:
            Background embedding ID (UUID) if successful

        Raises:
            DatabaseException: If database insert fails
        """
        try:
            # Get user to verify existence
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user is None:
                logger.warning(f"Cannot create background embedding: user_id={user_id} not found")
                raise DatabaseException(
                    message=f"User not found: {user_id}",
                    error_code=DATABASE_ERROR,
                    details={"user_id": user_id},
                )
            
            # Create new background embedding record
            bg_embedding = BackgroundEmbedding(
                user_id=user_id,
                embedding=background_embedding,
                created_by=user.email,
            )
            
            self.session.add(bg_embedding)
            await self.session.commit()
            await self.session.refresh(bg_embedding)
            
            logger.info(
                f"Background embedding created successfully: "
                f"bg_id={bg_embedding.id} user_id={user_id} embedding_dim={len(background_embedding)}"
            )
            
            return bg_embedding.id
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                f"Database error while creating background embedding: user_id={user_id}",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to create background embedding due to database error",
                error_code=DATABASE_ERROR,
                details={"user_id": user_id},
            ) from e

    async def get_latest_background_embedding(
        self,
        user_id: str,
    ) -> tuple[str, list[float]] | None:
        """Get the latest background embedding for a user.

        Fetches the most recent background embedding based on created_at timestamp.

        Args:
            user_id: User's unique identifier

        Returns:
            Tuple of (background_id, embedding) if found, None otherwise

        Raises:
            DatabaseException: If database query fails
        """
        try:
            # Get latest background embedding ordered by created_at DESC
            result = await self.session.execute(
                select(BackgroundEmbedding)
                .where(BackgroundEmbedding.user_id == user_id)
                .order_by(BackgroundEmbedding.created_at.desc())
                .limit(1)
            )
            bg_embedding = result.scalar_one_or_none()
            
            if bg_embedding:
                logger.info(
                    f"Latest background embedding found: "
                    f"bg_id={bg_embedding.id} user_id={user_id} "
                    f"created_at={bg_embedding.created_at}"
                )
                return (bg_embedding.id, bg_embedding.embedding)
            else:
                logger.info(f"No background embedding found for user_id={user_id}")
                return None
            
        except SQLAlchemyError as e:
            logger.error(
                f"Database error while fetching latest background embedding: user_id={user_id}",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to fetch background embedding due to database error",
                error_code=DATABASE_ERROR,
                details={"user_id": user_id},
            ) from e

    async def get_background_embedding_by_id(
        self,
        background_id: str,
    ) -> list[float] | None:
        """Get a specific background embedding by ID.

        Args:
            background_id: Background embedding ID

        Returns:
            Embedding vector if found, None otherwise

        Raises:
            DatabaseException: If database query fails
        """
        try:
            result = await self.session.execute(
                select(BackgroundEmbedding)
                .where(BackgroundEmbedding.id == background_id)
            )
            bg_embedding = result.scalar_one_or_none()
            
            if bg_embedding:
                logger.info(
                    f"Background embedding found by ID: "
                    f"bg_id={background_id} user_id={bg_embedding.user_id}"
                )
                return bg_embedding.embedding
            else:
                logger.info(f"No background embedding found with ID: {background_id}")
                return None
            
        except SQLAlchemyError as e:
            logger.error(
                f"Database error while fetching background embedding by ID: bg_id={background_id}",
                exc_info=True
            )
            raise DatabaseException(
                message="Failed to fetch background embedding due to database error",
                error_code=DATABASE_ERROR,
                details={"background_id": background_id},
            ) from e


    # Alias for backward compatibility
    get_by_id = get_user_by_id
