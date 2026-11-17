"""Authentication service for business logic."""

from __future__ import annotations

import bcrypt

from proctoring_engine.models.schemas import (
    UserLoginRequest,
    UserLoginResponse,
    UserRegisterRequest,
    UserRegisterResponse,
)
from proctoring_engine.repositories.user_repository import UserRepository
from proctoring_engine.services.face_service import FaceService
from proctoring_engine.utils.exceptions import AuthenticationException, ValidationException
from proctoring_engine.utils.exceptions.error_codes import INVALID_CREDENTIALS, USER_ALREADY_EXISTS
from proctoring_engine.utils.jwt_handler import create_access_token
from proctoring_engine.utils.logger import get_logger

logger = get_logger("auth_service")


class AuthService:
    """Service for authentication business logic."""

    def __init__(self, user_repository: UserRepository, face_service: FaceService | None = None) -> None:
        """Initialize service with user repository and optional face service.
        
        Args:
            user_repository: Repository for user data access
            face_service: Service for face processing (required for registration)
        """
        self.user_repository = user_repository
        self.face_service = face_service

    async def register_user(self, request: UserRegisterRequest) -> UserRegisterResponse:
        """Register a new user with face processing.

        Args:
            request: User registration request data

        Returns:
            UserRegisterResponse with user_id and success message

        Raises:
            ValidationException: If user already exists or face processing fails
        """
        logger.info(f"Attempting to register user: username={request.username}, email={request.email}")

        # Check if user already exists by email
        existing_user_by_email = await self.user_repository.get_user_by_email(request.email)
        if existing_user_by_email:
            logger.warning(f"Registration failed: email already exists: {request.email}")
            raise ValidationException(
                message=f"User with email '{request.email}' already exists",
                error_code=USER_ALREADY_EXISTS,
                details={"field": "email", "value": request.email},
            )

        # Check if user already exists by username
        existing_user_by_username = await self.user_repository.get_user_by_username(request.username)
        if existing_user_by_username:
            logger.warning(f"Registration failed: username already exists: {request.username}")
            raise ValidationException(
                message=f"User with username '{request.username}' already exists",
                error_code=USER_ALREADY_EXISTS,
                details={"field": "username", "value": request.username},
            )

        # Process face image (segmentation + embedding)
        if not self.face_service:
            raise ValidationException(
                message="Face service not available",
                error_code="FACE_SERVICE_UNAVAILABLE",
                details=None,
            )

        logger.info("Processing face image for registration")
        face_result = await self.face_service.process_face_for_registration(request.profile_photo)
        processed_photo = face_result["processed_photo"]
        face_embedding = face_result["face_embedding"]
        logger.info(
            f"Face processing completed: photo_size={len(processed_photo)} bytes, "
            f"embedding_size={len(face_embedding)}"
        )

        # Hash password
        hashed_password = self._hash_password(request.password)

        # Create user with processed photo and embeddings
        user = await self.user_repository.create_user(
            username=request.username,
            email=request.email,
            hashed_password=hashed_password,
            profile_photo=processed_photo,
            face_embedding=face_embedding,
        )

        logger.info(f"User registered successfully: user_id={user.id}, username={user.username}")
        return UserRegisterResponse(
            status_code=201,
            user_id=user.id,
            message="User registered successfully"
        )

    async def login_user(self, request: UserLoginRequest) -> UserLoginResponse:
        """Authenticate user and return user ID.

        Args:
            request: User login request data

        Returns:
            UserLoginResponse with user_id and success message

        Raises:
            AuthenticationException: If credentials are invalid
        """
        logger.info(f"Login attempt for email: {request.email}")

        # Get user by email
        user = await self.user_repository.get_user_by_email(request.email)
        if not user:
            logger.warning(f"Login failed: user not found: {request.email}")
            raise AuthenticationException(
                message="Invalid email or password",
                error_code=INVALID_CREDENTIALS,
                details={"email": request.email},
            )

        # Verify password
        if not self._verify_password(request.password, user.hashed_password):
            logger.warning(f"Login failed: invalid password for user: {request.email}")
            raise AuthenticationException(
                message="Invalid email or password",
                error_code=INVALID_CREDENTIALS,
                details={"email": request.email},
            )

        logger.info(f"User logged in successfully: user_id={user.id}, email={user.email}")
        access_token = create_access_token(user.id)
        return UserLoginResponse(
            status_code=200,
            user_id=user.id,
            access_token=access_token,
            message="Login successful"
        )

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hashed password.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            True if password matches, False otherwise
        """
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
