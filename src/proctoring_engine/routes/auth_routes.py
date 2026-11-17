"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from proctoring_engine.database.connection import get_db_session
from proctoring_engine.models.schemas import (
    UserLoginRequest,
    UserLoginResponse,
    UserRegisterRequest,
    UserRegisterResponse,
)
from proctoring_engine.services.dependencies import get_auth_service, get_face_service, get_user_repository
from proctoring_engine.utils.exceptions import AuthenticationException, DatabaseException, ValidationException
from proctoring_engine.utils.exceptions.error_responses import ErrorResponse
from proctoring_engine.utils.logger import get_logger

logger = get_logger("auth_routes")

router = APIRouter(prefix="/api/v1", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "User registered successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status_code": 201,
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "message": "User registered successfully",
                    }
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Bad request - validation error"},
        409: {"model": ErrorResponse, "description": "Conflict - user already exists"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def register(
    username: str = Form(..., description="Unique username"),
    email: str = Form(..., description="User email address"),
    password: str = Form(..., description="User password"),
    profile_photo: UploadFile = File(..., description="Profile photo (raw image bytes)"),
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Register a new user with username, email, password, and profile photo.

    Accepts raw image bytes from frontend (multipart/form-data).
    The profile_photo should be sent as raw frame bytes (numpy array/bytes).
    
    Process:
    1. Check if user already exists (by email and username)
    2. Segment face using MediaPipe + convex hull
    3. Generate 512D face embedding using InsightFace
    4. Hash password with bcrypt
    5. Save user to database with embedding
    
    Args:
        username: Unique username
        email: User email address
        password: Plain text password (will be hashed)
        profile_photo: Raw image bytes
        session: Database session (injected)
    
    Returns:
        JSONResponse with status_code, user_id, and message
    """
    try:
        photo_bytes = await profile_photo.read()

        logger.info(
            f"Registration request received: username={username}, email={email}, "
            f"photo_size={len(photo_bytes)} bytes"
        )

        # Create registration request with raw bytes
        registration_request = UserRegisterRequest(
            username=username,
            email=email,
            password=password,
            profile_photo=photo_bytes,
        )

        # Get dependencies
        user_repository = await get_user_repository(session)
        face_service = await get_face_service()
        auth_service = await get_auth_service(user_repository, face_service)
        
        # Register user
        result = await auth_service.register_user(registration_request)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status_code": 201,
                "user_id": result.user_id,
                "message": result.message
            },
        )

    except ValidationException as e:
        logger.warning(
            f"Validation error during registration: {e.message}",
            extra={"error_code": e.error_code, "details": e.details}
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status_code": 400,
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details
            },
        )
        
    except DatabaseException as e:
        logger.error(
            f"Database error during registration: {e.message}",
            extra={"error_code": e.error_code, "details": e.details}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": 500,
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details
            },
        )
        
    except Exception:
        logger.exception("Unexpected error during registration")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": 500,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": None,
            },
        )


@router.post(
    "/login",
    response_model=UserLoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Login successful",
            "content": {
                "application/json": {
                    "example": {
                        "status_code": 200,
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "bearer",
                        "message": "Login successful",
                    }
                }
            },
        },
        401: {
            "model": ErrorResponse,
            "description": "Unauthorized - invalid credentials"
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error"
        },
    },
)
async def login(
    request: UserLoginRequest,
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Authenticate user with email and password.
    
    Args:
        request: Login request with email and password
        session: Database session (injected)
    
    Returns:
        JSONResponse with status_code, user_id, and message
    """
    try:
        # Get dependencies
        user_repository = await get_user_repository(session)
        auth_service = await get_auth_service(user_repository)
        
        # Authenticate user
        result = await auth_service.login_user(request)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status_code": 200,
                "user_id": result.user_id,
                "access_token": result.access_token,
                "token_type": result.token_type,
                "message": result.message,
            },
        )

    except AuthenticationException as e:
        logger.warning(
            f"Authentication error during login: {e.message}",
            extra={"error_code": e.error_code, "details": e.details}
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "status_code": 401,
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details
            },
        )
        
    except DatabaseException as e:
        logger.error(
            f"Database error during login: {e.message}",
            extra={"error_code": e.error_code, "details": e.details}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": 500,
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details
            },
        )
        
    except Exception:
        logger.exception("Unexpected error during login")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": 500,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": None,
            },
        )
