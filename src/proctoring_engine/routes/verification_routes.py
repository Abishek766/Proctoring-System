"""Face verification routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from proctoring_engine.database.connection import get_db_session
from proctoring_engine.models.schemas import (
    FaceVerificationRequest,
    FaceVerificationResponse,
)
from proctoring_engine.services.dependencies import (
    get_face_service,
    get_face_verification_service,
    get_user_repository,
)
from proctoring_engine.utils.exceptions import AuthenticationException, DatabaseException, ValidationException
from proctoring_engine.utils.exceptions.error_responses import ErrorResponse
from proctoring_engine.utils.logger import get_logger

logger = get_logger("verification_routes")

router = APIRouter(prefix="/api/v1", tags=["Verification"])


@router.post(
    "/verify",
    response_model=FaceVerificationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Face verification completed",
            "content": {
                "application/json": {
                    "example": {
                        "status_code": 200,
                        "verified": True,
                        "similarity_score": 0.85,
                        "message": "Candidate verified successfully",
                    }
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Bad request - validation error"},
        401: {"model": ErrorResponse, "description": "Unauthorized - user not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def verify_face(
    user_id: str = Form(..., description="User ID to verify against"),
    image: UploadFile = File(..., description="Face image for verification (raw image bytes)"),
    source_type: str = Form(..., description="Source type: 'verify' or 'violation_verify'"),
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Verify user's face against stored embeddings.

    Accepts image file upload via multipart/form-data.
    The image is read as raw bytes and processed for face verification.

    Two verification flows based on source_type:
    
    1. source_type = "verify" (Normal verification):
       - Full image quality validation (brightness, sharpness, resolution)
       - Face detection (must be exactly 1 face)
       - Face segmentation using MediaPipe Face Mesh + convex hull
       - Generate 512D face embedding using InsightFace ArcFace
       - Compare with stored embeddings using PostgreSQL cosine similarity
       - Return verified=True if similarity >= 0.70 (70%)
    
    2. source_type = "violation_verify" (Violation check):
       - Skip image quality validation
       - Face detection (handles multiple faces, processes first detected face)
       - Face segmentation using MediaPipe Face Mesh + convex hull
       - Generate 512D face embedding using InsightFace ArcFace
       - Compare with stored embeddings using PostgreSQL cosine similarity
       - Return similarity score for violation analysis
    
    Args:
        user_id: User's unique identifier (UUID from database)
        image: Image file upload (JPEG/PNG)
        source_type: "verify" or "violation_verify"
        session: Database session (injected)
    
    Returns:
        JSONResponse with verification result, similarity score, and message
    """
    try:
        # Read image bytes from UploadFile
        image_bytes = await image.read()
        
        logger.info(
            f"Verification request received: user_id={user_id}, "
            f"source_type={source_type}, image_size={len(image_bytes)} bytes"
        )

        # Validate source_type
        if source_type not in ["verify", "violation_verify"]:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status_code": 400,
                    "error_code": "INVALID_SOURCE_TYPE",
                    "message": f"Invalid source_type: {source_type}. Must be 'verify' or 'violation_verify'",
                    "details": {"source_type": source_type},
                },
            )

        # Create verification request
        verification_request = FaceVerificationRequest(
            user_id=user_id,
            image=image_bytes,
            source_type=source_type,
        )

        # Get dependencies
        user_repository = await get_user_repository(session)
        face_service = await get_face_service()
        verification_service = await get_face_verification_service(user_repository, face_service)

        # Verify face
        result = await verification_service.verify_face(verification_request)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status_code": 200,
                "verified": result.verified,
                "similarity_score": result.similarity_score,
                "message": result.message,
            },
        )

    except ValidationException as e:
        logger.warning(
            f"Validation error during verification: {e.message}",
            extra={"error_code": e.error_code, "details": e.details}
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status_code": 400,
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )

    except AuthenticationException as e:
        logger.warning(
            f"Authentication error during verification: {e.message}",
            extra={"error_code": e.error_code, "details": e.details}
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "status_code": 401,
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )

    except DatabaseException as e:
        logger.error(
            f"Database error during verification: {e.message}",
            extra={"error_code": e.error_code, "details": e.details}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": 500,
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )

    except Exception:
        logger.exception("Unexpected error during verification")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": 500,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": None,
            },
        )
