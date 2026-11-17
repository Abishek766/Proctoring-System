"""FastAPI dependency injection — provides the SessionManager singleton."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from proctoring_engine.repositories.user_repository import UserRepository
from proctoring_engine.services.auth_service import AuthService
from proctoring_engine.services.face_service import FaceService
from proctoring_engine.services.session_manager import SessionManager
from proctoring_engine.settings import get_settings


@lru_cache(maxsize=1)
def get_session_manager() -> SessionManager:
    """Return the application-wide SessionManager singleton."""
    return SessionManager(get_settings())


async def get_user_repository(session: AsyncSession) -> UserRepository:
    """Get user repository instance.
    
    Args:
        session: Database session
    
    Returns:
        UserRepository instance
    """
    return UserRepository(session)


async def get_face_service() -> FaceService:
    """Get face service instance.
    
    Returns:
        FaceService instance
    """
    return FaceService()


async def get_auth_service(
    user_repository: UserRepository,
    face_service: FaceService | None = None
) -> AuthService:
    """Get auth service instance.
    
    Args:
        user_repository: User repository instance
        face_service: Optional face service instance (required for registration)
    
    Returns:
        AuthService instance
    """
    return AuthService(user_repository, face_service)


async def get_face_verification_service(
    user_repository: UserRepository,
    face_service: FaceService
) -> "FaceVerificationService":
    """Get face verification service instance.
    
    Args:
        user_repository: User repository instance
        face_service: Face service instance
    
    Returns:
        FaceVerificationService instance
    """
    from proctoring_engine.services.face_verification_service import FaceVerificationService
    return FaceVerificationService(user_repository, face_service)
