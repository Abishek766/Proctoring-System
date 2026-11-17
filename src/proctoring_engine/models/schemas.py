"""Pydantic models for authentication request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr


class UserLoginRequest(BaseModel):
    """User login request model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "john@example.com",
                "password": "SecurePass123!"
            }
        }
    )

    email: EmailStr
    """User email address"""

    password: str
    """User password"""


class UserLoginResponse(BaseModel):
    """User login response model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_code": 200,
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "message": "Login successful"
            }
        }
    )

    status_code: int = 200
    """HTTP status code"""

    user_id: str
    """User unique identifier"""

    access_token: str
    """JWT bearer token"""

    token_type: str = "bearer"
    """Token type"""

    message: str
    """Success message"""


class UserRegisterRequest(BaseModel):
    """User registration request model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "password": "SecurePass123!"
            }
        }
    )

    username: str
    """Unique username"""

    email: EmailStr
    """User email address"""

    password: str
    """User password (will be hashed)"""

    profile_photo: bytes
    """Raw image bytes from frontend (multipart/form-data)"""


class UserRegisterResponse(BaseModel):
    """User registration response model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_code": 201,
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "User registered successfully"
            }
        }
    )

    status_code: int = 201
    """HTTP status code"""

    user_id: str
    """User unique identifier"""

    message: str
    """Success message"""


class FaceVerificationRequest(BaseModel):
    """Face verification request model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "source_type": "verify"
            }
        }
    )

    user_id: str
    """User unique identifier to verify against"""

    image: bytes
    """Raw image bytes from frontend (multipart/form-data)"""

    source_type: str
    """Source type: 'verify' for normal verification, 'violation_verify' for violation check"""


class FaceVerificationResponse(BaseModel):
    """Face verification response model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_code": 200,
                "verified": True,
                "similarity_score": 0.85,
                "message": "Candidate verified successfully"
            }
        }
    )

    status_code: int = 200
    """HTTP status code"""

    verified: bool
    """Whether the face was verified (similarity >= 0.70)"""

    similarity_score: float
    """Cosine similarity score between 0 and 1"""

    message: str
    """Verification result message"""
