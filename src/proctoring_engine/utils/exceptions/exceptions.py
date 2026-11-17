"""Custom exception classes for the proctoring engine."""


class ProctoringEngineError(Exception):
    """Base exception for all proctoring engine errors."""

    def __init__(self, message: str, error_code: str = "") -> None:
        super().__init__(message)
        self.error_code = error_code


class ModelInitError(ProctoringEngineError):
    """Raised when a model fails to initialise."""


class CameraInitError(ProctoringEngineError):
    """Raised when the camera cannot be opened."""


class AudioThreadError(ProctoringEngineError):
    """Raised when the audio capture thread encounters a fatal error."""


class AuthenticationException(Exception):
    """Exception raised for authentication errors."""

    def __init__(self, message: str, error_code: str, details: dict | None = None) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


class DatabaseException(Exception):
    """Exception raised for database errors."""

    def __init__(self, message: str, error_code: str, details: dict | None = None) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


class ValidationException(Exception):
    """Exception raised for validation errors (face detection, image quality, etc.)."""

    def __init__(self, message: str, error_code: str, details: dict | None = None) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)
