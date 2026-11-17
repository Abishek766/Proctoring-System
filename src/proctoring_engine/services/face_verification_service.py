"""Face verification service for comparing face embeddings."""

from __future__ import annotations

import asyncio
import numpy as np

from proctoring_engine.models.schemas import (
    FaceVerificationRequest,
    FaceVerificationResponse,
)
from proctoring_engine.repositories.user_repository import UserRepository
from proctoring_engine.services.face_service import FaceService
from proctoring_engine.settings import get_settings
from proctoring_engine.utils.exceptions import AuthenticationException, ValidationException
from proctoring_engine.utils.exceptions.error_codes import USER_NOT_FOUND
from proctoring_engine.utils.logger import get_logger

logger = get_logger("face_verification_service")
verify_logger = get_logger("face_verification")


class FaceVerificationService:
    """Service for face verification operations."""

    def __init__(self, user_repository: UserRepository | None = None, face_service: FaceService | None = None) -> None:
        """Initialize service with user repository and face service.
        
        Args:
            user_repository: Repository for user data access (optional, lazy-loaded)
            face_service: Service for face processing (optional, lazy-loaded)
        """
        self.user_repository = user_repository
        self.face_service = face_service
        self._settings = get_settings()
        verify_logger.info("[INIT] FaceVerificationService initialized")

    async def verify_face(self, request: FaceVerificationRequest) -> FaceVerificationResponse:
        """Verify face against stored embeddings.

        Two flows based on source_type:
        1. "verify" - Full validation with image quality checks
        2. "violation_verify" - Skip image quality checks, handle multiple faces

        Args:
            request: Face verification request data

        Returns:
            FaceVerificationResponse with verification result

        Raises:
            AuthenticationException: If user not found
            ValidationException: If face processing fails
        """
        verify_logger.info(
            f"[VERIFY_FACE_START] user_id={request.user_id} source_type={request.source_type}"
        )

        # Step 1: Validate user exists and fetch stored embeddings
        user = await self.user_repository.get_user_by_id(request.user_id)
        if not user:
            verify_logger.warning(f"[USER_NOT_FOUND] user_id={request.user_id}")
            raise AuthenticationException(
                message=f"User with ID '{request.user_id}' not found",
                error_code=USER_NOT_FOUND,
                details={"user_id": request.user_id},
            )

        stored_embedding = user.face_embedding
        verify_logger.info(
            f"[USER_FOUND] user_id={request.user_id} username={user.username} "
            f"has_embedding={stored_embedding is not None}"
        )

        # Step 2: Process face based on source_type
        if request.source_type == "verify":
            # Flow 1: Full validation (normal verification)
            verify_logger.info("[PROCESSING] Full validation (verify flow)")
            face_result = await self.face_service.process_face_for_verification(
                request.image,
                validate_quality=True
            )
        else:
            # Flow 2: Skip quality validation (violation check)
            verify_logger.info("[PROCESSING] Skip quality validation (violation_verify flow)")
            face_result = await self.face_service.process_face_for_verification(
                request.image,
                validate_quality=False
            )

        current_embedding = face_result["face_embedding"]
        verify_logger.info(
            f"[EMBEDDING_EXTRACTED] user_id={request.user_id} "
            f"embedding_shape={current_embedding.shape if hasattr(current_embedding, 'shape') else 'N/A'}"
        )

        # Step 3: Calculate cosine similarity using NumPy
        # Formula: distance = (A · B) / (||A|| * ||B||)
        # Similarity = 1 - distance
        similarity_score = await self.user_repository.calculate_cosine_similarity(
            stored_embedding,
            current_embedding
        )
        
        verify_logger.info(
            f"[SIMILARITY_CALCULATED] user_id={request.user_id} "
            f"similarity={similarity_score:.4f} threshold={self._settings.face_similarity_threshold}"
        )

        # Step 4: Determine verification result based on threshold
        # Only include results above threshold (default: 0.60 = 60%)
        verified = similarity_score >= self._settings.face_similarity_threshold
        
        if verified:
            message = "Candidate verified successfully"
            verify_logger.info(
                f"[MATCH_SUCCESS] user_id={request.user_id} username={user.username} "
                f"similarity={similarity_score:.4f}"
            )
        else:
            message = "Verification failed: Face does not match"
            verify_logger.warning(
                f"[MATCH_FAILED] user_id={request.user_id} username={user.username} "
                f"similarity={similarity_score:.4f} (threshold={self._settings.face_similarity_threshold})"
            )

        return FaceVerificationResponse(
            status_code=200,
            verified=verified,
            similarity_score=round(similarity_score, 4),
            message=message
        )

    async def verify_from_frame(
        self,
        frame: np.ndarray,
        user_id: str,
        skip_quality_check: bool = False,
    ) -> dict:
        """Simple verification: use new multi-face detection for violation scenarios.
        
        Args:
            frame: BGR frame from video
            user_id: User ID to verify
            skip_quality_check: If True, skip brightness/sharpness validation (for initial verification)
            
        Returns:
            Dictionary with verification result
        """
        verify_logger.info(f"[FRAME_VERIFY_START] user_id={user_id} skip_quality_check={skip_quality_check}")
        
        # Use the new verify_all_faces_in_frame function for violation scenarios
        return await self.verify_all_faces_in_frame(frame, user_id, skip_quality_check=skip_quality_check)

    def verify_user_identity_sync(
        self,
        frame: np.ndarray,
        user_id: str,
        db_session,
    ) -> dict:
        """Simple face verification: fetch embedding, generate embedding from frame, compare.
        
        Args:
            frame: BGR frame to verify
            user_id: User ID to fetch embedding for
            db_session: Database session
            
        Returns:
            Dictionary with verification result
        """
        verify_logger.info(f"[VERIFY_START] user_id={user_id} frame_shape={frame.shape}")
        
        try:
            # Step 1: Fetch stored embedding from database
            from proctoring_engine.repositories.user_repository import UserRepository  # noqa: PLC0415
            import asyncio  # noqa: PLC0415
            
            async def get_embedding():
                repo = UserRepository(db_session)
                user = await repo.get_user_by_id(user_id)
                if not user or not user.face_embedding:
                    return None
                return user.face_embedding
            
            stored_embedding = asyncio.run(get_embedding())
            
            if stored_embedding is None:
                verify_logger.warning(f"[NO_EMBEDDING] user_id={user_id}")
                return {
                    "is_match": False,
                    "similarity": 0.0,
                    "confidence": "none",
                    "message": "No stored embedding found",
                    "user_id": user_id,
                    "username": None,
                }
            
            verify_logger.debug(f"[EMBEDDING_FETCHED] user_id={user_id}")
            
            # Step 2: Convert BGR to RGB
            import cv2  # noqa: PLC0415
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Step 3: Detect and segment face
            from proctoring_engine.core.face_detection import detect_face_with_mediapipe  # noqa: PLC0415
            from proctoring_engine.core.image_processing import segment_face  # noqa: PLC0415
            
            try:
                face_result = detect_face_with_mediapipe(rgb_frame)
                bbox = face_result["bbox"]
                segmented_face = segment_face(rgb_frame, bbox)
                verify_logger.debug(f"[FACE_SEGMENTED] user_id={user_id}")
            except Exception as e:
                # Use full frame if face detection fails
                segmented_face = rgb_frame
                verify_logger.warning(f"[USING_FULL_FRAME] user_id={user_id} reason={str(e)}")
            
            # Step 4: Generate embedding from segmented face
            from proctoring_engine.core.face_recognition import generate_face_embedding  # noqa: PLC0415
            current_embedding = generate_face_embedding(segmented_face)
            verify_logger.debug(f"[EMBEDDING_GENERATED] user_id={user_id}")
            
            # Step 5: Convert stored embedding to list if bytes
            if isinstance(stored_embedding, bytes):
                stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32).tolist()
            
            # Step 6: Calculate similarity using repository function
            async def calc_similarity():
                repo = UserRepository(db_session)
                return await repo.calculate_cosine_similarity(
                    stored_embedding,
                    current_embedding.tolist()
                )
            
            similarity = asyncio.run(calc_similarity())
            
            verify_logger.info(
                f"[SIMILARITY] user_id={user_id} similarity={similarity:.4f} threshold={self._settings.face_similarity_threshold}"
            )
            
            # Step 7: Determine result
            is_match = similarity >= self._settings.face_similarity_threshold
            
            if similarity >= 0.85:
                confidence = "high"
            elif similarity >= 0.75:
                confidence = "medium"
            elif similarity >= 0.65:
                confidence = "low"
            else:
                confidence = "none"
            
            message = "Identity verified" if is_match else "Identity mismatch"
            
            verify_logger.info(
                f"[RESULT] user_id={user_id} is_match={is_match} similarity={similarity:.4f} confidence={confidence}"
            )
            
            return {
                "is_match": is_match,
                "similarity": round(similarity, 4),
                "confidence": confidence,
                "message": message,
                "user_id": user_id,
                "username": None,
            }
            
        except Exception as exc:
            verify_logger.error(f"[ERROR] user_id={user_id} error={str(exc)}", exc_info=True)
            return {
                "is_match": False,
                "similarity": 0.0,
                "confidence": "error",
                "message": f"Verification error: {str(exc)}",
                "user_id": user_id,
                "username": None,
            }



    async def verify_all_faces_in_frame(
        self,
        frame: np.ndarray,
        user_id: str,
        skip_quality_check: bool = False,
    ) -> dict:
        """Verify ALL faces in frame for violation scenarios.
        
        Detects all faces, generates embeddings for each, and checks similarity.
        If ANY face fails similarity check, returns face mismatch.
        
        Args:
            frame: BGR frame from video
            user_id: User ID to verify against
            skip_quality_check: If True, skip brightness/sharpness validation (for initial verification)
            
        Returns:
            Dictionary with verification result including all face results
        """
        verify_logger.info(f"[VERIFY_ALL_FACES_START] user_id={user_id}")
        
        try:
            import cv2  # noqa: PLC0415
            from proctoring_engine.core.face_detection import (  # noqa: PLC0415
                detect_all_faces_for_violation,
                get_face_landmarks,
                create_convex_hull_mask,
            )
            from proctoring_engine.core.face_recognition import generate_face_embedding  # noqa: PLC0415
            from proctoring_engine.core.image_processing import crop_face_region  # noqa: PLC0415
            from proctoring_engine.core.image_validation import (  # noqa: PLC0415
                validate_brightness,
                validate_sharpness,
            )
            from proctoring_engine.utils.exceptions import ValidationException  # noqa: PLC0415
            
            # Step 0: Validate image quality (brightness and sharpness)
            # This prevents processing poor quality frames that would result in low similarity
            # SKIP for initial verification to avoid blocking session start
            if not skip_quality_check:
                try:
                    verify_logger.debug(f"[QUALITY_CHECK_START] user_id={user_id}")
                    
                    # Check brightness
                    brightness_info = validate_brightness(frame)
                    verify_logger.debug(
                        f"[BRIGHTNESS_OK] user_id={user_id} "
                        f"brightness={brightness_info['brightness']:.2f} "
                        f"status={brightness_info['status']}"
                    )
                    
                    # Check sharpness (blur detection)
                    sharpness_info = validate_sharpness(frame)
                    verify_logger.debug(
                        f"[SHARPNESS_OK] user_id={user_id} "
                        f"sharpness={sharpness_info['sharpness']:.2f} "
                        f"status={sharpness_info['status']}"
                    )
                    
                    verify_logger.info(f"[QUALITY_CHECK_PASSED] user_id={user_id}")
                    
                except ValidationException as ve:
                    # Image quality check failed - return error to trigger retry
                    verify_logger.warning(
                        f"[QUALITY_CHECK_FAILED] user_id={user_id} "
                        f"reason={ve.message} error_code={ve.error_code}"
                    )
                    return {
                        "is_match": False,
                        "similarity": 0.0,
                        "confidence": "none",
                        "message": f"Image quality check failed: {ve.message}",
                        "user_id": user_id,
                        "username": None,
                        "faces_detected": 0,
                        "face_results": [],
                    }
            else:
                verify_logger.info(f"[QUALITY_CHECK_SKIPPED] user_id={user_id} reason=skip_quality_check=True")
            
            # Step 1: Convert BGR to RGB for MediaPipe detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Step 2: Detect ALL faces (no exceptions)
            faces = detect_all_faces_for_violation(rgb_frame)
            
            if not faces:
                verify_logger.warning(f"[NO_FACES] user_id={user_id}")
                return {
                    "is_match": False,
                    "similarity": 0.0,
                    "confidence": "none",
                    "message": "No faces detected in frame",
                    "user_id": user_id,
                    "username": None,
                    "faces_detected": 0,
                    "face_results": [],
                }
            
            verify_logger.info(f"[FACES_DETECTED] user_id={user_id} count={len(faces)}")
            
            # Step 3: Get stored embedding from database (same as verify_face)
            user = await self.user_repository.get_user_by_id(user_id)
            if not user:
                verify_logger.warning(f"[USER_NOT_FOUND] user_id={user_id}")
                return {
                    "is_match": False,
                    "similarity": 0.0,
                    "confidence": "none",
                    "message": "User not found",
                    "user_id": user_id,
                    "username": None,
                    "faces_detected": len(faces),
                    "face_results": [],
                }
            
            stored_embedding = user.face_embedding
            verify_logger.info(
                f"[USER_FOUND] user_id={user_id} username={user.username} "
                f"has_embedding={stored_embedding is not None}"
            )
            
            if stored_embedding is None:
                verify_logger.warning(f"[NO_STORED_EMBEDDING] user_id={user_id}")
                return {
                    "is_match": False,
                    "similarity": 0.0,
                    "confidence": "none",
                    "message": "No stored embedding found for user",
                    "user_id": user_id,
                    "username": user.username,
                    "faces_detected": len(faces),
                    "face_results": [],
                }
            
            # Step 4: Process each face using REGISTRATION FLOW (get landmarks → segment → crop → embed)
            face_results = []
            any_match = False
            best_similarity = 0.0
            
            for face in faces:
                try:
                    bbox = face["bbox"]
                    face_idx = face["face_index"]
                    verify_logger.info(f"[PROCESS_FACE_{face_idx}] bbox={bbox}")
                    
                    # Step 4a: Get facial landmarks from FULL RGB frame (same as registration)
                    verify_logger.info(f"[STEP_4a] Getting landmarks for face {face_idx} from FULL frame")
                    landmarks = get_face_landmarks(rgb_frame)
                    
                    if landmarks is None:
                        verify_logger.warning(f"[NO_LANDMARKS] face_index={face_idx} - skipping")
                        continue
                    
                    verify_logger.info(f"[LANDMARKS_OK] face_index={face_idx} points={len(landmarks)}")
                    
                    # Step 4b: Create convex hull mask (same as registration)
                    verify_logger.info(f"[STEP_4b] Creating convex hull mask for face {face_idx}")
                    face_mask = create_convex_hull_mask(rgb_frame, landmarks)
                    verify_logger.info(f"[MASK_CREATED] face_index={face_idx}")
                    
                    # Step 4c: Apply mask to segment face from FULL frame (same as registration)
                    verify_logger.info(f"[STEP_4c] Segmenting face {face_idx} from FULL frame")
                    segmented_face_rgb = cv2.bitwise_and(rgb_frame, rgb_frame, mask=face_mask)
                    
                    non_zero_pixels = np.count_nonzero(segmented_face_rgb)
                    verify_logger.info(f"[FACE_SEGMENTED] face_index={face_idx} non_zero_pixels={non_zero_pixels}")
                    
                    # Step 4d: Convert segmented RGB to BGR for InsightFace (same as registration)
                    segmented_face_bgr = cv2.cvtColor(segmented_face_rgb, cv2.COLOR_RGB2BGR)
                    verify_logger.info(f"[STEP_4d] Converted to BGR for face {face_idx}")
                    
                    # Step 4e: Crop the segmented face with padding (same as registration)
                    verify_logger.info(f"[STEP_4e] Cropping segmented face {face_idx} with 50% padding")
                    cropped_segmented_face = crop_face_region(segmented_face_bgr, bbox)
                    
                    if cropped_segmented_face.size == 0 or cropped_segmented_face.shape[0] < 50 or cropped_segmented_face.shape[1] < 50:
                        verify_logger.warning(
                            f"[CROPPED_TOO_SMALL] face_index={face_idx} shape={cropped_segmented_face.shape} - skipping"
                        )
                        continue
                    
                    verify_logger.info(f"[FACE_CROPPED] face_index={face_idx} shape={cropped_segmented_face.shape}")
                    
                    # Debug: Save the cropped segmented face
                    try:
                        debug_path = f"debug_crop_image_face{face_idx}_{user_id[:8]}.jpg"
                        cv2.imwrite(debug_path, cropped_segmented_face)
                        verify_logger.info(
                            f"[DEBUG_IMAGE_SAVED] face_index={face_idx} path={debug_path} "
                            f"shape={cropped_segmented_face.shape}"
                        )
                    except Exception as debug_exc:
                        verify_logger.warning(f"[DEBUG_IMAGE_SAVE_FAILED] {debug_exc}")
                    
                    # Step 4f: Generate embedding from cropped segmented face (same as registration)
                    verify_logger.info(f"[STEP_4f] Generating embedding for face {face_idx}")
                    current_embedding = generate_face_embedding(cropped_segmented_face)
                    
                    verify_logger.info(
                        f"[EMBEDDING_GENERATED] face_index={face_idx} "
                        f"embedding_shape={current_embedding.shape}"
                    )
                    
                    # Calculate similarity with stored embedding
                    similarity = await self.user_repository.calculate_cosine_similarity(
                        stored_embedding,
                        current_embedding.tolist()
                    )
                    
                    is_match = similarity >= self._settings.face_similarity_threshold
                    if is_match:
                        any_match = True
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                    
                    face_results.append({
                        "face_index": face_idx,
                        "similarity": round(similarity, 4),
                        "is_match": is_match,
                        "bbox": bbox,
                    })
                    
                    verify_logger.info(
                        f"[FACE_VERIFIED] user_id={user_id} face_index={face_idx} "
                        f"similarity={similarity:.4f} is_match={is_match}"
                    )
                    
                except Exception as face_exc:
                    verify_logger.error(
                        f"[FACE_PROCESS_ERROR] user_id={user_id} face_index={face.get('face_index')} "
                        f"error={str(face_exc)}"
                    )
                    continue
            
            # Step 5: Determine overall result
            # If ANY face matches, consider it a match
            # If NO faces match, it's a mismatch
            confidence = "high" if best_similarity >= 0.85 else "medium" if best_similarity >= 0.75 else "low" if best_similarity >= 0.65 else "none"
            
            if any_match:
                message = f"Identity verified: candidate face found among {len(faces)} detected faces"
            else:
                message = f"Identity mismatch: none of the {len(faces)} detected faces match the candidate"
            
            verify_logger.info(
                f"[VERIFY_ALL_FACES_COMPLETE] user_id={user_id} "
                f"faces_detected={len(faces)} any_match={any_match} "
                f"best_similarity={best_similarity:.4f}"
            )
            
            return {
                "is_match": any_match,
                "similarity": round(best_similarity, 4),
                "confidence": confidence,
                "message": message,
                "user_id": user_id,
                "username": user.username,
                "faces_detected": len(faces),
                "face_results": face_results,
            }
            
        except Exception as exc:
            verify_logger.error(
                f"[VERIFY_ALL_FACES_ERROR] user_id={user_id} error={str(exc)}",
                exc_info=True
            )
            return {
                "is_match": False,
                "similarity": 0.0,
                "confidence": "error",
                "message": f"Verification error: {str(exc)}",
                "user_id": user_id,
                "username": None,
                "faces_detected": 0,
                "face_results": [],
            }
