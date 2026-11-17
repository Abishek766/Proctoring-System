"""WebSocket endpoint for real-time video proctoring.

Protocol
--------
1. Client connects to  ws://<host>/ws/proctor/{session_id}
2. Server immediately sends a SessionStartResponse JSON.
3. Client sends messages at 30 FPS — two message types are supported:

   TEXT message — frame JSON (sent every frame):
       {
           "session_id": "candidate-001",
           "frame_id": 1,
           "timestamp_ms": 1700000000000,
           "frame_b64": "<base64 JPEG>",
           "audio_b64": "<base64 float32 PCM, optional>"
       }

   BINARY message — raw audio bytes (sent separately when available):
       Raw float32 PCM bytes, 16 kHz mono.
       No framing or header — just the raw IEEE 754 float32 samples.
       The engine buffers these until it has enough for one YAMNet inference.

4. Server accumulates 30 frames (~1 second), then sends a WindowResponse JSON.
5. Client disconnects when the exam ends; server cleans up the session.
"""

from __future__ import annotations

import asyncio
import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from proctoring_engine.models.api_models import (
    ErrorResponse,
    FrameMessage,
    SessionStartResponse,
)
from proctoring_engine.services.dependencies import get_session_manager
from proctoring_engine.settings import get_settings
from proctoring_engine.utils.logger import get_logger

logger     = get_logger("video_pipeline")   # existing engine logger
api_logger = get_logger("api_session")      # dedicated API session log → api_sessions.log
error_logger = get_logger("error_logger")   # dedicated error log → errors.log
verify_logger = get_logger("face_verification")  # dedicated verification log → face_verification.log

# Get settings instance
settings = get_settings()

router = APIRouter(tags=["proctoring"])


@router.websocket("/ws/proctor/{session_id}")
async def proctor_ws(
    websocket: WebSocket, 
    session_id: str,
    user_id: str | None = Query(None, description="User ID for identity verification")
) -> None:
    """WebSocket endpoint — one connection per candidate session.
    
    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier
        user_id: Optional user ID for identity verification (triggers face verification on violations)
    """
    manager = get_session_manager()
    await websocket.accept()

    api_logger.info(
        f"[CONNECT] session={session_id} user_id={user_id} "
        f"client={websocket.client.host if websocket.client else 'unknown'} "
        f"active_sessions={manager.active_count + 1}"
    )

    # ── Get database session for face verification ────────────────────────────
    db_session = None
    if user_id:
        try:
            from proctoring_engine.database.connection import get_db_session  # noqa: PLC0415
            # Create async database session
            async for session in get_db_session():
                db_session = session
                break  # Get the first session
            api_logger.info(f"[DB_SESSION_CREATED] session={session_id} user_id={user_id}")
        except Exception as exc:
            error_logger.error(
                f"[DB_SESSION_ERROR] session={session_id} user_id={user_id} error={str(exc)}",
                exc_info=True
            )
            api_logger.warning(f"[DB_SESSION_FAILED] session={session_id} user_id={user_id} - continuing without verification")
            # Continue without verification capability
    else:
        api_logger.info(f"[NO_USER_ID] session={session_id} - face verification disabled")

    # ── Create session ─────────────────────────────────────────────────────────
    try:
        session = await manager.create_session(session_id, user_id=user_id, db_session=db_session)
    except ValueError as exc:
        api_logger.warning(f"[REJECT] session={session_id} reason=session_exists")
        await websocket.send_text(
            ErrorResponse(
                session_id=session_id,
                error="session_exists",
                detail=str(exc),
            ).model_dump_json()
        )
        await websocket.close(code=4000)
        return

    # Confirm connection to client
    await websocket.send_text(
        SessionStartResponse(session_id=session_id).model_dump_json()
    )

    session_start = time.perf_counter()
    frame_count   = 0
    audio_chunks  = 0
    window_count  = 0
    
    # ── Initial identity verification flags ────────────────────────────────────
    initial_verification_done = False
    initial_verification_required = user_id is not None and db_session is not None
    initial_verification_in_progress = False  # NEW: Track if initial verification is running
    
    # ── Violation verification state ───────────────────────────────────────────
    violation_verification_in_progress = False  # NEW: Track if violation verification is running
    
    # ── Periodic verification configuration (from settings) ────────────────────
    VERIFICATION_INTERVAL_SECONDS = settings.verification_interval_seconds
    MAX_VERIFICATION_ATTEMPTS = settings.max_verification_attempts
    FRAMES_BETWEEN_ATTEMPTS = settings.frames_between_attempts
    
    last_verification_time = time.perf_counter()
    verification_in_progress = False
    verification_attempts = 0
    
    # ── Background validation configuration ────────────────────────────────────
    last_background_validation_time = time.perf_counter()
    background_validation_in_progress = False

    try:
        while True:
            # ── Receive next message (text or binary) ──────────────────────────
            data = await websocket.receive()

            # ── Binary message → raw audio bytes ──────────────────────────────
            if data.get("bytes") is not None:
                raw_audio = data["bytes"]
                try:
                    # Trim to nearest multiple of 4 (float32 = 4 bytes)
                    trimmed = raw_audio[: len(raw_audio) - len(raw_audio) % 4]
                    if not trimmed:
                        continue
                    pcm = np.frombuffer(trimmed, dtype=np.float32).copy()
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, session.engine.process_audio, pcm
                    )
                    audio_chunks += 1
                except Exception as exc:
                    api_logger.warning(
                        f"[AUDIO_ERR] session={session_id} "
                        f"bytes={len(raw_audio)} error={exc}"
                    )
                continue

            # ── Text message → frame JSON ──────────────────────────────────────
            raw = data.get("text")
            if raw is None:
                break

            # ── Parse ──────────────────────────────────────────────────────────
            try:
                msg = FrameMessage.model_validate_json(raw)
            except ValidationError as exc:
                api_logger.warning(
                    f"[PARSE_ERR] session={session_id} "
                    f"frame={frame_count + 1} error={exc.error_count()} validation errors"
                )
                await websocket.send_text(
                    ErrorResponse(
                        session_id=session_id,
                        error="invalid_message",
                        detail=str(exc),
                    ).model_dump_json()
                )
                continue

            # ── Decode JPEG frame ──────────────────────────────────────────────
            try:
                bgr_frame = _decode_jpeg(msg.frame_b64)
            except Exception as exc:
                api_logger.warning(
                    f"[FRAME_ERR] session={session_id} "
                    f"frame_id={msg.frame_id} error={exc}"
                )
                await websocket.send_text(
                    ErrorResponse(
                        session_id=session_id,
                        error="invalid_frame",
                        detail=str(exc),
                    ).model_dump_json()
                )
                continue

            frame_count += 1

            # ── Periodic verification timer check ──────────────────────────────
            current_time = time.perf_counter()
            time_since_last_verification = current_time - last_verification_time
            
            # Start verification cycle when interval has passed
            # COORDINATION: Skip if initial or violation verification is running
            # COORDINATION: Skip if person violations (multiple_persons, no_person) are active
            if time_since_last_verification >= VERIFICATION_INTERVAL_SECONDS and not verification_in_progress:
                if user_id and db_session:
                    # Check if other verifications are blocking
                    if initial_verification_in_progress or violation_verification_in_progress:
                        verify_logger.info(
                            f"[PERIODIC_SKIPPED] session={session_id} user_id={user_id} "
                            f"reason=other_verification_in_progress "
                            f"initial_in_progress={initial_verification_in_progress} "
                            f"violation_in_progress={violation_verification_in_progress} "
                            f"elapsed={time_since_last_verification:.1f}s "
                            f"will_retry_next_cycle=True"
                        )
                        # Don't reset last_verification_time - will try again next cycle
                    # Check if person violations are active (higher priority than periodic)
                    elif "multiple_persons" in session.active_violations or "no_person" in session.active_violations:
                        verify_logger.info(
                            f"[PERIODIC_SKIPPED] session={session_id} user_id={user_id} "
                            f"reason=person_violations_active "
                            f"multiple_persons_active={'multiple_persons' in session.active_violations} "
                            f"no_person_active={'no_person' in session.active_violations} "
                            f"elapsed={time_since_last_verification:.1f}s "
                            f"will_retry_next_cycle=True "
                            f"note=violation_verification_has_higher_priority"
                        )
                        # Don't reset last_verification_time - will try again next cycle
                    else:
                        verification_in_progress = True
                        verification_attempts = 0
                        
                        # Sync state with engine for coordination
                        session.engine.periodic_verification_in_progress = True
                        
                        # Log to both API and verification logs
                        api_logger.info(
                            f"[PERIODIC_VERIFICATION_START] session={session_id} "
                            f"user_id={user_id} elapsed={time_since_last_verification:.1f}s "
                            f"max_attempts={MAX_VERIFICATION_ATTEMPTS}"
                        )
                        verify_logger.info(
                            f"[PERIODIC_CYCLE_START] session={session_id} user_id={user_id} "
                            f"interval={VERIFICATION_INTERVAL_SECONDS}s elapsed={time_since_last_verification:.1f}s "
                            f"max_attempts={MAX_VERIFICATION_ATTEMPTS} frames_between={FRAMES_BETWEEN_ATTEMPTS}"
                        )
            
            # ── Periodic verification attempts (every Nth frame) ───────────────
            if verification_in_progress:
                if frame_count % FRAMES_BETWEEN_ATTEMPTS == 0:
                    verification_attempts += 1
                    
                    verify_logger.info(
                        f"[PERIODIC_FRAME_CHECK] session={session_id} user_id={user_id} "
                        f"frame_count={frame_count} "
                        f"verification_in_progress={verification_in_progress} "
                        f"frame_divisible_by_{FRAMES_BETWEEN_ATTEMPTS}=True "
                        f"will_attempt_verification=YES "
                        f"attempt_number={verification_attempts}"
                    )
                    
                    api_logger.info(
                        f"[PERIODIC_VERIFICATION_ATTEMPT] session={session_id} "
                        f"attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                        f"frame_id={msg.frame_id}"
                    )
                    verify_logger.info(
                        f"[PERIODIC_ATTEMPT_{verification_attempts}] session={session_id} user_id={user_id} "
                        f"frame_id={msg.frame_id} timestamp_ms={msg.timestamp_ms} "
                        f"frame_count={frame_count}"
                    )
                    
                    try:
                        from proctoring_engine.services.face_verification_service import FaceVerificationService  # noqa: PLC0415
                        from proctoring_engine.repositories.user_repository import UserRepository  # noqa: PLC0415
                        from proctoring_engine.services.face_service import FaceService  # noqa: PLC0415
                        
                        user_repo = UserRepository(db_session)
                        face_service = FaceService()
                        verification_service = FaceVerificationService(user_repo, face_service)
                        
                        # Verify using current frame
                        verification_start = time.perf_counter()
                        verification_result = await verification_service.verify_from_frame(
                            bgr_frame,
                            user_id
                        )
                        verification_latency_ms = round((time.perf_counter() - verification_start) * 1000.0, 1)
                        
                        api_logger.info(
                            f"[PERIODIC_VERIFICATION_RESULT] session={session_id} "
                            f"attempt={verification_attempts} "
                            f"is_match={verification_result['is_match']} "
                            f"similarity={verification_result['similarity']:.4f} "
                            f"confidence={verification_result['confidence']} "
                            f"faces_detected={verification_result.get('faces_detected', 'N/A')}"
                        )
                        
                        # Detailed logging to face_verification.log
                        verify_logger.info(
                            f"[PERIODIC_RESULT] session={session_id} user_id={user_id} "
                            f"attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                            f"is_match={verification_result['is_match']} "
                            f"similarity={verification_result['similarity']:.4f} "
                            f"confidence={verification_result['confidence']} "
                            f"faces_detected={verification_result.get('faces_detected', 0)} "
                            f"username={verification_result.get('username', 'N/A')} "
                            f"message=\"{verification_result.get('message', '')}\" "
                            f"latency_ms={verification_latency_ms} "
                            f"frame_id={msg.frame_id}"
                        )
                        
                        # Log individual face results if available
                        if 'face_results' in verification_result and verification_result['face_results']:
                            for face_result in verification_result['face_results']:
                                verify_logger.info(
                                    f"[PERIODIC_FACE_DETAIL] session={session_id} user_id={user_id} "
                                    f"attempt={verification_attempts} "
                                    f"face_index={face_result.get('face_index', 'N/A')} "
                                    f"similarity={face_result.get('similarity', 0):.4f} "
                                    f"is_match={face_result.get('is_match', False)} "
                                    f"bbox={face_result.get('bbox', [])}"
                                )
                        
                        # Check if verification was successful (face matched)
                        if verification_result['is_match']:
                            api_logger.info(
                                f"[PERIODIC_VERIFICATION_SUCCESS] session={session_id} "
                                f"matched_on_attempt={verification_attempts} "
                                f"similarity={verification_result['similarity']:.4f}"
                            )
                            verify_logger.info(
                                f"[PERIODIC_SUCCESS] session={session_id} user_id={user_id} "
                                f"matched_on_attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                                f"similarity={verification_result['similarity']:.4f} "
                                f"confidence={verification_result['confidence']} "
                                f"total_latency_ms={verification_latency_ms} "
                                f"username={verification_result.get('username', 'N/A')} "
                                f"frame_id={msg.frame_id} timestamp_ms={msg.timestamp_ms}"
                            )
                            
                            # DO NOT store success result - only log it
                            # Frontend only needs to know about failures (face mismatch)
                            verify_logger.info(
                                f"[PERIODIC_FRONTEND_RESPONSE] session={session_id} user_id={user_id} "
                                f"response_type=SUCCESS "
                                f"will_send_to_frontend=NO "
                                f"reason=success_logged_only_no_alert_needed "
                                f"verification_data={{is_match: True, similarity: {verification_result['similarity']:.4f}, "
                                f"confidence: '{verification_result['confidence']}', message: '{verification_result.get('message', '')}'}}"
                            )
                            verification_in_progress = False
                            last_verification_time = current_time
                            
                            # Sync state with engine for coordination
                            session.engine.periodic_verification_in_progress = False
                            
                        elif not verification_result['is_match']:
                            # Check if this is a face mismatch or a technical error
                            faces_detected = verification_result.get('faces_detected', 0)
                            confidence = verification_result.get('confidence', 'none')
                            similarity = verification_result.get('similarity', 0.0)
                            
                            # DECISION LOG: Determine if this is face mismatch or technical error
                            verify_logger.info(
                                f"[PERIODIC_DECISION_CHECK] session={session_id} user_id={user_id} "
                                f"attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                                f"is_match=False "
                                f"faces_detected={faces_detected} "
                                f"confidence={confidence} "
                                f"similarity={similarity:.4f} "
                                f"threshold={settings.face_similarity_threshold}"
                            )
                            
                            if faces_detected > 0 and confidence != 'error':
                                # DECISION: FACE MISMATCH (face detected but similarity too low)
                                verify_logger.warning(
                                    f"[PERIODIC_DECISION_RESULT] session={session_id} user_id={user_id} "
                                    f"decision=FACE_MISMATCH "
                                    f"reason=face_detected_but_similarity_below_threshold "
                                    f"faces_detected={faces_detected} "
                                    f"confidence={confidence} "
                                    f"similarity={similarity:.4f} "
                                    f"threshold={settings.face_similarity_threshold} "
                                    f"will_retry=NO "
                                    f"will_send_alert=YES_IMMEDIATELY"
                                )
                                # FACE MISMATCH DETECTED - Send immediately (don't wait for retries)
                                api_logger.warning(
                                    f"[PERIODIC_FACE_MISMATCH] session={session_id} "
                                    f"face_mismatch_detected_on_attempt={verification_attempts} "
                                    f"similarity={verification_result['similarity']:.4f}"
                                )
                                verify_logger.warning(
                                    f"[PERIODIC_FACE_MISMATCH] session={session_id} user_id={user_id} "
                                    f"attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                                    f"similarity={verification_result['similarity']:.4f} "
                                    f"confidence={verification_result['confidence']} "
                                    f"faces_detected={verification_result.get('faces_detected', 0)} "
                                    f"message=\"{verification_result.get('message', '')}\" "
                                    f"username={verification_result.get('username', 'N/A')} "
                                    f"frame_id={msg.frame_id} timestamp_ms={msg.timestamp_ms}"
                                )
                                
                                # Trigger face verification through the engine to add face_mismatch to active_violations
                                # This will add IdentityVerificationEvent to the frame results IMMEDIATELY
                                verify_logger.warning(
                                    f"[PERIODIC_TRIGGER_VERIFICATION] session={session_id} user_id={user_id} "
                                    f"triggering_face_verification_to_add_to_violations "
                                    f"similarity={verification_result['similarity']:.4f} "
                                    f"reason=face_mismatch_detected_send_immediately"
                                )
                                session.engine.trigger_face_verification(bgr_frame, msg.frame_id, msg.timestamp_ms)
                                
                                verify_logger.warning(
                                    f"[PERIODIC_FRONTEND_RESPONSE] session={session_id} user_id={user_id} "
                                    f"response_type=FAILURE_FACE_MISMATCH "
                                    f"will_send_to_frontend=YES "
                                    f"send_method=active_violations_face_mismatch "
                                    f"similarity={verification_result['similarity']:.4f}"
                                )
                                verify_logger.warning(
                                    f"[PERIODIC_RECOMMENDATION] session={session_id} user_id={user_id} "
                                    f"action=ALERT_PROCTOR "
                                    f"reason=identity_verification_failed_face_mismatch "
                                    f"similarity={verification_result['similarity']:.4f} "
                                    f"threshold_required={settings.face_similarity_threshold} "
                                    f"gap={settings.face_similarity_threshold - verification_result['similarity']:.4f}"
                                )
                                verification_in_progress = False
                                last_verification_time = current_time
                                
                                # Sync state with engine for coordination
                                session.engine.periodic_verification_in_progress = False
                                
                                # Sync state with engine for coordination
                                session.engine.periodic_verification_in_progress = False
                            else:
                                # DECISION: TECHNICAL ERROR (no face detected or processing error)
                                verify_logger.warning(
                                    f"[PERIODIC_DECISION_RESULT] session={session_id} user_id={user_id} "
                                    f"decision=TECHNICAL_ERROR "
                                    f"reason=no_face_detected_or_processing_error "
                                    f"faces_detected={faces_detected} "
                                    f"confidence={confidence} "
                                    f"similarity={similarity:.4f} "
                                    f"current_attempt={verification_attempts} "
                                    f"max_attempts={MAX_VERIFICATION_ATTEMPTS} "
                                    f"will_retry={verification_attempts < MAX_VERIFICATION_ATTEMPTS}"
                                )
                                
                                # TECHNICAL ERROR - No face detected or processing error
                                # Retry if attempts remaining, otherwise just log (don't send to frontend yet)
                                if verification_attempts >= MAX_VERIFICATION_ATTEMPTS:
                                    api_logger.error(
                                        f"[PERIODIC_TECHNICAL_ERROR] session={session_id} "
                                        f"all_attempts_exhausted={MAX_VERIFICATION_ATTEMPTS} "
                                        f"reason=technical_errors_prevented_verification "
                                        f"faces_detected={faces_detected} confidence={confidence}"
                                    )
                                    verify_logger.error(
                                        f"[PERIODIC_TECHNICAL_ERROR] session={session_id} user_id={user_id} "
                                        f"all_attempts_exhausted={MAX_VERIFICATION_ATTEMPTS} "
                                        f"faces_detected={faces_detected} "
                                        f"confidence={confidence} "
                                        f"message=\"{verification_result.get('message', '')}\" "
                                        f"username={verification_result.get('username', 'N/A')} "
                                        f"frame_id={msg.frame_id} timestamp_ms={msg.timestamp_ms}"
                                    )
                                    verify_logger.error(
                                        f"[PERIODIC_RECOMMENDATION] session={session_id} user_id={user_id} "
                                        f"action=CHECK_CAMERA_AND_LIGHTING "
                                        f"reason=verification_failed_due_to_technical_issues "
                                        f"note=not_sent_to_frontend_yet_will_be_handled_later"
                                    )
                                    verify_logger.error(
                                        f"[PERIODIC_RETRY_EXHAUSTED] session={session_id} user_id={user_id} "
                                        f"all_attempts_used={MAX_VERIFICATION_ATTEMPTS} "
                                        f"final_decision=STOP_RETRYING "
                                        f"faces_detected={faces_detected} "
                                        f"confidence={confidence}"
                                    )
                                    verification_in_progress = False
                                    last_verification_time = current_time
                                    
                                    # Sync state with engine for coordination
                                    session.engine.periodic_verification_in_progress = False
                                else:
                                    # Still have attempts remaining - will retry
                                    verify_logger.info(
                                        f"[PERIODIC_RETRY_DECISION] session={session_id} user_id={user_id} "
                                        f"decision=WILL_RETRY "
                                        f"reason=technical_error_and_attempts_remaining "
                                        f"current_attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                                        f"remaining_attempts={MAX_VERIFICATION_ATTEMPTS - verification_attempts} "
                                        f"faces_detected={faces_detected} "
                                        f"confidence={confidence} "
                                        f"next_retry_frame={frame_count + FRAMES_BETWEEN_ATTEMPTS}"
                                    )
                                    verify_logger.info(
                                        f"[PERIODIC_RETRY_SCHEDULED] session={session_id} user_id={user_id} "
                                        f"current_attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                                        f"remaining_attempts={MAX_VERIFICATION_ATTEMPTS - verification_attempts} "
                                        f"next_retry_in={FRAMES_BETWEEN_ATTEMPTS}_frames "
                                        f"reason=technical_error_will_retry_with_different_frame "
                                        f"faces_detected={faces_detected}"
                                    )
                        else:
                            # Still have attempts remaining - will retry on next eligible frame
                            # Retries are ONLY for technical errors (no face detected, embedding generation failed, etc.)
                            # NOT for face mismatch - face mismatch is sent immediately
                            verify_logger.info(
                                f"[PERIODIC_RETRY_SCHEDULED] session={session_id} user_id={user_id} "
                                f"current_attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                                f"remaining_attempts={MAX_VERIFICATION_ATTEMPTS - verification_attempts} "
                                f"next_retry_in={FRAMES_BETWEEN_ATTEMPTS}_frames "
                                f"reason=technical_error_will_retry_with_different_frame"
                            )
                        
                    except Exception as verification_exc:
                        error_logger.error(
                            f"[PERIODIC_VERIFICATION_ERROR] session={session_id} "
                            f"attempt={verification_attempts} error={str(verification_exc)}",
                            exc_info=True
                        )
                        verify_logger.error(
                            f"[PERIODIC_ERROR] session={session_id} user_id={user_id} "
                            f"attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS} "
                            f"error_type={type(verification_exc).__name__} "
                            f"error_message=\"{str(verification_exc)}\" "
                            f"frame_id={msg.frame_id}",
                            exc_info=True
                        )
                        
                        # On error, check if we should continue trying
                        if verification_attempts >= MAX_VERIFICATION_ATTEMPTS:
                            api_logger.error(
                                f"[PERIODIC_VERIFICATION_ABORTED] session={session_id} "
                                f"max_attempts_reached_with_errors"
                            )
                            verify_logger.error(
                                f"[PERIODIC_ABORTED] session={session_id} user_id={user_id} "
                                f"max_attempts_reached={MAX_VERIFICATION_ATTEMPTS} "
                                f"all_attempts_failed_with_errors"
                            )
                            verification_in_progress = False
                            last_verification_time = current_time
                            
                            # Sync state with engine for coordination
                            session.engine.periodic_verification_in_progress = False
                    else:
                        # Frame is being skipped - waiting for next eligible frame
                        frames_until_next = FRAMES_BETWEEN_ATTEMPTS - (frame_count % FRAMES_BETWEEN_ATTEMPTS)
                        verify_logger.debug(
                            f"[PERIODIC_FRAME_SKIP] session={session_id} user_id={user_id} "
                            f"frame_count={frame_count} "
                            f"verification_in_progress=True "
                            f"frame_divisible_by_{FRAMES_BETWEEN_ATTEMPTS}=False "
                            f"skipping_frame=YES "
                            f"frames_until_next_attempt={frames_until_next} "
                            f"current_attempt={verification_attempts}/{MAX_VERIFICATION_ATTEMPTS}"
                        )

            # ══════════════════════════════════════════════════════════════════════
            # ── PERIODIC BACKGROUND VALIDATION ────────────────────────────────
            # ══════════════════════════════════════════════════════════════════
            # Runs every 60 seconds (always)
            # Logic:
            # - If count < 3: Wait for 3 consecutive mismatches before warning
            # - If count >= 3: Send warning immediately on next mismatch
            # ══════════════════════════════════════════════════════════════════
            
            if hasattr(session, 'background_service') and session.background_service is not None:
                # Check if background service is registered
                if session.background_service.is_registered():
                    # Always use 60-second interval
                    bg_interval = session.background_service._settings.bg_check_interval_seconds
                    time_since_last_bg = current_time - last_background_validation_time
                    
                    # Check if we should validate background
                    if time_since_last_bg >= bg_interval and not background_validation_in_progress:
                        background_validation_in_progress = True
                        
                        bg_logger = get_logger("background_validation")
                        bg_logger.info(
                            f"[PERIODIC_BG_START] session={session_id} user_id={user_id} "
                            f"elapsed={time_since_last_bg:.1f}s interval={bg_interval}s "
                            f"frame_id={msg.frame_id}"
                        )
                        
                        try:
                            # Validate background from current frame
                            bg_validation_start = time.perf_counter()
                            bg_result = await session.background_service.validate_from_frame(
                                bgr_frame=bgr_frame,
                                user_id=user_id,
                                frame_id=msg.frame_id,
                                timestamp_ms=msg.timestamp_ms
                            )
                            bg_validation_latency_ms = round((time.perf_counter() - bg_validation_start) * 1000.0, 1)
                            
                            if bg_result:
                                bg_logger.info(
                                    f"[PERIODIC_BG_RESULT] session={session_id} user_id={user_id} "
                                    f"status={bg_result['status']} "
                                    f"is_match={bg_result['is_match']} "
                                    f"similarity={bg_result['similarity']:.4f} "
                                    f"consecutive_failures={bg_result['consecutive_failures']} "
                                    f"latency_ms={bg_validation_latency_ms}"
                                )
                                
                                # Check if background mismatch warning should be triggered
                                if bg_result['status'] == 'warning':
                                    # Add to active_violations
                                    session.active_violations['background_mismatch'] = (msg.frame_id, 0, False, False)
                                    
                                    bg_logger.warning(
                                        f"[PERIODIC_BG_WARNING] session={session_id} user_id={user_id} "
                                        f"background_changed similarity={bg_result['similarity']:.4f} "
                                        f"consecutive_failures={bg_result['consecutive_failures']} "
                                        f"message=\"{bg_result['message']}\""
                                    )
                                    
                                    api_logger.warning(
                                        f"[BACKGROUND_MISMATCH] session={session_id} user_id={user_id} "
                                        f"similarity={bg_result['similarity']:.4f} "
                                        f"consecutive_failures={bg_result['consecutive_failures']}"
                                    )
                                elif bg_result['status'] == 'matched':
                                    # Background matched - remove from active_violations if present
                                    if 'background_mismatch' in session.active_violations:
                                        bg_logger.info(
                                            f"[PERIODIC_BG_CLEARED] session={session_id} user_id={user_id} "
                                            f"background_matched similarity={bg_result['similarity']:.4f} "
                                            f"removing_from_violations"
                                        )
                                    
                                    api_logger.info(
                                        f"[BACKGROUND_MATCHED] session={session_id} user_id={user_id} "
                                        f"similarity={bg_result['similarity']:.4f}"
                                    )
                            
                            # Update last validation time
                            last_background_validation_time = current_time
                            background_validation_in_progress = False
                            
                        except Exception as bg_exc:
                            error_logger.error(
                                f"[PERIODIC_BG_ERROR] session={session_id} user_id={user_id} "
                                f"error={str(bg_exc)}",
                                exc_info=True
                            )
                            bg_logger.error(
                                f"[PERIODIC_BG_ERROR] session={session_id} user_id={user_id} "
                                f"error_type={type(bg_exc).__name__} "
                                f"error_message=\"{str(bg_exc)}\"",
                                exc_info=True
                            )
                            background_validation_in_progress = False
                            last_background_validation_time = current_time

            # ── Initial identity verification (first frame only) ───────────────
            if initial_verification_required and not initial_verification_done:
                initial_verification_in_progress = True  # Mark as in progress
                
                # Sync state with engine for coordination
                session.engine.initial_verification_in_progress = True
                session.engine.periodic_verification_in_progress = False
                
                api_logger.info(
                    f"[INITIAL_VERIFICATION_START] session={session_id} "
                    f"user_id={user_id} frame_id={msg.frame_id} "
                    f"in_progress=True"
                )
                verify_logger.info(
                    f"[INITIAL_VERIFICATION_STATE] session={session_id} user_id={user_id} "
                    f"in_progress=True "
                    f"will_block_periodic=True "
                    f"will_block_violation=True"
                )
                
                # ═══════════════════════════════════════════════════════════════
                # RUN IN PARALLEL: Face Verification + Background Registration
                # ═══════════════════════════════════════════════════════════════
                # STRATEGY:
                # 1. Run MODEL INFERENCE in parallel (face detection + background segmentation)
                # 2. Run DATABASE OPERATIONS sequentially (face embedding store, then background embedding store)
                # ═══════════════════════════════════════════════════════════════
                
                # Task 1: Trigger face verification (runs async in background)
                # This queues the face verification task
                # Pass is_initial_verification=True to skip quality checks
                session.engine.trigger_face_verification(
                    bgr_frame, 
                    msg.frame_id, 
                    msg.timestamp_ms,
                    is_initial_verification=True
                )
                
                # Task 2: Register background
                # Create background service ONCE and store it in session
                # The actual registration will happen AFTER face verification completes
                # to avoid concurrent database operations
                if user_id and db_session and not hasattr(session, 'background_service'):
                    try:
                        from proctoring_engine.services.background_service import BackgroundService  # noqa: PLC0415
                        from proctoring_engine.repositories.user_repository import UserRepository  # noqa: PLC0415
                        
                        # Create shared user repository (used by background service)
                        user_repo = UserRepository(db_session)
                        bg_service = BackgroundService(user_repository=user_repo)
                        
                        # Store background service in session
                        session.background_service = bg_service
                        session.last_background_validation_time = time.perf_counter()
                        
                        # Queue background registration (will run after current frame processing)
                        # This ensures database operations don't conflict
                        session.background_registration_pending = True
                        session.background_registration_frame = bgr_frame.copy()
                        session.background_registration_frame_id = msg.frame_id
                        session.background_registration_timestamp = msg.timestamp_ms
                        
                        api_logger.info(
                            f"[BACKGROUND_REGISTRATION_QUEUED] session={session_id} "
                            f"user_id={user_id} frame_id={msg.frame_id} "
                            f"will_register_after_face_verification"
                        )
                        
                    except Exception as bg_exc:
                        error_logger.error(
                            f"[BACKGROUND_SERVICE_INIT_ERROR] session={session_id} "
                            f"user_id={user_id} error={str(bg_exc)}",
                            exc_info=True
                        )
                        api_logger.warning(
                            f"[BACKGROUND_SERVICE_INIT_FAILED] session={session_id} "
                            f"user_id={user_id} - continuing without background validation"
                        )
                
                api_logger.info(
                    f"[INITIAL_VERIFICATION_TRIGGERED] session={session_id} "
                    f"user_id={user_id} frame_id={msg.frame_id} - verification queued"
                )

            # ── Inline audio (audio_b64 in the JSON) — still supported ─────────
            if msg.audio_b64:
                try:
                    pcm = _decode_pcm(msg.audio_b64)
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, session.engine.process_audio, pcm
                    )
                    audio_chunks += 1
                except Exception as exc:
                    api_logger.warning(
                        f"[AUDIO_ERR] session={session_id} "
                        f"frame_id={msg.frame_id} inline error={exc}"
                    )

            # ── Run inference (CPU-bound — offload to thread pool) ─────────────
            loop = asyncio.get_event_loop()
            infer_start = time.perf_counter()
            frame_result = await loop.run_in_executor(
                None,
                session.engine.process_frame,
                bgr_frame,
                msg.frame_id,
                msg.timestamp_ms,
            )
            infer_ms = round((time.perf_counter() - infer_start) * 1000.0, 1)
            
            # ── Continuous face verification (until match found) ───────────────
            # Check if initial verification completed successfully
            if initial_verification_required and not initial_verification_done:
                # Check if verification result is in frame_result
                if frame_result.identity_verification:
                    for verification_event in frame_result.identity_verification:
                        if verification_event.is_match:
                            # Face matched - initial verification complete!
                            initial_verification_done = True
                            initial_verification_in_progress = False
                            
                            # Sync state with engine for coordination
                            session.engine.initial_verification_in_progress = False
                            
                            api_logger.info(
                                f"[INITIAL_VERIFICATION_COMPLETE] session={session_id} "
                                f"user_id={user_id} frame_id={msg.frame_id} "
                                f"similarity={verification_event.similarity:.4f} "
                                f"in_progress=False success=True"
                            )
                            verify_logger.info(
                                f"[INITIAL_VERIFICATION_STATE] session={session_id} user_id={user_id} "
                                f"in_progress=False "
                                f"done=True "
                                f"will_block_periodic=False "
                                f"will_block_violation=False "
                                f"similarity={verification_event.similarity:.4f}"
                            )
                            
                            # ── Process queued background registration ─────────────────
                            # Now that face verification DB operation is complete,
                            # we can safely do background registration DB operation
                            if hasattr(session, 'background_registration_pending') and session.background_registration_pending:
                                try:
                                    bg_logger = get_logger("background_validation")
                                    bg_logger.info(
                                        f"[BACKGROUND_REGISTRATION_START] session={session_id} "
                                        f"user_id={user_id} frame_id={session.background_registration_frame_id} "
                                        f"processing_queued_registration"
                                    )
                                    
                                    # Register background from queued frame
                                    bg_result = await session.background_service.register_background_from_frame(
                                        bgr_frame=session.background_registration_frame,
                                        user_id=user_id,
                                        frame_id=session.background_registration_frame_id,
                                        timestamp_ms=session.background_registration_timestamp
                                    )
                                    
                                    api_logger.info(
                                        f"[BACKGROUND_REGISTERED] session={session_id} "
                                        f"user_id={user_id} frame_id={session.background_registration_frame_id} "
                                        f"status={bg_result.status} "
                                        f"similarity={bg_result.similarity_score:.4f}"
                                    )
                                    
                                    # Clear pending flag and frame data
                                    session.background_registration_pending = False
                                    session.background_registration_frame = None
                                    session.background_registration_frame_id = None
                                    session.background_registration_timestamp = None
                                    
                                except Exception as bg_exc:
                                    error_logger.error(
                                        f"[BACKGROUND_REGISTRATION_ERROR] session={session_id} "
                                        f"user_id={user_id} error={str(bg_exc)}",
                                        exc_info=True
                                    )
                                    api_logger.warning(
                                        f"[BACKGROUND_REGISTRATION_FAILED] session={session_id} "
                                        f"user_id={user_id} - continuing without background validation"
                                    )
                            
                            break

            # ── Accumulate into window ─────────────────────────────────────────
            session.add_frame(frame_result)

            # ── Send response every WINDOW_SIZE_FRAMES frames ──────────────────
            if session.is_window_ready():
                response = session.build_response()
                window_count += 1
                
                # ── Trigger face verification if needed ────────────────────────
                if response.trigger_face_verification:
                    api_logger.info(
                        f"[VERIFICATION_CHECK] session={session_id} "
                        f"trigger={response.trigger_face_verification} "
                        f"user_id={user_id} has_db={db_session is not None}"
                    )
                    
                    if user_id and db_session:
                        api_logger.info(
                            f"[FACE_VERIFICATION_START] session={session_id} "
                            f"user_id={user_id} window={window_count}"
                        )
                        try:
                            # Run face verification asynchronously
                            from proctoring_engine.services.face_verification_service import FaceVerificationService  # noqa: PLC0415
                            from proctoring_engine.repositories.user_repository import UserRepository  # noqa: PLC0415
                            from proctoring_engine.services.face_service import FaceService  # noqa: PLC0415
                            
                            user_repo = UserRepository(db_session)
                            face_service = FaceService()
                            verification_service = FaceVerificationService(user_repo, face_service)
                            
                            # Use the current frame for verification
                            verification_result = await verification_service.verify_from_frame(
                                bgr_frame,
                                user_id
                            )
                            
                            api_logger.info(
                                f"[FACE_VERIFICATION_RESULT] session={session_id} "
                                f"user_id={user_id} is_match={verification_result['is_match']} "
                                f"similarity={verification_result['similarity']:.4f} "
                                f"confidence={verification_result['confidence']} "
                                f"faces_detected={verification_result.get('faces_detected', 'N/A')}"
                            )
                        except Exception as verification_exc:
                            error_logger.error(
                                f"[FACE_VERIFICATION_ERROR] session={session_id} "
                                f"user_id={user_id} window={window_count} error={str(verification_exc)}",
                                exc_info=True
                            )
                            api_logger.error(
                                f"[FACE_VERIFICATION_ERROR] session={session_id} "
                                f"user_id={user_id} error={str(verification_exc)}"
                            )
                    else:
                        reason = 'no_user_id' if not user_id else 'no_db_session'
                        api_logger.warning(
                            f"[VERIFICATION_SKIPPED] session={session_id} reason={reason}"
                        )
                        error_logger.warning(
                            f"[VERIFICATION_SKIPPED] session={session_id} "
                            f"user_id={user_id} has_db={db_session is not None} reason={reason}"
                        )
                
                # Remove internal flag before sending to frontend
                response_dict = response.model_dump()
                response_dict.pop('trigger_face_verification', None)
                
                await websocket.send_json(response_dict)

                # ── Log the exact JSON response sent to the backend ────────────
                api_logger.info(
                    f"[RESPONSE] session={session_id} "
                    f"window={window_count} "
                    f"latency={response.total_latency_ms:.0f}ms "
                    f"payload={response.model_dump_json()}"
                )

    except WebSocketDisconnect:
        pass   # normal disconnect — logged below
    except Exception as exc:
        error_logger.error(
            f"[WEBSOCKET_ERROR] session={session_id} "
            f"frames_processed={frame_count} error={str(exc)}",
            exc_info=True,
        )
        api_logger.error(
            f"[ERROR] session={session_id} "
            f"frames_processed={frame_count} error={exc}"
        )
        try:
            await websocket.send_text(
                ErrorResponse(
                    session_id=session_id,
                    error="internal_error",
                    detail=str(exc),
                ).model_dump_json()
            )
        except Exception:
            pass
    finally:
        duration_s = round(time.perf_counter() - session_start, 1)
        await manager.close_session(session_id)
        
        # Close database session if it was created
        if db_session:
            try:
                await db_session.close()
                api_logger.info(f"[DB_SESSION_CLOSED] session={session_id}")
            except Exception as exc:
                error_logger.error(
                    f"[DB_SESSION_CLOSE_ERROR] session={session_id} error={str(exc)}",
                    exc_info=True
                )
                api_logger.warning(f"[DB_SESSION_CLOSE_FAILED] session={session_id}")
        
        api_logger.info(
            f"[DISCONNECT] session={session_id} "
            f"duration={duration_s}s "
            f"frames={frame_count} "
            f"audio_chunks={audio_chunks} "
            f"windows_sent={window_count} "
            f"active_sessions={manager.active_count}"
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _decode_jpeg(b64: str) -> np.ndarray:
    """Decode a base64-encoded JPEG into a BGR numpy array.

    The frame is horizontally flipped here (same as the camera mode's
    cv2.flip(frame, 1)) so all downstream pipelines receive a mirrored image.
    """
    raw_bytes = base64.b64decode(b64)
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None — invalid JPEG data")
    # Mirror so left/right match the candidate's actual movement
    return cv2.flip(bgr, 1)


def _decode_pcm(b64: str) -> np.ndarray:
    """Decode a base64-encoded float32 PCM chunk into a numpy array.

    Expected: little-endian IEEE 754 float32, 16 kHz, mono.
    4096 samples = 16384 bytes base64-encoded.
    """
    raw_bytes = base64.b64decode(b64)
    return np.frombuffer(raw_bytes, dtype=np.float32).copy()
