"""Face detection using MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np

from proctoring_engine.settings import get_settings
from proctoring_engine.utils.exceptions import ValidationException
from proctoring_engine.utils.exceptions.error_codes import (
    FACE_NOT_DETECTED,
    FACE_TOO_SMALL,
    MULTIPLE_FACES_DETECTED,
)
from proctoring_engine.utils.logger import get_logger

logger = get_logger("face_detection")

# Get settings instance
_settings = get_settings()# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


def detect_face_with_mediapipe(image: np.ndarray) -> dict:
    """Detect face using MediaPipe Face Detection.

    Args:
        image: Input image (BGR format)

    Returns:
        dict with face detection results

    Raises:
        ValidationException: If face detection fails
    """
    logger.info("Starting MediaPipe face detection")

    height, width = image.shape[:2]
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MediaPipe Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=_settings.detection_confidence
    ) as face_detection:
        results = face_detection.process(rgb_image)
        
        if not results.detections:
            logger.warning("No face detected in image")
            raise ValidationException(
                message="No face detected in the image. Please ensure your face is clearly visible.",
                error_code=FACE_NOT_DETECTED,
                details=None,
            )

        if len(results.detections) > _settings.max_faces_allowed:
            logger.warning(f"Multiple faces detected: {len(results.detections)}")
            raise ValidationException(
                message="Multiple faces detected. Please ensure only one person is in the frame.",
                error_code=MULTIPLE_FACES_DETECTED,
                details={"faces_detected": len(results.detections)},
            )

        # Get bounding box from first detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Convert to absolute coordinates
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)

        # Validate face size
        face_area = w * h
        image_area = width * height
        face_ratio = face_area / image_area

        logger.info(f"Face detected: bbox=({x},{y},{w},{h}), ratio={face_ratio:.3f}")

        if w < _settings.min_face_size or h < _settings.min_face_size:
            logger.warning(f"Face too small: {w}x{h}")
            raise ValidationException(
                message="Face is too small. Please move closer to the camera.",
                error_code=FACE_TOO_SMALL,
                details={"face_size": f"{w}x{h}", "min_required": f"{_settings.min_face_size}x{_settings.min_face_size}"},
            )

        if face_ratio < _settings.min_face_ratio:
            logger.warning(f"Face too far: ratio={face_ratio:.3f}")
            raise ValidationException(
                message="Face is too far from camera. Please move closer.",
                error_code=FACE_TOO_SMALL,
                details={"face_ratio": float(face_ratio), "min_required": _settings.min_face_ratio},
            )

        # Get confidence score
        confidence = detection.score[0] if detection.score else 0.0

        return {
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "confidence": float(confidence),
            "face_ratio": float(face_ratio),
        }


def get_face_landmarks(image: np.ndarray) -> np.ndarray:
    """Get facial landmarks using MediaPipe Face Mesh.

    Args:
        image: Input image (BGR format)

    Returns:
        numpy array of facial landmarks
    """
    logger.info("Starting MediaPipe Face Mesh landmark extraction")
    
    height, width = image.shape[:2]
    logger.info(f"Image dimensions: {width}x{height}")

    # Convert BGR to RGB
    logger.info("Converting image from BGR to RGB for MediaPipe Face Mesh")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MediaPipe Face Mesh
    logger.info("Initializing MediaPipe Face Mesh (468 landmarks, refined mode)")
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        logger.info("Processing image with Face Mesh model")
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            logger.error("No facial landmarks detected by MediaPipe Face Mesh")
            return None

        logger.info(f"Face Mesh detected {len(results.multi_face_landmarks)} face(s)")
        landmarks = results.multi_face_landmarks[0]

        # Convert to pixel coordinates
        logger.info("Converting relative landmarks to pixel coordinates")
        points = []
        for idx, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append([x, y])

        landmark_array = np.array(points, dtype=np.int32)
        logger.info(f"Facial landmarks extracted successfully: {len(landmark_array)} points")
        logger.info(f"Landmark bounds: x=[{landmark_array[:, 0].min()}, {landmark_array[:, 0].max()}], y=[{landmark_array[:, 1].min()}, {landmark_array[:, 1].max()}]")
        
        return landmark_array


def create_convex_hull_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Create convex hull mask from facial landmarks.

    Args:
        image: Input image
        landmarks: Facial landmarks

    Returns:
        Binary mask of face region
    """
    logger.info("Starting convex hull mask creation")
    
    height, width = image.shape[:2]
    logger.info(f"Creating mask for image size: {width}x{height}")
    logger.info(f"Using {len(landmarks)} landmark points for convex hull")

    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    logger.info("Empty mask initialized")

    # Create convex hull from landmarks
    logger.info("Computing convex hull from facial landmarks")
    hull = cv2.convexHull(landmarks)
    logger.info(f"Convex hull computed with {len(hull)} points")

    # Fill convex hull on mask
    logger.info("Filling convex hull polygon on mask")
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Calculate mask statistics
    mask_area = np.count_nonzero(mask)
    total_area = width * height
    mask_ratio = mask_area / total_area
    logger.info(f"Convex hull mask created: area={mask_area} pixels ({mask_ratio:.2%} of image)")

    return mask



def detect_all_faces_for_violation(image: np.ndarray) -> list[dict]:
    """Detect ALL faces in image for violation verification (no exceptions).
    
    This function is specifically for violation scenarios where we want to:
    - Detect all faces (no limit)
    - No validation errors for multiple faces or small faces
    - Return all detected faces for individual verification
    
    Args:
        image: Input image (RGB format)
        
    Returns:
        List of face detection results, each containing:
        - bbox: dict with x, y, width, height in absolute coordinates
        - confidence: detection confidence score
        - face_index: index of the face
        Empty list if no faces detected
    """
    logger.info("Starting MediaPipe face detection for violation verification (all faces)")
    
    height, width = image.shape[:2]
    
    # Use MediaPipe Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=_settings.detection_confidence
    ) as face_detection:
        results = face_detection.process(image)
        
        if not results.detections:
            logger.warning("No faces detected in violation verification")
            return []
        
        faces = []
        for idx, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to absolute coordinates
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # Ensure bbox coordinates are within image bounds
            # Don't clip width/height - just ensure x,y are valid
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            # Ensure bbox doesn't exceed image bounds
            if x + w > width:
                w = width - x
            if y + h > height:
                h = height - y
            
            confidence = detection.score[0] if detection.score else 0.0
            
            faces.append({
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "confidence": float(confidence),
                "face_index": idx,
            })
            
            logger.info(
                f"Face {idx} detected: bbox=({x},{y},{w},{h}), "
                f"confidence={confidence:.3f}"
            )
        
        logger.info(f"Total faces detected for violation verification: {len(faces)}")
        return faces
