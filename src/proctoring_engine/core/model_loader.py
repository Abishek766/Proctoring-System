"""Model loader - centralized model management for all ML models.

This module provides a single source of truth for all model instances used
across the application. Models are loaded once and reused across all requests.

Models managed:
- InsightFace (ArcFace) - Face recognition and embedding generation
- MediaPipe FaceMesh - Face landmark detection
- MediaPipe Hands - Hand detection and tracking
- YOLOv8 - Object detection
- YAMNet - Audio classification
"""

from pathlib import Path

from insightface.app import FaceAnalysis

from proctoring_engine.settings import get_settings
from proctoring_engine.utils.logger import get_logger

logger = get_logger("model_loader")

# Get settings instance
_settings = get_settings()

# Model directory
MODEL_DIR = _settings.model_dir
MODEL_DIR.mkdir(exist_ok=True)

# Global model instances (shared across all requests)
_global_face_app = None
_global_face_mesh = None
_global_hands = None
_global_yolo = None
_global_yamnet = None
_global_selfie_segmentation = None
_global_dinov2 = None


# ═══════════════════════════════════════════════════════════════════════════
# InsightFace (ArcFace) - Face Recognition
# ═══════════════════════════════════════════════════════════════════════════

def preload_insightface_model():
    """Preload InsightFace model at startup with ONNX Runtime optimizations.
    
    This function should be called during app startup to avoid
    loading the model during the first API request.
    
    Returns:
        FaceAnalysis: Initialized InsightFace model
    """
    global _global_face_app
    
    if _global_face_app is not None:
        logger.info("InsightFace model already loaded")
        return _global_face_app
    
    try:
        logger.info("Preloading InsightFace model (buffalo_l) with ONNX Runtime optimizations...")

        # Configure ONNX Runtime execution providers with optimizations
        providers = [
            ('CPUExecutionProvider', {
                'arena_extend_strategy': 'kSameAsRequested',
                'intra_op_num_threads': 4,
                'inter_op_num_threads': 2,
            })
        ]

        # This will auto-download buffalo_l model if not present
        _global_face_app = FaceAnalysis(
            name="buffalo_l", 
            root=str(MODEL_DIR), 
            providers=providers
        )
        _global_face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        logger.info("InsightFace model preloaded successfully with optimizations")
        logger.info("  - Intra-op threads: 4")
        logger.info("  - Inter-op threads: 2")
        return _global_face_app

    except Exception as e:
        logger.error(f"Failed to preload InsightFace model: {e}")
        raise


def get_insightface_model():
    """Get the global InsightFace model instance.
    
    If the model hasn't been preloaded, it will be loaded on first call.
    
    Returns:
        FaceAnalysis: Initialized InsightFace model
    """
    global _global_face_app
    
    if _global_face_app is None:
        logger.warning("InsightFace model not preloaded - loading now (this may take 5-10 seconds)")
        return preload_insightface_model()
    
    return _global_face_app


# ═══════════════════════════════════════════════════════════════════════════
# MediaPipe FaceMesh - Face Landmark Detection
# ═══════════════════════════════════════════════════════════════════════════

def preload_facemesh_model():
    """Preload MediaPipe FaceMesh model at startup.
    
    Returns:
        FaceMesh: Initialized MediaPipe FaceMesh model
    """
    global _global_face_mesh
    
    if _global_face_mesh is not None:
        logger.info("MediaPipe FaceMesh model already loaded")
        return _global_face_mesh
    
    try:
        import mediapipe as mp  # noqa: PLC0415
        
        logger.info("Preloading MediaPipe FaceMesh model...")
        
        _global_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        logger.info("MediaPipe FaceMesh model preloaded successfully")
        return _global_face_mesh
        
    except Exception as e:
        logger.error(f"Failed to preload MediaPipe FaceMesh model: {e}")
        raise


def get_facemesh_model():
    """Get the global MediaPipe FaceMesh model instance.
    
    Returns:
        FaceMesh: Initialized MediaPipe FaceMesh model
    """
    global _global_face_mesh
    
    if _global_face_mesh is None:
        logger.warning("MediaPipe FaceMesh model not preloaded - loading now")
        return preload_facemesh_model()
    
    return _global_face_mesh


# ═══════════════════════════════════════════════════════════════════════════
# MediaPipe Hands - Hand Detection
# ═══════════════════════════════════════════════════════════════════════════

def preload_hands_model():
    """Preload MediaPipe Hands model at startup.
    
    Returns:
        Hands: Initialized MediaPipe Hands model
    """
    global _global_hands
    
    if _global_hands is not None:
        logger.info("MediaPipe Hands model already loaded")
        return _global_hands
    
    try:
        import mediapipe as mp  # noqa: PLC0415
        
        logger.info("Preloading MediaPipe Hands model...")
        
        _global_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        logger.info("MediaPipe Hands model preloaded successfully")
        return _global_hands
        
    except Exception as e:
        logger.error(f"Failed to preload MediaPipe Hands model: {e}")
        raise


def get_hands_model():
    """Get the global MediaPipe Hands model instance.
    
    Returns:
        Hands: Initialized MediaPipe Hands model
    """
    global _global_hands
    
    if _global_hands is None:
        logger.warning("MediaPipe Hands model not preloaded - loading now")
        return preload_hands_model()
    
    return _global_hands


# ═══════════════════════════════════════════════════════════════════════════
# YOLOv8 - Object Detection
# ═══════════════════════════════════════════════════════════════════════════

def preload_yolo_model(model_path: str):
    """Preload YOLOv8 model at startup.
    
    Args:
        model_path: Path to YOLO model file
    
    Returns:
        YOLO: Initialized YOLO model
    """
    global _global_yolo
    
    if _global_yolo is not None:
        logger.info("YOLO model already loaded")
        return _global_yolo
    
    try:
        from ultralytics import YOLO  # noqa: PLC0415
        
        logger.info(f"Preloading YOLO model from {model_path}...")
        
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            model_name = model_path_obj.stem
            pt_name = f"{model_name}.pt"
            logger.info(f"Model '{model_path}' not found. Downloading '{pt_name}' from Ultralytics hub...")
            import shutil  # noqa: PLC0415
            YOLO(pt_name)  # triggers auto-download to cwd
            if Path(pt_name).exists():
                shutil.move(pt_name, str(model_path_obj))
                logger.info(f"Downloaded to {model_path_obj}")
        
        _global_yolo = YOLO(str(model_path_obj))
        
        logger.info(f"YOLO model preloaded successfully from {model_path}")
        return _global_yolo
        
    except Exception as e:
        logger.error(f"Failed to preload YOLO model: {e}")
        raise


def get_yolo_model():
    """Get the global YOLO model instance.
    
    Returns:
        YOLO: Initialized YOLO model
    """
    global _global_yolo
    
    if _global_yolo is None:
        logger.warning("YOLO model not preloaded - this should not happen")
        raise RuntimeError("YOLO model not preloaded. Call preload_yolo_model() first.")
    
    return _global_yolo


# ═══════════════════════════════════════════════════════════════════════════
# YAMNet - Audio Classification
# ═══════════════════════════════════════════════════════════════════════════

def preload_yamnet_model(model_url: str):
    """Preload YAMNet model at startup.
    
    Args:
        model_url: TensorFlow Hub URL for YAMNet model
    
    Returns:
        YAMNet model from TensorFlow Hub
    """
    global _global_yamnet
    
    if _global_yamnet is not None:
        logger.info("YAMNet model already loaded")
        return _global_yamnet
    
    try:
        import tensorflow_hub as hub  # noqa: PLC0415
        
        logger.info(f"Preloading YAMNet model from {model_url}...")
        
        _global_yamnet = hub.load(model_url)
        
        logger.info("YAMNet model preloaded successfully")
        return _global_yamnet
        
    except Exception as e:
        logger.error(f"Failed to preload YAMNet model: {e}")
        raise


def get_yamnet_model():
    """Get the global YAMNet model instance.
    
    Returns:
        YAMNet model from TensorFlow Hub
    """
    global _global_yamnet
    
    if _global_yamnet is None:
        logger.warning("YAMNet model not preloaded - this should not happen")
        raise RuntimeError("YAMNet model not preloaded. Call preload_yamnet_model() first.")
    
    return _global_yamnet


# ═══════════════════════════════════════════════════════════════════════════
# MediaPipe Selfie Segmentation - Person Segmentation for Background Validation
# ═══════════════════════════════════════════════════════════════════════════

def preload_selfie_segmentation_model():
    """Preload MediaPipe Selfie Segmentation model at startup.
    
    Returns:
        SelfieSegmentation: Initialized MediaPipe Selfie Segmentation model
    """
    global _global_selfie_segmentation
    
    if _global_selfie_segmentation is not None:
        logger.info("MediaPipe Selfie Segmentation model already loaded")
        return _global_selfie_segmentation
    
    try:
        import mediapipe as mp  # noqa: PLC0415
        
        logger.info("Preloading MediaPipe Selfie Segmentation model...")
        
        # Model selection: 0 = general model (portrait), 1 = landscape model
        # Using model 1 for better performance with exam environment
        _global_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        logger.info("MediaPipe Selfie Segmentation model preloaded successfully")
        return _global_selfie_segmentation
        
    except Exception as e:
        logger.error(f"Failed to preload MediaPipe Selfie Segmentation model: {e}")
        raise


def get_selfie_segmentation_model():
    """Get the global MediaPipe Selfie Segmentation model instance.
    
    Returns:
        SelfieSegmentation: Initialized MediaPipe Selfie Segmentation model
    """
    global _global_selfie_segmentation
    
    if _global_selfie_segmentation is None:
        logger.warning("MediaPipe Selfie Segmentation model not preloaded - loading now")
        return preload_selfie_segmentation_model()
    
    return _global_selfie_segmentation


# ═══════════════════════════════════════════════════════════════════════════
# DINOv2 - Vision Transformer for Background Embeddings
# ═══════════════════════════════════════════════════════════════════════════

def preload_dinov2_model():
    """Preload DINOv2 Vision Transformer model at startup using ONNX Runtime.
    
    Uses ONNX Runtime for optimized CPU inference (3-5x faster than PyTorch).
    Downloads the pre-converted ONNX model from Hugging Face.
    
    Returns:
        ONNX Runtime InferenceSession for DINOv2
    """
    global _global_dinov2
    
    if _global_dinov2 is not None:
        logger.info("DINOv2 ONNX model already loaded")
        return _global_dinov2
    
    try:
        import onnxruntime as ort  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        
        logger.info("Loading DINOv2 model with ONNX Runtime for optimized CPU inference...")
        
        # Model path in models directory
        model_dir = MODEL_DIR / "dinov2"
        model_path = model_dir / "dinov2_vitb14.onnx"
        
        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download ONNX model if not exists
        if not model_path.exists():
            logger.info("Downloading DINOv2 ONNX model from Hugging Face...")
            logger.info("This is a one-time download (~330MB)...")
            
            # Download from Hugging Face deepghs/dinov2_onnx repository
            # URL format: https://huggingface.co/{repo}/resolve/main/{path}/{filename}
            url = "https://huggingface.co/deepghs/dinov2_onnx/resolve/main/facebook/dinov2-base/model.onnx"
            
            try:
                urllib.request.urlretrieve(url, str(model_path))
                logger.info(f"Downloaded DINOv2 ONNX model to {model_path}")
            except Exception as download_error:
                logger.error(f"Failed to download ONNX model from {url}: {download_error}")
                raise RuntimeError(
                    f"Could not download DINOv2 ONNX model. Please check your internet connection "
                    f"or manually download from {url} to {model_path}"
                )
        
        # Configure ONNX Runtime session options for optimal CPU performance
        session_options = ort.SessionOptions()
        
        # Enable all graph optimizations
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set thread count for parallel execution
        session_options.intra_op_num_threads = 4  # Parallel ops within a node
        session_options.inter_op_num_threads = 2  # Parallel execution of nodes
        
        # Enable memory pattern optimization
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        # Set execution mode to sequential (better for CPU)
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Load ONNX model with optimized settings
        _global_dinov2 = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        
        logger.info(f"DINOv2 ONNX model loaded successfully with optimizations")
        logger.info(f"  - Graph optimization: ENABLED")
        logger.info(f"  - Intra-op threads: 4")
        logger.info(f"  - Inter-op threads: 2")
        logger.info(f"  - Memory optimization: ENABLED")
        
        return _global_dinov2
        
    except Exception as e:
        logger.error(f"Failed to load DINOv2 ONNX model: {e}")
        raise


def get_dinov2_model():
    """Get the global DINOv2 model instance.
    
    Returns:
        DINOv2 model from torch.hub
    """
    global _global_dinov2
    
    if _global_dinov2 is None:
        logger.warning("DINOv2 model not preloaded - loading now (this may take 10-30 seconds)")
        return preload_dinov2_model()
    
    return _global_dinov2


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup Functions
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_models():
    """Clean up all models on app shutdown.
    
    This properly closes MediaPipe models that were kept alive during
    the application lifetime for reuse across sessions.
    """
    global _global_face_mesh, _global_hands, _global_selfie_segmentation
    
    logger.info("Cleaning up shared models...")
    
    # Close MediaPipe models
    if _global_face_mesh is not None:
        try:
            _global_face_mesh.close()
            logger.info("MediaPipe FaceMesh closed")
        except Exception as e:
            logger.warning(f"Error closing FaceMesh: {e}")
        _global_face_mesh = None
    
    if _global_hands is not None:
        try:
            _global_hands.close()
            logger.info("MediaPipe Hands closed")
        except Exception as e:
            logger.warning(f"Error closing Hands: {e}")
        _global_hands = None
    
    if _global_selfie_segmentation is not None:
        try:
            _global_selfie_segmentation.close()
            logger.info("MediaPipe Selfie Segmentation closed")
        except Exception as e:
            logger.warning(f"Error closing Selfie Segmentation: {e}")
        _global_selfie_segmentation = None
    
    # Other models (InsightFace, YOLO, YAMNet, DINOv2) don't need explicit cleanup
    logger.info("Model cleanup complete")


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Functions (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

def download_models():
    """Download InsightFace models on startup.
    
    Deprecated: Use preload_insightface_model() instead.
    """
    return preload_insightface_model()
