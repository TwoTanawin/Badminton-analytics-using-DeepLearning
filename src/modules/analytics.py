from ultralytics import YOLO
import asyncio
import torch


class YoloAnalytics:
    """
    Core YOLO analytics class for object detection and tracking.
    Handles model initialization, device detection, and frame analysis.
    """
    def __init__(self, model_path, tracker="bytetrack.yaml"):
        """
        Initialize YOLO analytics with model and device detection.
        
        Args:
            model_path: Path to the pre-trained YOLO model file
            tracker: Tracker configuration (default: "bytetrack.yaml")
        """
        # Detect and set device for Mac GPU (MPS) or fallback to CPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using Mac GPU (MPS) for acceleration")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("Using CUDA GPU for acceleration")
        else:
            self.device = 'cpu'
            print("Using CPU (no GPU acceleration available)")
        
        # Initialize YOLO model with the detected device
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # ByteTrack tracker configuration
        # Using Ultralytics built-in ByteTrack tracker
        self.tracker = tracker
        
    def analyze_frame(self, frame):
        """
        Perform object detection on a single frame (synchronous).
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            YOLO detection results
        """
        results = self.model(frame)
        return results
    
    async def analyze_frame_async(self, frame):
        """
        Perform object detection and tracking using ByteTrack (asynchronous).
        Run in thread pool to avoid blocking.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            YOLO tracking results with track IDs
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            lambda: self.model.track(frame, device=self.device, tracker=self.tracker, persist=True)
        )
        return results