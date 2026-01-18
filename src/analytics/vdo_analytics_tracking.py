import cv2
from ultralytics import YOLO
import asyncio
import torch
import numpy as np

class VDOAnalytics:
    def __init__(self):
        model_path = '/Volumes/PortableSSD/Two/Badminton-analytics/analytics/model/best_bad_stc.pt'  # Path to the pre-trained YOLO model
        self.video_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/videos/IMG_0211.MOV"
        
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
        self.tracker = "bytetrack.yaml"

    async def analyze_frame(self, frame):
        # Perform object detection and tracking using ByteTrack
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            lambda: self.model.track(frame, device=self.device, tracker=self.tracker, persist=True)
        )
        return results

    async def visualize_detections(self, frame, results):
        # Draw bounding boxes, labels, and tracking IDs from ByteTrack
        annotated_frame = frame.copy()
        
        # Get class names from model
        class_names = self.model.names
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            box_coords = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, box_coords[i][:4])
                conf = float(confidences[i])
                cls = int(classes[i])
                class_name = class_names.get(cls, f"class_{cls}")
                
                # Get track ID if available
                track_id = int(track_ids[i]) if track_ids is not None else None
                
                # Draw bounding box with color based on track ID
                if track_id is not None:
                    color = self._get_color_for_track_id(track_id)
                    label = f"ID: {track_id} | {class_name} | {conf:.2f}"
                else:
                    color = (0, 255, 0)  # Green for untracked
                    label = f"{class_name} | {conf:.2f}"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with tracking ID, class name, and confidence
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(annotated_frame, (x1, label_y - label_size[1] - 10), 
                             (x1 + label_size[0], label_y), color, -1)
                cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _get_color_for_track_id(self, track_id):
        """Generate a consistent color for each track ID"""
        # Convert track_id to integer if it's a string, or use hash for consistent color
        if isinstance(track_id, str):
            # Use hash to convert string to integer seed
            seed = hash(track_id) % (2**31)  # Ensure it's within int32 range
        else:
            seed = int(track_id)
        
        np.random.seed(seed)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color

    async def display_frame(self, frame, delay=1):
        # Display the annotated frame (must run on main thread - OpenCV GUI requirement)
        cv2.imshow('Annotated Frame', frame)
        # Wait for key press (non-blocking check) or 'q' to quit
        key = cv2.waitKey(delay) & 0xFF
        # Yield control to event loop
        await asyncio.sleep(0)
        return key == ord('q')
        
    async def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {self.video_path}")
        
        # Get video FPS for proper playback speed
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get detections and tracking from YOLO with ByteTrack
                results = await self.analyze_frame(frame)
                
                # Visualize tracked objects
                annotated_frame = await self.visualize_detections(frame, results)
                
                # Break if 'q' is pressed
                if await self.display_frame(annotated_frame):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
async def main():
    vdo_analytics = VDOAnalytics()
    await vdo_analytics.process_video()
        
if __name__ == "__main__":
    asyncio.run(main())