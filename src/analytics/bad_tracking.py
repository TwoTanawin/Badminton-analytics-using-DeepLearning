import cv2
from ultralytics import YOLO
import asyncio
import torch

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

    async def analyze_frame(self, frame):
        # Perform object detection and tracking on the frame (run in thread pool to avoid blocking)
        # Using track() method enables tracking with persistent IDs across frames
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self.model.track(frame, persist=True))
        return results

    async def visualize_detections(self, frame, results):
        # Draw bounding boxes, labels, and tracking IDs on the frame
        loop = asyncio.get_event_loop()
        annotated_frame = await loop.run_in_executor(None, lambda: results[0].plot())
        
        # Add explicit tracking ID labels if tracking is available
        # The plot() method may already show IDs, but we ensure they're clearly visible
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.cpu().numpy()
            box_coords = boxes.xyxy.cpu().numpy()
            
            for i in range(len(boxes)):
                track_id = int(track_ids[i])
                x1, y1, x2, y2 = map(int, box_coords[i][:4])
                # Draw tracking ID above the bounding box
                cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return annotated_frame

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
                
                results = await self.analyze_frame(frame)
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