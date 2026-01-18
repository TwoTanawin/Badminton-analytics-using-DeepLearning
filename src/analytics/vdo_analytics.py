import cv2
from ultralytics import YOLO
import asyncio

class VDOAnalytics:
    def __init__(self):
        model_path = '/Volumes/PortableSSD/Two/Badminton-analytics/analytics/model/best_bad_stc.pt'  # Path to the pre-trained YOLO model
        self.video_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/videos/IMG_0211.MOV"
        self.model = YOLO(model_path)

    async def analyze_frame(self, frame):
        # Perform object detection on the frame (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self.model, frame)
        return results

    async def visualize_detections(self, frame, results):
        # Draw bounding boxes and labels on the frame
        loop = asyncio.get_event_loop()
        annotated_frame = await loop.run_in_executor(None, lambda: results[0].plot())
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