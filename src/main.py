import cv2
import asyncio
import numpy as np
from .modules.draw_roi import ROITool
from .modules.analytics import YoloAnalytics
from .modules.score_system import ScoreSystem

class VDOAnalytics:
    def __init__(self):
        model_path = '/Volumes/PortableSSD/Two/Badminton-analytics/analytics/model/best_bad_stc.pt'  # Path to the pre-trained YOLO model
        self.video_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/videos/IMG_0211.MOV"
        
        # Initialize core YOLO analytics (handles device detection and model setup)
        self.analytics = YoloAnalytics(model_path, tracker="bytetrack.yaml")
        
        # Access model and device from analytics instance if needed
        self.model = self.analytics.model
        self.device = self.analytics.device
        
        roi_config_half_0_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_0.yaml"
        roi_config_half_1_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_1.yaml"

        self.roi_config_half_0 = ROITool.load_roi_config(roi_config_half_0_path)
        self.roi_config_half_1 = ROITool.load_roi_config(roi_config_half_1_path)
        
        # Store all ROIs in a list for easy iteration
        self.rois = [self.roi_config_half_0, self.roi_config_half_1]
        
        # Initialize score system with ROIs
        self.score_system = ScoreSystem(self.rois, frame_threshold=10)

    
    def draw_filtered_detections(self, frame, results, filtered_indices):
        """Draw only the filtered detections on the frame with tracking IDs"""
        if len(results) == 0 or results[0].boxes is None or len(filtered_indices) == 0:
            # Still draw scores even if no detections
            self.score_system.draw_scores(frame)
            return frame
        
        boxes = results[0].boxes
        
        # Get class names from model
        class_names = self.model.names
        
        # Get track IDs if available (from tracking)
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
        
        for idx in filtered_indices:
            # Get box coordinates
            box = boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Get confidence and class
            conf = float(boxes.conf[idx].cpu().numpy()) if boxes.conf is not None else 1.0
            cls_id = int(boxes.cls[idx].cpu().numpy()) if boxes.cls is not None else 0
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            
            # Get track ID if available
            track_id = int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else None
            
            # Get frame count for this track_id in its ROI using score system
            frame_count_text = ""
            if track_id is not None:
                roi_index = self.score_system.get_roi_index_for_detection(box)
                if roi_index is not None:
                    frame_count = self.score_system.get_frame_count(track_id, roi_index)
                    if frame_count > 0:
                        frame_count_text = f" | Frames: {frame_count}"
            
            # Draw bounding box with color based on track ID
            if track_id is not None:
                color = self._get_color_for_track_id(track_id)
                label = f"ID: {track_id} | {class_name} | {conf:.2f}{frame_count_text}"
            else:
                color = (0, 255, 0)  # Green for untracked
                label = f"{class_name} | {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID, class name, and confidence
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, label_size[1] + 10)
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), 
                         (x1 + label_size[0], label_y), color, -1)
            cv2.putText(frame, label, (x1, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw scores on frame using score system
        self.score_system.draw_scores(frame)
        
        return frame
    
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

    async def analyze_frame(self, frame):
        # Use the analytics class for frame analysis
        results = await self.analytics.analyze_frame_async(frame)
        return results

    async def visualize_detections(self, frame, filtered_indices, results):
        # Draw only filtered detections on the frame
        loop = asyncio.get_event_loop()
        annotated_frame = await loop.run_in_executor(
            None, self.draw_filtered_detections, frame, results, filtered_indices
        )
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
                
                # Draw ROIs on the frame
                frame_with_rois = ROITool.draw_rois_on_frame(frame.copy(), self.rois)
                
                # Perform object detection on the frame
                results = await self.analyze_frame(frame)
                
                # Get indices of detections within ROIs using score system
                filtered_indices = self.score_system.get_filtered_indices(results)
                
                # Update score counting based on objects staying in ROIs
                self.score_system.update_score_counting(results, filtered_indices)
                
                # Visualize only filtered detections
                annotated_frame = await self.visualize_detections(frame_with_rois, filtered_indices, results)
                
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