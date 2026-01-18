import cv2
from ultralytics import YOLO
import asyncio
import numpy as np
import torch

from draw_roi import ROITool

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
        
        roi_config_0_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_half_0.yaml"
        roi_config_1_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_half_1.yaml"
        roi_config_2_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_mid.yaml"
        roi_config_half_0_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_0.yaml"
        roi_config_half_1_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_1.yaml"
        roi_config_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config.yaml"
        self.roi_config_0 = ROITool.load_roi_config(roi_config_0_path)
        self.roi_config_1 = ROITool.load_roi_config(roi_config_1_path)
        self.roi_config_2 = ROITool.load_roi_config(roi_config_2_path)
        self.roi_config = ROITool.load_roi_config(roi_config_path)
        self.roi_config_half_0 = ROITool.load_roi_config(roi_config_half_0_path)
        self.roi_config_half_1 = ROITool.load_roi_config(roi_config_half_1_path)
        
        
        # Store all ROIs in a list for easy iteration
        self.rois = [self.roi_config_half_0, self.roi_config_half_1]

    def is_detection_in_rois(self, detection_box):
        """Check if a detection (bounding box center) is within any ROI"""
        # Get center point of bounding box
        # YOLO boxes are in format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = detection_box[:4]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_point = (center_x, center_y)
        
        # Check if center is in any ROI
        for roi in self.rois:
            if ROITool.is_point_in_roi(center_point, roi):
                return True
        return False
    
    def get_filtered_indices(self, results):
        """Get indices of detections that are within ROIs"""
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        
        boxes = results[0].boxes
        filtered_indices = []
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()  # Get box coordinates
            if self.is_detection_in_rois(box):
                filtered_indices.append(i)
        
        return filtered_indices
    
    def draw_filtered_detections(self, frame, results, filtered_indices):
        """Draw only the filtered detections on the frame"""
        if len(results) == 0 or results[0].boxes is None or len(filtered_indices) == 0:
            return frame
        
        boxes = results[0].boxes
        
        # Get class names from model
        class_names = self.model.names
        
        for idx in filtered_indices:
            # Get box coordinates
            box = boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Get confidence and class
            conf = float(boxes.conf[idx].cpu().numpy()) if boxes.conf is not None else 1.0
            cls_id = int(boxes.cls[idx].cpu().numpy()) if boxes.cls is not None else 0
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y1, label_size[1] + 10)
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), 
                         (x1 + label_size[0], label_y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame

    async def analyze_frame(self, frame):
        # Perform object detection on the frame (run in thread pool to avoid blocking)
        # Explicitly pass device to ensure GPU acceleration
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            lambda: self.model(frame, device=self.device)
        )
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
                
                # Get indices of detections within ROIs
                filtered_indices = self.get_filtered_indices(results)
                
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