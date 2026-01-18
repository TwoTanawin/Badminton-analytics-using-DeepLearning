import cv2
from ultralytics import YOLO
import asyncio
import torch
import numpy as np
from draw_roi import ROITool

class VDOAnalytics:
    def __init__(self):
        model_path = '/Volumes/PortableSSD/Two/Badminton-analytics/analytics/model/best_m_bad_stc.pt'  # Path to the pre-trained YOLO model
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
        
        roi_config_half_0_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_0.yaml"
        roi_config_half_1_path = "/Volumes/PortableSSD/Two/Badminton-analytics/analytics/src/config/roi/frame_000575_roi_config_1.yaml"

        self.roi_config_half_0 = ROITool.load_roi_config(roi_config_half_0_path)
        self.roi_config_half_1 = ROITool.load_roi_config(roi_config_half_1_path)
        
        # Store all ROIs in a list for easy iteration
        self.rois = [self.roi_config_half_0, self.roi_config_half_1]
        
        # Score counting: track frame counts per track_id per ROI
        # Structure: {track_id: {roi_index: frame_count}}
        self.track_roi_frame_counts = {}
        
        # Track which objects have already been scored (to avoid double counting)
        # Structure: {track_id: {roi_index: bool}}
        self.track_roi_scored = {}
        
        # Score per ROI
        self.scores = [0, 0]  # One score per ROI

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
    
    def get_roi_index_for_detection(self, detection_box):
        """Get the ROI index (0 or 1) that a detection is in, or None if not in any ROI"""
        # Get center point of bounding box
        x1, y1, x2, y2 = detection_box[:4]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_point = (center_x, center_y)
        
        # Check which ROI the center is in
        for idx, roi in enumerate(self.rois):
            if ROITool.is_point_in_roi(center_point, roi):
                return idx
        return None
    
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
    
    def update_score_counting(self, results, filtered_indices):
        """Update score counting based on objects staying in ROIs for more than 10 frames"""
        boxes = results[0].boxes if len(results) > 0 and results[0].boxes is not None else None
        track_ids = boxes.id.cpu().numpy() if boxes is not None and boxes.id is not None else None
        
        # Track which track_ids are currently in which ROIs
        current_track_roi_map = {}  # {track_id: roi_index}
        
        if boxes is not None and track_ids is not None and len(filtered_indices) > 0:
            # Map current track IDs to their ROIs
            for idx in filtered_indices:
                box = boxes.xyxy[idx].cpu().numpy()
                track_id = int(track_ids[idx]) if idx < len(track_ids) else None
                
                if track_id is None:
                    continue
                
                roi_index = self.get_roi_index_for_detection(box)
                if roi_index is not None:
                    current_track_roi_map[track_id] = roi_index
        
        # Update frame counts for objects currently in ROIs
        for track_id, roi_index in current_track_roi_map.items():
            # Initialize tracking structures if needed
            if track_id not in self.track_roi_frame_counts:
                self.track_roi_frame_counts[track_id] = {}
            if track_id not in self.track_roi_scored:
                self.track_roi_scored[track_id] = {}
            
            # Check if this track_id was in a different ROI last frame
            # If so, reset the count for the previous ROI
            for prev_roi_idx in list(self.track_roi_frame_counts[track_id].keys()):
                if prev_roi_idx != roi_index:
                    # Object moved to different ROI, reset previous ROI count and scored status
                    self.track_roi_frame_counts[track_id][prev_roi_idx] = 0
                    if prev_roi_idx in self.track_roi_scored[track_id]:
                        self.track_roi_scored[track_id][prev_roi_idx] = False
            
            # Initialize or increment frame count for this track_id in this ROI
            if roi_index not in self.track_roi_frame_counts[track_id]:
                self.track_roi_frame_counts[track_id][roi_index] = 0
            
            self.track_roi_frame_counts[track_id][roi_index] += 1
            
            # Check if object has been in ROI for more than 10 frames
            if self.track_roi_frame_counts[track_id][roi_index] > 10:
                # Check if we've already scored this track_id for this ROI
                if roi_index not in self.track_roi_scored[track_id]:
                    self.track_roi_scored[track_id][roi_index] = False
                
                # If not scored yet, increment score for the OPPOSITE ROI
                # If object stays in ROI 0, ROI 1 gets the point, and vice versa
                if not self.track_roi_scored[track_id][roi_index]:
                    opposite_roi_index = 1 - roi_index  # 0 becomes 1, 1 becomes 0
                    self.scores[opposite_roi_index] += 1
                    self.track_roi_scored[track_id][roi_index] = True
                    print(f"Score! Track ID {track_id} stayed in ROI {roi_index} for {self.track_roi_frame_counts[track_id][roi_index]} frames -> ROI {opposite_roi_index} gets 1 point")
        
        # Reset frame counts for objects that left ROIs (not in current_track_roi_map)
        for track_id in list(self.track_roi_frame_counts.keys()):
            if track_id not in current_track_roi_map:
                # Object left ROI - reset all counts and scored status for this track_id
                for roi_index in list(self.track_roi_frame_counts[track_id].keys()):
                    self.track_roi_frame_counts[track_id][roi_index] = 0
                    if roi_index in self.track_roi_scored[track_id]:
                        self.track_roi_scored[track_id][roi_index] = False
    
    def draw_filtered_detections(self, frame, results, filtered_indices):
        """Draw only the filtered detections on the frame with tracking IDs"""
        if len(results) == 0 or results[0].boxes is None or len(filtered_indices) == 0:
            # Still draw scores even if no detections
            self._draw_scores(frame)
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
            
            # Get frame count for this track_id in its ROI
            frame_count_text = ""
            if track_id is not None:
                roi_index = self.get_roi_index_for_detection(box)
                if roi_index is not None and track_id in self.track_roi_frame_counts:
                    if roi_index in self.track_roi_frame_counts[track_id]:
                        frame_count = self.track_roi_frame_counts[track_id][roi_index]
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
        
        # Draw scores on frame
        self._draw_scores(frame)
        
        return frame
    
    def _draw_scores(self, frame):
        """Draw score information on the frame"""
        # Draw scores in top-left corner
        y_offset = 30
        for idx, score in enumerate(self.scores):
            score_text = f"ROI {idx} Score: {score}"
            cv2.putText(frame, score_text, (10, y_offset + idx * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
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
        # Perform object detection and tracking using ByteTrack
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            lambda: self.model.track(frame, device=self.device, tracker=self.tracker, persist=True)
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
                
                # Update score counting based on objects staying in ROIs
                self.update_score_counting(results, filtered_indices)
                
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