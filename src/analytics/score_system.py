import cv2
import numpy as np
from draw_roi import ROITool


class ScoreSystem:
    """
    Score system for tracking objects in ROIs and calculating scores.
    Handles ROI-based detection filtering, frame counting, and score calculation.
    """
    
    def __init__(self, rois, frame_threshold=10):
        """
        Initialize the score system.
        
        Args:
            rois: List of ROI configurations (from ROITool.load_roi_config)
            frame_threshold: Number of frames an object must stay in ROI to score (default: 10)
        """
        self.rois = rois
        self.frame_threshold = frame_threshold
        
        # Score counting: track frame counts per track_id per ROI
        # Structure: {track_id: {roi_index: frame_count}}
        self.track_roi_frame_counts = {}
        
        # Track which objects have already been scored (to avoid double counting)
        # Structure: {track_id: {roi_index: bool}}
        self.track_roi_scored = {}
        
        # Score per ROI
        self.scores = [0] * len(rois)
    
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
        """Get the ROI index that a detection is in, or None if not in any ROI"""
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
    
    def get_frame_count(self, track_id, roi_index):
        """
        Get the frame count for a specific track_id in a specific ROI.
        
        Args:
            track_id: Track ID of the object
            roi_index: Index of the ROI
            
        Returns:
            Frame count (0 if not tracked)
        """
        if track_id is None:
            return 0
        if track_id in self.track_roi_frame_counts:
            if roi_index in self.track_roi_frame_counts[track_id]:
                return self.track_roi_frame_counts[track_id][roi_index]
        return 0
    
    def update_score_counting(self, results, filtered_indices):
        """
        Update score counting based on objects staying in ROIs for more than frame_threshold frames.
        
        Args:
            results: YOLO detection/tracking results
            filtered_indices: Indices of detections within ROIs
        """
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
            
            # Check if object has been in ROI for more than frame_threshold frames
            if self.track_roi_frame_counts[track_id][roi_index] > self.frame_threshold:
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
    
    def draw_scores(self, frame):
        """
        Draw score information on the frame.
        
        Args:
            frame: Frame to draw scores on
            
        Returns:
            Frame with scores drawn
        """
        # Draw scores in top-left corner
        y_offset = 30
        for idx, score in enumerate(self.scores):
            score_text = f"ROI {idx} Score: {score}"
            cv2.putText(frame, score_text, (10, y_offset + idx * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame
    
    def get_scores(self):
        """Get current scores for all ROIs"""
        return self.scores.copy()
    
    def reset_scores(self):
        """Reset all scores and tracking data"""
        self.track_roi_frame_counts = {}
        self.track_roi_scored = {}
        self.scores = [0] * len(self.rois)
