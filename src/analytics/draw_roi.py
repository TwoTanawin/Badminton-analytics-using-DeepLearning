import cv2
import os
import yaml
import numpy as np

class ROITool:
    
    @staticmethod
    def load_roi_config(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'roi_coordinates' not in config:
            raise ValueError("Invalid config file: 'roi_coordinates' not found")
        
        return config['roi_coordinates']
    
    @staticmethod
    def draw_roi_on_image(image_path, roi_coordinates):
        """Draw ROI polygon on the image"""
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from: {image_path}")
        
        # Convert coordinates to numpy array for drawing
        points = np.array(roi_coordinates, dtype=np.int32)
        
        # Draw filled polygon with transparency
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        alpha = 0.3
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Draw polygon outline
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw points and numbers
        for i, point in enumerate(roi_coordinates):
            x, y = point
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(image, str(i+1), (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return image
    
    @staticmethod
    def is_point_in_roi(point, roi_coordinates):
        """Check if a point is inside a ROI polygon"""
        roi_points = np.array(roi_coordinates, dtype=np.int32)
        result = cv2.pointPolygonTest(roi_points, tuple(point), False)
        return result >= 0  # >= 0 means inside or on edge
    
    @staticmethod
    def draw_rois_on_frame(frame, rois, colors=None):
        """Draw all ROIs on the frame"""
        if colors is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
        
        for idx, roi in enumerate(rois):
            color = colors[idx % len(colors)]
            points = np.array(roi, dtype=np.int32)
            
            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            alpha = 0.2
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw polygon outline
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
            
            # Draw ROI label
            if len(points) > 0:
                # Get centroid for label placement
                M = cv2.moments(points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame, f"ROI {idx}", (cx - 30, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame