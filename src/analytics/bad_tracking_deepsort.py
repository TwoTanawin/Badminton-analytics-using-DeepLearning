import cv2
from ultralytics import YOLO
import asyncio
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

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
        
        # Initialize YOLO model with the detected device (for detection only)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Initialize Deep SORT tracker
        # max_age: maximum number of missed frames before a track is deleted
        # n_init: number of consecutive detections before the track is confirmed
        # max_cosine_distance: maximum cosine distance for association
        # embedder: feature extractor model ('mobilenet' or 'torchreid')
        use_gpu = (self.device == 'cuda')
        self.tracker = DeepSort(
            max_age=50,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",  # or "torchreid" for better accuracy but slower
            embedder_wts=None,
            polygon=False
        )
        
        # Store class and confidence info for each track_id
        # This helps preserve detection information through tracking
        self.track_info = {}  # {track_id: {'class_id': int, 'confidence': float}}

    async def analyze_frame(self, frame):
        # Perform object detection only (not tracking) - Deep SORT will handle tracking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self.model(frame, device=self.device))
        return results
    
    def extract_detections(self, results):
        """Extract detections from YOLO results in format required by Deep SORT
        Deep SORT expects: ([left, top, width, height], confidence, class_id)
        """
        detections = []
        detection_info = []  # Store class and confidence for each detection
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            box_coords = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = box_coords[i][:4]
                conf = float(confidences[i])
                cls = int(classes[i])
                
                # Convert from [x1, y1, x2, y2] to [left, top, width, height]
                left = x1
                top = y1
                width = x2 - x1
                height = y2 - y1
                
                # Deep SORT expects: ([left, top, width, height], confidence, class_id)
                bbox = [left, top, width, height]
                detections.append((bbox, conf, cls))
                detection_info.append({'class_id': cls, 'confidence': conf})
        
        return detections, detection_info

    async def visualize_detections(self, frame, tracked_objects):
        # Draw bounding boxes, labels, and tracking IDs from Deep SORT
        annotated_frame = frame.copy()
        
        # Get class names from model
        class_names = self.model.names
        
        for track in tracked_objects:
            # Deep SORT returns: (bbox, track_id, class_id, confidence)
            # Format: [x1, y1, x2, y2] or (ltrb, track_id, class_id, confidence)
            if len(track) >= 2:
                bbox = track[0]  # [x1, y1, x2, y2]
                track_id = track[1]
                
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # Get class and confidence if available
                class_id = track[2] if len(track) > 2 else 0
                confidence = track[3] if len(track) > 3 else 1.0
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                # Draw bounding box
                color = self._get_color_for_track_id(track_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with tracking ID, class name, and confidence
                label = f"ID: {track_id} | {class_name} | {confidence:.2f}"
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

    async def display_frame(self, frame):
        # Display the annotated frame (must run on main thread - OpenCV GUI requirement)
        cv2.imshow('Annotated Frame', frame)
        # Non-blocking key check (1ms timeout) - allows continuous playback
        key = cv2.waitKey(1) & 0xFF
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
                
                # Step 1: Get detections from YOLO
                results = await self.analyze_frame(frame)
                
                # Step 2: Extract detections in Deep SORT format
                detections, detection_info = self.extract_detections(results)
                
                # Step 3: Update Deep SORT tracker
                # The tracker.update_tracks() returns a generator of Track objects
                loop = asyncio.get_event_loop()
                
                def update_tracker():
                    return list(self.tracker.update_tracks(detections, frame=frame))
                
                tracked_objects = await loop.run_in_executor(None, update_tracker)
                
                # Step 4: Match tracked objects with detection info and update track_info
                # Extract track data from Deep SORT tracker objects
                tracks = []
                for track in tracked_objects:
                    if track.is_confirmed():
                        ltrb = track.to_ltrb()  # Get bounding box [left, top, right, bottom]
                        track_id = track.track_id
                        
                        # Try to get class and confidence from track attributes first
                        class_id = getattr(track, 'det_class', None)
                        confidence = getattr(track, 'det_conf', None)
                        
                        # If not available, try to match with current detections by position
                        if class_id is None or confidence is None:
                            # Find the closest detection to this track
                            track_center = [(ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2]
                            min_dist = float('inf')
                            best_match_idx = -1
                            
                            for idx, det in enumerate(detections):
                                det_bbox = det[0]
                                det_center = [(det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2]
                                dist = ((track_center[0] - det_center[0])**2 + (track_center[1] - det_center[1])**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match_idx = idx
                            
                            # Use matched detection info or fall back to stored info
                            if best_match_idx >= 0 and min_dist < 50:  # Threshold for matching
                                class_id = detection_info[best_match_idx]['class_id']
                                confidence = detection_info[best_match_idx]['confidence']
                                # Update stored info for this track
                                self.track_info[track_id] = {'class_id': class_id, 'confidence': confidence}
                            else:
                                # Use stored info if available, otherwise default
                                if track_id in self.track_info:
                                    class_id = self.track_info[track_id]['class_id']
                                    confidence = self.track_info[track_id]['confidence']
                                else:
                                    class_id = 0
                                    confidence = 1.0
                        else:
                            # Update stored info with new values
                            self.track_info[track_id] = {'class_id': class_id, 'confidence': confidence}
                        
                        # Store track info: [bbox, track_id, class_id, confidence]
                        tracks.append([ltrb, track_id, class_id, confidence])
                
                # Clean up old track info (remove tracks that are no longer active)
                active_track_ids = {track[1] for track in tracks}
                self.track_info = {tid: info for tid, info in self.track_info.items() if tid in active_track_ids}
                
                # Step 5: Visualize tracked objects
                annotated_frame = await self.visualize_detections(frame, tracks)
                
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