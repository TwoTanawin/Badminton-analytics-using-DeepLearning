import cv2
import asyncio
import os
import numpy as np
from pathlib import Path

class ExtractVdoToImg:
    def __init__(self, vdo_path: str, output_base_dir: str = None):
        self.vdo_path = vdo_path
        self.frame_count = 0
        
        # Create output directory based on video filename
        video_name = Path(vdo_path).stem
        if output_base_dir:
            self.img_dir = os.path.join(output_base_dir, video_name)
        else:
            self.img_dir = os.path.join(os.path.dirname(vdo_path), 'img', video_name)
        os.makedirs(self.img_dir, exist_ok=True)

    async def extract(self, target_fps: int = 30):
        cap = cv2.VideoCapture(self.vdo_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.vdo_path}")
            return
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing: {os.path.basename(self.vdo_path)}")
        print(f"  Video FPS: {video_fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Target extraction FPS: {target_fps}")
        
        # Calculate frame skip to extract at target_fps
        if video_fps > 0:
            frame_skip = max(1, int(video_fps / target_fps))
        else:
            frame_skip = 1
        
        print(f"  Saving every {frame_skip} frame(s)")
        print(f"  Output directory: {self.img_dir}")
        
        frame_index = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame if it matches the target FPS interval
            if frame_index % frame_skip == 0:
                await self.save_frame(frame, saved_count)
                saved_count += 1
            
            frame_index += 1
        
        cap.release()
        print(f"  Saved {saved_count} frames\n")
    
    async def save_frame(self, frame: np.ndarray, frame_number: int):
        img_path = os.path.join(self.img_dir, f'frame_{frame_number:06d}.jpg')
        cv2.imwrite(img_path, frame)
        return img_path
    
async def process_videos(videos_dir: str, output_dir: str, target_fps: int = 30):
    """Process all video files in the specified directory"""
    videos_path = Path(videos_dir)
    
    # Find all video files (including in subdirectories)
    video_extensions = {'.mov', '.mp4', '.avi', '.mkv', '.MOV', '.MP4', '.AVI', '.MKV'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(videos_path.rglob(f'*{ext}'))
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
    
    print(f"Found {len(video_files)} video file(s)\n")
    
    # Process each video
    for video_path in sorted(video_files):
        extractor = ExtractVdoToImg(str(video_path), output_base_dir=output_dir)
        await extractor.extract(target_fps)
    
async def main():
    videos_dir = '/Volumes/PortableSSD/Two/Badminton-analytics/videos'
    output_dir = '/Volumes/PortableSSD/Two/Badminton-analytics/analytics/images'
    target_fps = 30
    
    await process_videos(videos_dir, output_dir, target_fps)
    print("All videos processed!")

if __name__ == '__main__':
    asyncio.run(main())
        