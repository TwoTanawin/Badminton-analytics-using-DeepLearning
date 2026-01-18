
class YoloAnalytics:
    def __init__(self):
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
    def analyze_frame(self, frame):
        results = self.model(frame)
        return results