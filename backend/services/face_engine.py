import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, model_name: str = "buffalo_sc"):
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_embedding(self, image_bytes: bytes):
        """Converts image bytes to cv2 matrix and returns the 512-d normalized embedding."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces = self.app.get(frame)
        if len(faces) == 0:
            return None, "No face detected"
        if len(faces) > 1:
            return None, "Multiple faces detected; ensure only 1 person is in frame"
        
        # Normed embedding is already L2-normalized
        embedding = faces[0].normed_embedding
        return embedding.astype(np.float32), None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

face_engine = FaceEngine()