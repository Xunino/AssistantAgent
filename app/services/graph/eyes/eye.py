import os
import cv2
from typing import Dict, Any
from loguru import logger
from ultralytics import YOLO
from app.settings import Settings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.runnables import Runnable

settings = Settings()


class Eye(Runnable):
    def __init__(self):

        # Load YOLO model for human detection
        yolo_model_path = settings.yolo.face_weight
        face_model_path = settings.yolo.embedding_weight

        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        self.model = YOLO(yolo_model_path)

        # Load face recognition model (assuming classification-based YOLO for faces)
        if not os.path.exists(face_model_path):
            raise FileNotFoundError(
                f"Face recognition model not found: {face_model_path}"
            )
        # Load face recognition model
        self.face_model = YOLO(face_model_path)

        # Predefined vectors for known identities
        self.predefined_vectors: dict[str, Any] = {}
        self.similarity_threshold = 0.98

    def embed_face(self, face_img) -> Any:
        """Extracts embeddings from the face image using the face detection model."""
        embeded_image = self.face_model.embed(face_img)
        return embeded_image

    def recognize_person(self, face_vector) -> str:
        """Identifies the closest matching person based on cosine similarity."""
        max_similarity = -1  # Initialize to lowest similarity
        identity = "Unknown"

        for name, vector in self.predefined_vectors.items():
            if vector is None:
                logger.warning(f"No vector found for {name}.")
                continue

            similarity = cosine_similarity(face_vector, vector)[0][0]
            logger.debug(f"Similarity with {name}: {similarity}")
            if similarity > max_similarity:
                max_similarity = similarity
                identity = name

        return identity if max_similarity >= self.similarity_threshold else "Unknown"

    def process_detection_results(self, image, results, name: str):
        """Processes detection results to extract faces and update identity vectors."""
        for result in results:
            for box in result.boxes:
                label = int(box.cls[0].cpu().numpy())
                if label == 0:  # Only process humans
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    face_img = image[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue

                    face_vector = self.embed_face(face_img)
                    if face_vector is not None:
                        self.predefined_vectors[name] = face_vector

    def draw_detections(self, frame, results):
        """Draws bounding boxes and recognized identities on the frame."""
        for result in results:
            for box in result.boxes:
                label = int(box.cls[0].cpu().numpy())
                if label == 0:  # Human class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())

                    if confidence > 0.8:  # Confidence threshold
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size == 0:
                            continue

                        face_vector = self.embed_face(face_img)
                        identity = self.recognize_person(face_vector)

                        # Draw bounding box and identity
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{identity} {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )

    def run_webcam_detection(self):
        """Captures webcam input and runs human detection in real-time."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Could not open webcam.")
            return

        logger.info("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame.")
                break

            # Perform human detection
            results = self.model(frame)
            self.draw_detections(frame, results)

            # Display the frame with annotations
            cv2.imshow("Real-time Human Detection", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def invoke(self, image_path: str, **kwargs) -> Dict:
        """Runs human detection on the input image."""
        image = cv2.imread(image_path)
        results = self.model(image)
        self.draw_detections(image, results)
        return {"image": image}

    async def ainvoke(self, image_path: str, **kwargs) -> Dict:
        """Asynchronously runs human detection on the input image."""
        return self.invoke(image_path, **kwargs)
