"""Motor facial basado 100% en OpenCV.

- Detección:       cv2.FaceDetectorYN  (modelo YuNet)
- Reconocimiento:  cv2.FaceRecognizerSF (modelo SFace, vector de 128 valores)
- Procesamiento:   mejora de iluminación con CLAHE antes de detectar

Este módulo no depende de Django, así se puede probar por separado.

Los modelos de OpenCV no se pueden usar desde dos hilos a la vez: el video se
procesa en un hilo y el registro de personas en otro. Por eso cada uso de los
modelos pasa por un candado (lock).
"""
import threading
from pathlib import Path

import cv2
import numpy as np


class ModelosNoEncontrados(FileNotFoundError):
    pass


class MotorFacial:
    def __init__(self, ruta_detector, ruta_reconocedor, confianza_minima=0.85):
        for ruta in (ruta_detector, ruta_reconocedor):
            if not Path(ruta).is_file():
                raise ModelosNoEncontrados(
                    f"No se encontró el modelo {ruta}. Ejecuta: python descargar_modelos.py"
                )
        self.detector = cv2.FaceDetectorYN.create(
            str(ruta_detector), "", (320, 320), confianza_minima, 0.3, 5000
        )
        self.reconocedor = cv2.FaceRecognizerSF.create(str(ruta_reconocedor), "")
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._lock = threading.Lock()

    # ---------- Procesamiento ----------
    def mejorar_iluminacion(self, frame: np.ndarray) -> np.ndarray:
        """Ecualiza el brillo (canal L de LAB) para ambientes oscuros o con contraluz."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        with self._lock:
            l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ---------- Detección ----------
    def detectar(self, frame: np.ndarray) -> np.ndarray:
        """Devuelve un arreglo (N, 15): x, y, w, h, 5 puntos faciales (x, y) y la confianza."""
        alto, ancho = frame.shape[:2]
        with self._lock:
            self.detector.setInputSize((ancho, alto))
            _, caras = self.detector.detect(frame)
        if caras is None:
            return np.empty((0, 15), dtype=np.float32)
        return caras

    # ---------- Reconocimiento ----------
    def extraer_vector(self, frame: np.ndarray, cara: np.ndarray) -> np.ndarray:
        """Alinea el rostro con los puntos faciales y devuelve su vector normalizado (128,)."""
        with self._lock:
            alineado = self.reconocedor.alignCrop(frame, cara)
            vector = self.reconocedor.feature(alineado).flatten().astype(np.float32)
        return normalizar(vector)


def normalizar(vector: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(vector)
    return vector / norma if norma > 0 else vector


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalizar(a), normalizar(b)))