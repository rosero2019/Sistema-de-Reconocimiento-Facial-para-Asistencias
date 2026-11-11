"""Une cámara + OpenCV + base de datos y produce el video anotado en tiempo real."""
import functools
import logging
import threading
import time

import cv2
import numpy as np
from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from .. import servicios
from .camara import Camara
from .galeria import Galeria
from .motor import MotorFacial

log = logging.getLogger(__name__)

VERDE = (60, 200, 60)
ROJO = (60, 60, 230)
AMARILLO = (0, 210, 255)


class Procesador:
    def __init__(self):
        cfg = settings.VISION
        self.motor = MotorFacial(cfg["MODELO_DETECTOR"], cfg["MODELO_RECONOCEDOR"])
        self.galeria = Galeria(cfg["UMBRAL_SIMILITUD"])
        self.cuadros_confirmacion = cfg["CUADROS_CONFIRMACION"]
        self.recargar_galeria()
        self.camara = Camara(cfg["CAMARA_INDICE"])

        self._jpeg = None
        self._lock = threading.Lock()
        self._rachas: dict[int, int] = {}       # persona_id -> cuadros seguidos reconocido
        self._marcados: dict[int, object] = {}  # persona_id -> fecha en que ya se marcó
        self.fps = 0.0
        threading.Thread(target=self._bucle, daemon=True).start()

    # ------------------------------------------------------------------
    def recargar_galeria(self):
        close_old_connections()
        self.galeria.cargar(servicios.vectores_registrados())
        log.info("Galería cargada: %s personas", len(self.galeria))

    def jpeg(self):
        with self._lock:
            return self._jpeg

    # ------------------------------------------------------------------
    def _bucle(self):
        while True:
            frame = self.camara.leer()
            if frame is None:
                # Sin señal: se muestra un aviso en lugar de dejar el video congelado
                with self._lock:
                    self._jpeg = imagen_sin_senal()
                time.sleep(0.2)
                continue
            inicio = time.perf_counter()
            try:
                anotado = self.procesar(frame)
            except Exception:  # un error en un cuadro no debe detener el video
                log.exception("Error procesando cuadro")
                anotado = frame
            ok, buffer = cv2.imencode(".jpg", anotado, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._lock:
                    self._jpeg = buffer.tobytes()
            duracion = time.perf_counter() - inicio
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(duracion, 1e-3))
            time.sleep(0.01)

    def procesar(self, frame: np.ndarray) -> np.ndarray:
        mejorado = self.motor.mejorar_iluminacion(frame)
        caras = self.motor.detectar(mejorado)
        vistos = set()

        for cara in caras:
            vector = self.motor.extraer_vector(mejorado, cara)
            coincidencia = self.galeria.identificar(vector)

            if coincidencia is None:
                self._dibujar(frame, cara, "Desconocido", ROJO)
                continue

            pid = coincidencia.persona_id
            vistos.add(pid)
            self._rachas[pid] = self._rachas.get(pid, 0) + 1

            if self._ya_marcado(pid):
                color, sufijo = VERDE, " - OK"
            elif self._rachas[pid] >= self.cuadros_confirmacion:
                self._marcar(pid, coincidencia.similitud)
                color, sufijo = VERDE, " - OK"
            else:
                color, sufijo = AMARILLO, " ..."
            texto = f"{coincidencia.nombre} {coincidencia.similitud:.0%}{sufijo}"
            self._dibujar(frame, cara, texto, color)

        # Si alguien deja de verse, su racha vuelve a cero
        self._rachas = {pid: n for pid, n in self._rachas.items() if pid in vistos}
        self._dibujar_encabezado(frame, len(caras))
        return frame

    # ------------------------------------------------------------------
    def _ya_marcado(self, persona_id) -> bool:
        return self._marcados.get(persona_id) == timezone.localdate()

    def _marcar(self, persona_id, similitud):
        close_old_connections()
        try:
            _, creada = servicios.registrar_asistencia(persona_id, similitud)
            self._marcados[persona_id] = timezone.localdate()
            if creada:
                log.info("Asistencia registrada: persona %s", persona_id)
        except Exception:
            log.exception("No se pudo registrar la asistencia")

    @staticmethod
    def _dibujar(frame, cara, texto, color):
        x, y, w, h = map(int, cara[:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        for i in range(5):  # ojos, nariz y comisuras de la boca
            px, py = int(cara[4 + 2 * i]), int(cara[5 + 2 * i])
            cv2.circle(frame, (px, py), 2, color, -1)
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_texto = max(y - 8, th + 8)
        cv2.rectangle(frame, (x, y_texto - th - 6), (x + tw + 8, y_texto + 4), color, -1)
        cv2.putText(frame, texto, (x + 4, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    def _dibujar_encabezado(self, frame, n_caras):
        ahora = timezone.localtime().strftime("%d/%m/%Y %H:%M:%S")
        texto = f"{ahora} | Rostros: {n_caras} | FPS: {self.fps:.0f} | Camara: {self.camara.fps:.0f} fps"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (30, 30, 30), -1)
        cv2.putText(frame, texto, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def capturar_muestras(self, cantidad: int, tiempo_max: float = 8.0):
        """Toma varias muestras del rostro frente a la cámara (para el registro).

        Solo acepta cuadros con exactamente un rostro, para no mezclar personas.
        """
        vectores = []
        limite = time.monotonic() + tiempo_max
        while len(vectores) < cantidad and time.monotonic() < limite:
            frame = self.camara.leer()
            if frame is not None:
                mejorado = self.motor.mejorar_iluminacion(frame)
                caras = self.motor.detectar(mejorado)
                if len(caras) == 1:
                    vectores.append(self.motor.extraer_vector(mejorado, caras[0]))
            time.sleep(0.3)  # pausa para obtener muestras algo distintas
        return vectores


# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def imagen_sin_senal() -> bytes:
    """JPEG que se muestra mientras la cámara se conecta o se reconecta."""
    imagen = np.full((480, 640, 3), 35, dtype=np.uint8)
    cv2.putText(imagen, "Esperando senal de la camara...", (110, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, "Revisa que este conectada y que otra app no la use", (95, 265),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", imagen)[1].tobytes()


_instancia = None
_lock_instancia = threading.Lock()


def obtener_procesador() -> Procesador:
    """Crea el procesador la primera vez que se necesita (no al iniciar Django)."""
    global _instancia
    with _lock_instancia:
        if _instancia is None:
            _instancia = Procesador()
        return _instancia