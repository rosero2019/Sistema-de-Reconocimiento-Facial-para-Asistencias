"""Captura de video en tiempo real.

La cámara se lee en un hilo aparte para no bloquear al servidor:
- Siempre se guarda solo el último cuadro (no se acumula retraso).
- Si la cámara se desconecta o deja de enviar imagen, se reconecta sola.
- Se mide cuántos cuadros por segundo entrega la cámara.
"""
import logging
import platform
import threading
import time

import cv2

log = logging.getLogger(__name__)

# Lecturas fallidas seguidas antes de intentar reconectar (~1 segundo)
FALLOS_PARA_RECONECTAR = 20
# Si el último cuadro tiene más de estos segundos, se considera que no hay señal
SEGUNDOS_SIN_SENAL = 2.0
# Pausa antes de volver a abrir la cámara
ESPERA_RECONEXION = 1.0


class Camara:
    def __init__(self, indice: int = 0, ancho: int = 640, alto: int = 480):
        self.indice = indice
        self.ancho, self.alto = ancho, alto
        self.captura = self._abrir()
        if self.captura is None:
            raise RuntimeError(f"No se pudo abrir la cámara {indice}")

        self.fps = 0.0
        self.reconexiones = 0
        self._frame = None
        self._momento = 0.0  # time.monotonic() del último cuadro recibido
        self._lock = threading.Lock()
        self._activa = True
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    # ------------------------------------------------------------------
    def _abrir(self):
        """Prueba los backends disponibles y devuelve la captura abierta, o None."""
        if platform.system() == "Windows":
            # DirectShow abre la cámara mucho más rápido; MSMF queda como respaldo
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        else:
            backends = [cv2.CAP_ANY]
        for backend in backends:
            captura = cv2.VideoCapture(self.indice, backend)
            if captura.isOpened():
                captura.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
                captura.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
                captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos retraso entre la realidad y el video
                return captura
            captura.release()
        return None

    def _leer(self):
        fallos = 0
        anterior = time.perf_counter()
        while self._activa:
            ok, frame = self.captura.read() if self.captura is not None else (False, None)
            if ok and frame is not None:
                fallos = 0
                ahora = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(ahora - anterior, 1e-3))
                anterior = ahora
                espejo = cv2.flip(frame, 1)  # efecto espejo
                with self._lock:
                    self._frame = espejo
                    self._momento = time.monotonic()
                continue

            fallos += 1
            if fallos >= FALLOS_PARA_RECONECTAR:
                self._reconectar()
                fallos = 0
            else:
                time.sleep(0.05)

        if self.captura is not None:
            self.captura.release()

    def _reconectar(self):
        log.warning("La cámara %s dejó de responder, intentando reconectar...", self.indice)
        if self.captura is not None:
            self.captura.release()
        time.sleep(ESPERA_RECONEXION)
        self.captura = self._abrir()
        if self.captura is not None:
            self.reconexiones += 1
            log.info("Cámara %s reconectada", self.indice)

    # ------------------------------------------------------------------
    @property
    def conectada(self) -> bool:
        """True si llegó un cuadro hace poco."""
        with self._lock:
            return self._frame is not None and time.monotonic() - self._momento < SEGUNDOS_SIN_SENAL

    def leer(self):
        """Devuelve una copia del último cuadro, o None si aún no hay o la señal se perdió."""
        with self._lock:
            if self._frame is None or time.monotonic() - self._momento >= SEGUNDOS_SIN_SENAL:
                return None
            return self._frame.copy()

    def liberar(self):
        """Detiene el hilo; el propio hilo libera la cámara al salir."""
        self._activa = False
        self._hilo.join(timeout=2)