"""Casos de prueba. Ejecutar con: python manage.py test"""
import time as reloj
from datetime import datetime, time
from unittest import mock

import numpy as np
from django.db import IntegrityError, transaction
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from . import servicios
from .forms import PersonaForm
from .models import Asistencia, Persona, Rostro
from .vision import camara as modulo_camara
from .vision.camara import Camara
from .vision.galeria import Galeria
from .vision.motor import similitud_coseno


def vector_aleatorio(semilla):
    return np.random.default_rng(semilla).normal(size=128).astype(np.float32)


class GaleriaTests(TestCase):
    """Pruebas del reconocimiento (comparación de vectores) sin necesidad de cámara."""

    def setUp(self):
        self.ana, self.luis = vector_aleatorio(1), vector_aleatorio(2)
        self.galeria = Galeria(umbral=0.4)
        self.galeria.cargar([(1, "Ana", self.ana), (2, "Luis", self.luis)])

    def test_reconoce_a_la_persona_correcta(self):
        ruido = vector_aleatorio(99) * 0.2
        resultado = self.galeria.identificar(self.ana + ruido)
        self.assertEqual(resultado.nombre, "Ana")
        self.assertGreater(resultado.similitud, 0.9)

    def test_rechaza_a_un_desconocido(self):
        self.assertIsNone(self.galeria.identificar(vector_aleatorio(50)))

    def test_galeria_vacia(self):
        self.assertIsNone(Galeria(0.4).identificar(self.ana))

    def test_similitud_de_un_vector_consigo_mismo_es_1(self):
        self.assertAlmostEqual(similitud_coseno(self.ana, self.ana), 1.0, places=5)


@override_settings(HORA_LIMITE_ENTRADA="08:15")
class AsistenciaTests(TestCase):
    def setUp(self):
        self.persona = Persona.objects.create(nombre="Ana Pérez", cedula="0912345678")

    def momento(self, hora, minuto):
        return timezone.make_aware(datetime(2026, 11, 5, hora, minuto))

    def test_llegada_a_tiempo_es_puntual(self):
        self.assertEqual(servicios.calcular_estado(time(8, 0)), Asistencia.Estado.PUNTUAL)

    def test_llegada_tarde_es_atrasado(self):
        self.assertEqual(servicios.calcular_estado(time(8, 30)), Asistencia.Estado.ATRASADO)

    def test_no_duplica_asistencia_el_mismo_dia(self):
        _, creada1 = servicios.registrar_asistencia(self.persona.id, 0.8, self.momento(8, 0))
        _, creada2 = servicios.registrar_asistencia(self.persona.id, 0.9, self.momento(9, 0))
        self.assertTrue(creada1)
        self.assertFalse(creada2)
        self.assertEqual(Asistencia.objects.count(), 1)

    def test_base_de_datos_impide_duplicados(self):
        servicios.registrar_asistencia(self.persona.id, 0.8, self.momento(8, 0))
        with self.assertRaises(IntegrityError), transaction.atomic():
            Asistencia.objects.create(persona=self.persona, fecha=self.momento(8, 0).date(),
                                      hora=time(10, 0), estado="PUNTUAL", similitud=0.7)


class RegistroPersonaTests(TestCase):
    def test_guarda_y_recupera_vectores(self):
        vectores = [vector_aleatorio(i) for i in range(3)]
        persona = servicios.registrar_persona("Luis Mora", "0923456789", "5to A", vectores)
        self.assertEqual(persona.rostros.count(), 3)
        recuperado = Rostro.objects.filter(persona=persona).first().get_vector()
        self.assertEqual(recuperado.shape, (128,))

    def test_sin_muestras_no_registra(self):
        with self.assertRaises(ValueError):
            servicios.registrar_persona("Luis Mora", "0923456789", "", [])
        self.assertEqual(Persona.objects.count(), 0)

    def test_formulario_valida_cedula(self):
        form = PersonaForm(data={"nombre": "ana", "cedula": "12ab", "curso": ""})
        self.assertFalse(form.is_valid())
        self.assertIn("cedula", form.errors)


class VistasTests(TestCase):
    """Las páginas cargan sin abrir la cámara."""

    def test_paginas_responden(self):
        for nombre in ["panel", "personas", "historial", "api_hoy"]:
            with self.subTest(pagina=nombre):
                self.assertEqual(self.client.get(reverse(nombre)).status_code, 200)

    def test_exportar_csv(self):
        persona = Persona.objects.create(nombre="Ana", cedula="0912345678")
        servicios.registrar_asistencia(persona.id, 0.85)
        respuesta = self.client.get(reverse("historial"), {"formato": "csv"})
        self.assertEqual(respuesta["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("Ana", respuesta.content.decode("utf-8"))


class CapturaFalsa:
    """Imita a cv2.VideoCapture para probar la captura sin una cámara real."""

    abiertas = []

    def __init__(self, indice, backend=None, falla_despues_de=None):
        self.abierta = True
        self.lecturas = 0
        self.falla_despues_de = falla_despues_de
        CapturaFalsa.abiertas.append(self)

    def isOpened(self):
        return self.abierta

    def set(self, propiedad, valor):
        return True

    def read(self):
        self.lecturas += 1
        reloj.sleep(0.01)
        if self.falla_despues_de is not None and self.lecturas > self.falla_despues_de:
            return False, None
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32] = 255  # mitad izquierda blanca
        return True, frame

    def release(self):
        self.abierta = False


def esperar(condicion, segundos=3.0):
    limite = reloj.monotonic() + segundos
    while reloj.monotonic() < limite:
        if condicion():
            return True
        reloj.sleep(0.02)
    return False


class CamaraTests(TestCase):
    """Pruebas de la captura de video en tiempo real con una cámara simulada."""

    def setUp(self):
        CapturaFalsa.abiertas = []

    def test_lee_cuadros_con_efecto_espejo(self):
        with mock.patch.object(modulo_camara.cv2, "VideoCapture", CapturaFalsa):
            camara = Camara(0)
            try:
                self.assertTrue(esperar(lambda: camara.leer() is not None))
                frame = camara.leer()
                # Por el espejo, la mitad blanca pasa a la derecha
                self.assertEqual(frame[0, -1, 0], 255)
                self.assertEqual(frame[0, 0, 0], 0)
                self.assertTrue(camara.conectada)
            finally:
                camara.liberar()

    def test_leer_devuelve_una_copia(self):
        with mock.patch.object(modulo_camara.cv2, "VideoCapture", CapturaFalsa):
            camara = Camara(0)
            try:
                esperar(lambda: camara.leer() is not None)
                frame = camara.leer()
                frame[:] = 7
                self.assertNotEqual(camara.leer()[0, 0, 0], 7)
            finally:
                camara.liberar()

    def test_error_si_no_abre_la_camara(self):
        class SinCamara(CapturaFalsa):
            def isOpened(self):
                return False

        with mock.patch.object(modulo_camara.cv2, "VideoCapture", SinCamara):
            with self.assertRaises(RuntimeError):
                Camara(5)

    def test_se_reconecta_si_la_camara_deja_de_responder(self):
        def fabrica(indice, backend=None):
            # La primera captura se "desconecta" a las 3 lecturas; las siguientes funcionan
            return CapturaFalsa(indice, backend, falla_despues_de=3 if not CapturaFalsa.abiertas else None)

        with mock.patch.object(modulo_camara.cv2, "VideoCapture", side_effect=fabrica), \
                mock.patch.object(modulo_camara, "FALLOS_PARA_RECONECTAR", 3), \
                mock.patch.object(modulo_camara, "ESPERA_RECONEXION", 0.01):
            camara = Camara(0)
            try:
                self.assertTrue(esperar(lambda: camara.reconexiones == 1))
                self.assertFalse(CapturaFalsa.abiertas[0].abierta)  # la vieja se liberó
                self.assertTrue(esperar(lambda: camara.leer() is not None))
            finally:
                camara.liberar()

    def test_liberar_detiene_y_suelta_la_camara(self):
        with mock.patch.object(modulo_camara.cv2, "VideoCapture", CapturaFalsa):
            camara = Camara(0)
            camara.liberar()
            self.assertFalse(camara._hilo.is_alive())
            self.assertFalse(CapturaFalsa.abiertas[0].abierta)