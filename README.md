# Sistema de Reconocimiento Facial para Asistencias

Proyecto de la materia Construcción de Software de la UNEMI. Es una página web hecha en Django que usa la cámara para reconocer a las personas y registrar su asistencia en PostgreSQL.

Integrantes:
- Lino Diaz Isaac Giovanny
- Chancay Rosel Angel Ernesto
- Keni Daniel Gómez Méndez
- Camilo Andres Rosero Mera

## Qué hace

- Muestra la cámara en vivo en el navegador.
- Detecta los rostros y los reconoce con OpenCV.
- Cuando reconoce a alguien, guarda su asistencia del día, solo una vez por día.
- Marca si la persona llegó puntual o atrasada según la hora límite.
- Permite registrar personas nuevas desde la cámara.
- Tiene un historial con filtros y se puede exportar a Excel en formato CSV.

Para la detección usamos el modelo YuNet y para el reconocimiento el modelo SFace, los dos vienen incluidos en OpenCV. Antes de detectar se mejora la iluminación de la imagen para que funcione mejor con poca luz.

## Tecnologías

- Python y Django
- OpenCV
- PostgreSQL

## Cómo ejecutarlo

1. Instalar PostgreSQL y crear la base de datos:

```sql
CREATE DATABASE asistencia_facial;
```

2. Clonar el repositorio y crear el entorno virtual:

```bash
git clone https://github.com/rosero2019/Sistema-de-Reconocimiento-Facial-para-Asistencias.git
cd Sistema-de-Reconocimiento-Facial-para-Asistencias
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Descargar los modelos de OpenCV:

```bash
python descargar_modelos.py
```

4. Hacer una copia de .env.example con el nombre .env y poner la contraseña de PostgreSQL en DB_PASSWORD.

5. Crear las tablas y levantar el servidor:

```bash
python manage.py migrate
python manage.py runserver
```

6. Abrir http://127.0.0.1:8000 en el navegador.

## Cómo se usa

Primero se registra a la persona en la sección Personas: se llenan sus datos y se mira a la cámara unos segundos mientras se toman las muestras del rostro. Después, en el Panel, cuando la persona aparece frente a la cámara se le marca la asistencia. El recuadro se pone amarillo mientras la está reconociendo y verde cuando ya se guardó.

## Pruebas

```bash
python manage.py test
```

## Carpetas principales

- config: configuración de Django.
- asistencia: la aplicación con los modelos, vistas y plantillas.
- asistencia/vision: todo lo de OpenCV, es decir la cámara, la detección y el reconocimiento.
- modelos: aquí se guardan los modelos descargados.

## Nota

El sistema no guarda fotos, solo los datos numéricos de cada rostro. Igual se debe pedir permiso a las personas antes de registrarlas.
