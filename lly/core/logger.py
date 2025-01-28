# lly/core/logger.py

import logging
import os
from datetime import datetime

# Erstelle das Log-Verzeichnis, falls es nicht existiert
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log")
os.makedirs(LOG_DIR, exist_ok=True)

# Erstelle einen eindeutigen Log-Dateinamen mit Datum und Uhrzeit
log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
log_path = os.path.join(LOG_DIR, log_filename)

# Konfiguriere den Logger
logger = logging.getLogger("ProjektLogger")
logger.setLevel(logging.DEBUG)

# Überprüfe, ob der Logger bereits Handler hat, um doppelte Logs zu vermeiden
if not logger.handlers:
    # Erstelle einen File Handler für die Log-Datei
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Erstelle einen Stream Handler für die Konsole (optional)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Erstelle einen Formatter und setze ihn für beide Handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Füge die Handler dem Logger hinzu
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)