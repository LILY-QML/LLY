# lly/ml/ml.py

import os
# Deaktiviere die GPU-Nutzung, um CUDA-Fehler zu vermeiden
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# ----------------------------------------------------------------------------
# ThreadMonitor Klasse
# ----------------------------------------------------------------------------

class ThreadMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Singleton-Pattern, um sicherzustellen, dass nur eine Instanz des Monitors existiert.
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ThreadMonitor, cls).__new__(cls)
                    cls._instance.thread_data = []
                    cls._instance.lock = threading.Lock()
        return cls._instance

    def log_thread_start(self, thread_id, process_name):
        """
        Protokolliert den Start eines Threads.

        :param thread_id: Die eindeutige ID des Threads.
        :param process_name: Der Name des Prozesses (z. B. Optimierung eines Qubits).
        """
        start_time = time.time()
        with self.lock:
            self.thread_data.append({
                "thread_id": thread_id,
                "process_name": process_name,
                "start_time": start_time,
                "end_time": None,
                "duration": None
            })

    def log_thread_end(self, thread_id):
        """
        Protokolliert das Ende eines Threads.

        :param thread_id: Die eindeutige ID des Threads.
        """
        end_time = time.time()
        with self.lock:
            for entry in self.thread_data:
                if entry["thread_id"] == thread_id and entry["end_time"] is None:
                    entry["end_time"] = end_time
                    entry["duration"] = end_time - entry["start_time"]
                    break

    def generate_pdf_report(self, folder="log", filename="Thread.pdf"):
        """
        Erstellt einen PDF-Bericht basierend auf den gesammelten Thread-Daten.

        :param folder: Der Ordner, in dem die PDF gespeichert wird.
        :param filename: Der Dateiname der generierten PDF-Datei.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, filename)
        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4

        # Titel
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Thread Activity Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Tabelle mit Thread-Daten
        c.drawString(50, height - 100, "Thread Overview:")
        y = height - 120
        c.setFont("Helvetica", 10)

        headers = ["Thread ID", "Process Name", "Start Time", "End Time", "Duration (s)"]
        c.drawString(50, y, " | ".join(headers))
        y -= 20

        for entry in self.thread_data:
            start_time = time.strftime('%H:%M:%S', time.localtime(entry["start_time"]))
            end_time = time.strftime('%H:%M:%S', time.localtime(entry["end_time"])) if entry["end_time"] else "N/A"
            duration = f"{entry['duration']:.2f}" if entry["duration"] else "N/A"

            line = f"{entry['thread_id']} | {entry['process_name']} | {start_time} | {end_time} | {duration}"
            c.drawString(50, y, line)
            y -= 15

            if y < 50:  # Neue Seite bei Platzmangel
                c.showPage()
                y = height - 50

        c.save()

    def __str__(self):
        """
        Gibt eine Übersicht der gesammelten Thread-Daten als String zurück.
        """
        overview = ["Thread Overview:"]
        for entry in self.thread_data:
            overview.append(
                f"Thread ID: {entry['thread_id']}, Process: {entry['process_name']}, "
                f"Start: {time.strftime('%H:%M:%S', time.localtime(entry['start_time']))}, "
                f"End: {time.strftime('%H:%M:%S', time.localtime(entry['end_time'])) if entry['end_time'] else 'N/A'}, "
                f"Duration: {entry['duration']:.2f}s" if entry['duration'] else "N/A"
            )
        return "\n".join(overview)

# ----------------------------------------------------------------------------
# ML Klasse mit ThreadMonitor Integration
# ----------------------------------------------------------------------------

class ML:
    def __init__(self, qubits, optimizer_type='adam', loss_function_type='binary_cross_entropy', params=None, target_state=None):
        """
        Initialisiert die ML-Klasse.

        :param qubits: Anzahl der Qubits
        :param optimizer_type: Typ des Optimierers ('adam', 'sgd', 'rmsprop', 'adagrad')
        :param loss_function_type: Typ der Verlustfunktion ('binary_cross_entropy', 'mean_squared_error', 'hinge', 'kl_divergence', 'huber')
        :param params: Dictionary mit zusätzlichen Parametern für den Optimierer
        :param target_state: Liste der Zielzustände für jedes Qubit (0 oder 1)
        """
        self.qubits = qubits
        self.target_state = target_state if target_state else [1] * qubits  # Standard: Alle Qubits auf 1
        self.params = params if params else {}
        self.logger = self.setup_logger()
        self.logger.info("ML-Klasse initialisiert.")
        self.optimizer_type = optimizer_type
        self.loss_function_type = loss_function_type
        self.loss_function = self.get_loss_function(loss_function_type)

        # Initialisiere ThreadMonitor Instanz
        self.thread_monitor = ThreadMonitor()

    def setup_logger(self):
        """
        Einrichtung eines einfachen Loggers.
        """
        logger = logging.getLogger('MLLogger')
        logger.setLevel(logging.INFO)
        # Verhindert doppelte Logs
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def get_optimizer(self, optimizer_type, params):
        """
        Gibt eine Instanz des gewünschten Optimierers zurück.

        :param optimizer_type: Typ des Optimierers
        :param params: Parameter für den Optimierer
        :return: TensorFlow Optimizer Instanz
        """
        if optimizer_type.lower() == 'adam':
            learning_rate = params.get('learning_rate', 0.001)
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            learning_rate = params.get('learning_rate', 0.01)
            momentum = params.get('momentum', 0.0)
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer_type.lower() == 'rmsprop':
            learning_rate = params.get('learning_rate', 0.001)
            rho = params.get('rho', 0.9)
            epsilon = params.get('epsilon', 1e-07)
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon)
        elif optimizer_type.lower() == 'adagrad':
            learning_rate = params.get('learning_rate', 0.01)
            initial_accumulator_value = params.get('initial_accumulator_value', 0.1)
            return tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=initial_accumulator_value)
        else:
            raise ValueError(f"Unbekannter Optimizer-Typ: {optimizer_type}")

    def get_loss_function(self, loss_function_type):
        """
        Gibt eine Instanz der gewünschten Verlustfunktion zurück.

        :param loss_function_type: Typ der Verlustfunktion
        :return: TensorFlow Verlustfunktion
        """
        if loss_function_type.lower() == 'binary_cross_entropy':
            return tf.keras.losses.BinaryCrossentropy()
        elif loss_function_type.lower() == 'mean_squared_error':
            return tf.keras.losses.MeanSquaredError()
        elif loss_function_type.lower() == 'hinge':
            return tf.keras.losses.Hinge()
        elif loss_function_type.lower() == 'kl_divergence':
            return tf.keras.losses.KLDivergence()
        elif loss_function_type.lower() == 'huber':
            delta = self.params.get('delta', 1.0)
            return tf.keras.losses.Huber(delta=delta)
        else:
            raise ValueError(f"Unbekannte Verlustfunktion: {loss_function_type}")

    def encode_measurements(self, measurement):
        """
        Encodiert die Messungsergebnisse in eine strukturierte Form für jedes Qubit.

        :param measurement: Dictionary mit Messungsergebnissen, z.B. {"1101": 221, "111": 212}
        :return: Liste der encodierten Messzustände pro Qubit
        """
        qubits_measurement_count = np.zeros((self.qubits, 2), dtype=int)

        self.logger.info("Starte Encodierung der Messungen.")

        # Überprüfen der Konsistenz der Daten
        if not measurement:
            self.logger.error("Leere Messdaten erhalten.")
            return None

        first_key = next(iter(measurement))
        if len(first_key) != self.qubits:
            error = {"Error Code": 1073, "Message": "Inkonsistente Daten aufgrund falscher Anzahl von Qubits."}
            self.logger.error(error)
            return None

        # Zählen der Zustände für jedes Qubit
        for key, value in measurement.items():
            if len(key) != self.qubits:
                self.logger.warning(f"Überspringe inkonsistente Messung: {key}")
                continue
            for index, c in enumerate(key):
                if c not in ['0', '1']:
                    self.logger.warning(f"Ungültiger Qubit-Zustand '{c}' bei Position {index} in Schlüssel '{key}'. Überspringe.")
                    continue
                qubits_measurement_count[index][int(c)] += value

        # Erstellen der encodierten Messzustände
        qubits_measurement = []
        for i in range(self.qubits):
            qubit_measurement = f"(1:{qubits_measurement_count[i][1]}; 0:{qubits_measurement_count[i][0]})"
            qubits_measurement.append(qubit_measurement)

        self.logger.info(f"Encodierte Messungen: {qubits_measurement}")

        return qubits_measurement

    def run_qubit_optimization(self, qubit_index, qubit_phases, qubit_target):
        """
        Führt die Optimierung für ein einzelnes Qubit durch.

        :param qubit_index: Index des Qubits
        :param qubit_phases: Initiale Phasen für das Qubit
        :param qubit_target: Zielzustand für das Qubit (0 oder 1)
        :return: Optimierte Phasen für das Qubit
        """
        # Logge den Start der Optimierung für dieses Qubit
        thread_id = threading.get_ident()
        self.thread_monitor.log_thread_start(thread_id, f"Optimierung Qubit {qubit_index}")

        try:
            # Erstelle eine separate Optimierer-Instanz für dieses Qubit
            optimizer = self.get_optimizer(self.optimizer_type, self.params)

            # Berechne Ziel-Wahrscheinlichkeit basierend auf dem Zielzustand
            if qubit_target == 1:
                target_probability = 1.0
            else:
                target_probability = 0.0

            # Initialisiere die Phasen als Tensor
            phases = tf.Variable(qubit_phases, dtype=tf.float32)

            for epoch in range(1000):
                with tf.GradientTape() as tape:
                    # Berechne die Wahrscheinlichkeiten mittels Sigmoid-Funktion
                    probabilities = tf.math.sigmoid(phases)
                    # Erstelle einen Ziel-Tensor mit derselben Form wie probabilities
                    target_tensor = tf.fill(tf.shape(probabilities), target_probability)
                    # Berechne den Verlust
                    loss = self.loss_function(target_tensor, probabilities)
                # Berechne die Gradienten
                gradients = tape.gradient(loss, [phases])
                # Wende die Gradienten an
                optimizer.apply_gradients(zip(gradients, [phases]))
                # Optional: Abbruchkriterium
                if loss.numpy() < 1e-4:
                    break

            return phases.numpy()
        finally:
            # Logge das Ende der Optimierung für dieses Qubit
            self.thread_monitor.log_thread_end(thread_id)

    def run(self, measurement, phases):
        """
        Führt den Optimierungsprozess aus, um die Phasenmatrix basierend auf den Messungsergebnissen anzupassen.

        :param measurement: Dictionary mit Messungsergebnissen, z.B. {"1101": 221, "111": 212}
        :param phases: n x m Matrix mit initialen Phasenwerten (n Qubits, m Phasen)
        :return: Optimierte Phasenmatrix
        """
        self.logger.info("Starte Optimierungsprozess.")

        # Encodieren der Messungen
        encoded_measurements = self.encode_measurements(measurement)
        if encoded_measurements is None:
            self.logger.error("Encodierung der Messungen fehlgeschlagen. Abbruch des Optimierungsprozesses.")
            return None

        # Zählen der Zustände für jedes Qubit
        qubits_measurement_count = np.zeros((self.qubits, 2), dtype=int)
        for key, value in measurement.items():
            if len(key) != self.qubits:
                continue
            for index, c in enumerate(key):
                if c not in ['0', '1']:
                    continue
                qubits_measurement_count[index][int(c)] += value

        # Optimierte Phasenmatrix initialisieren
        optimized_phases = phases.copy()

        # Verwendung von ThreadPoolExecutor zur Parallelisierung der Optimierung
        with ThreadPoolExecutor(max_workers=self.qubits) as executor:
            futures = []
            for qubit_index in range(self.qubits):
                qubit_phases = optimized_phases[qubit_index]
                qubit_target = self.target_state[qubit_index]
                futures.append(
                    executor.submit(
                        self.run_qubit_optimization,
                        qubit_index,
                        qubit_phases,
                        qubit_target
                    )
                )
            # Sammeln der Ergebnisse
            for qubit_index, future in enumerate(futures):
                try:
                    optimized_qubit_phases = future.result()
                    optimized_phases[qubit_index] = optimized_qubit_phases
                    self.logger.info(f"Optimierung abgeschlossen für Qubit {qubit_index}.")
                except Exception as e:
                    self.logger.error(f"Fehler bei der Optimierung für Qubit {qubit_index}: {e}")

        self.logger.info("Optimierungsprozess abgeschlossen.")
        return optimized_phases

    # ----------------------------------------------------------------------------
    # Beispielnutzung (optional, falls benötigt)
    # ----------------------------------------------------------------------------
    if __name__ == "__main__":
        # Initialisiere ThreadMonitor (wird automatisch durch ML-Klasse gemacht)
        # thread_monitor = ThreadMonitor()

        # Beispiel-Messdaten und Phasenmatrix
        measurement = {
            "1101": 221,
            "1110": 212,
            "1001": 150,
            "1011": 180,
            "0110": 200,
            "0101": 190
        }

        initial_phases = np.random.uniform(0, 2*np.pi, (4, 6))  # Beispiel für 4 Qubits mit 6 Phasen

        # Erstelle eine ML-Instanz mit 4 Qubits
        ml = ML(qubits=4, optimizer_type='adam', loss_function_type='binary_cross_entropy')

        # Führe die Optimierung durch
        optimized_phases = ml.run(measurement, initial_phases)

        # Ausgabe der optimierten Phasen
        print("Optimierte Phasenmatrix:")
        print(optimized_phases)

        # Generiere einen PDF-Bericht der Thread-Aktivitäten
        ml.thread_monitor.generate_pdf_report()

        print("\nThread Activity Report generated as 'log/Thread.pdf'.")

        # Optional: Ausgabe der Thread-Daten im Terminal
        print("\nCurrent Thread Activity Overview:")
        print(ml.thread_monitor)
