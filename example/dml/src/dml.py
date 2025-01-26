# example/dml/src/dml.py

import json
import os
import logging
import numpy as np
from lly.ml.ml import ML  # Importiere die ML-Klasse

class DML:
    def __init__(self, data_path):
        """
        Initialisiert die DML-Klasse.

        :param data_path: Pfad zur data.json Datei
        """
        self.data_path = data_path
        self.activation_matrices = []
        self.qubits = 0
        self.depth = 0
        self.logger = self.setup_logger()
        self.load_data()
        self.check()  # Automatisches Ausführen der Überprüfung beim Erzeugen der Klasse

    def setup_logger(self):
        """
        Einrichtung eines einfachen Loggers.
        """
        logger = logging.getLogger('DMLLogger')
        logger.setLevel(logging.INFO)
        # Verhindert doppelte Logs
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def load_data(self):
        """
        Lädt die Aktivierungsmatrizen aus der JSON-Datei.
        """
        if not os.path.exists(self.data_path):
            self.logger.error(f"Datei nicht gefunden: {self.data_path}")
            return

        with open(self.data_path, 'r') as file:
            try:
                data = json.load(file)
                self.activation_matrices = data.get("activation_matrices", [])
                self.logger.info(f"{len(self.activation_matrices)} Aktivierungsmatrizen geladen.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Fehler beim Parsen der JSON-Datei: {e}")

    def check(self):
        """
        Führt die Überprüfungen auf den Aktivierungsmatrizen durch.

        - Überprüft, ob alle Aktivierungsmatrizen gleich sind.
        - Überprüft, ob die Anzahl der Spalten jeder Matrix durch 3 teilbar ist.
        - Setzt die Werte für `qubits` und `depth`.

        :return: Dictionary mit Ergebnissen der Überprüfungen
        """
        if not self.activation_matrices:
            self.logger.error("Keine Aktivierungsmatrizen zum Überprüfen.")
            return {"status": "Fehler", "nachricht": "Keine Aktivierungsmatrizen vorhanden."}

        # Überprüfen, ob alle Matrizen gleich sind
        first_matrix = self.activation_matrices[0]["matrix"]
        all_equal = all(matrix["matrix"] == first_matrix for matrix in self.activation_matrices)
        self.logger.info(f"Alle Matrizen gleich: {all_equal}")

        # Überprüfen, ob die Anzahl der Spalten durch 3 teilbar ist
        num_qubits = len(first_matrix)  # Anzahl der Reihen
        for idx, matrix in enumerate(self.activation_matrices):
            num_columns = len(matrix["matrix"][0])  # Anzahl der Spalten
            if num_columns % 3 != 0:
                self.logger.warning(f"Matrize {idx} hat {num_columns} Spalten, die nicht durch 3 teilbar sind.")
                return {
                    "status": "Fehler",
                    "nachricht": f"Matrize {idx} hat {num_columns} Spalten, die nicht durch 3 teilbar sind."
                }

        # Setzen der Werte für `qubits` und `depth`
        self.qubits = num_qubits  # Anzahl der Reihen
        self.depth = len(first_matrix[0]) // 3  # Anzahl der Spalten geteilt durch 3

        self.logger.info(f"Qubits gesetzt auf: {self.qubits}")
        self.logger.info(f"Depth gesetzt auf: {self.depth}")

        return {
            "status": "Erfolg",
            "alle_matrizen_gleich": all_equal,
            "qubits": self.qubits,
            "depth": self.depth
        }

    def create(self):
        """
        Erstellt einen Circuit mit den gegebenen Parametern und einer randomisierten Trainings-Phasengatter-Matrix.

        - Importiert Circuit aus lly.core.circuit
        - Erstellt einen Circuit mit qubits und depth
        - Setzt die Aktivierungsphasen auf die erste Aktivierungsmatrix
        - Erstellt eine randomisierte Trainings-Phasengatter-Matrix (gleiche Dimensionen wie activation_phases)
        - Führt den Circuit aus und gibt ihn zusammen mit der Trainings-Phasengatter-Matrix zurück

        :return: Tuple (Circuit, training_phases)
        """
        try:
            from lly.core.circuit import Circuit
        except ImportError as e:
            self.logger.error(f"Fehler beim Importieren der Circuit-Klasse: {e}")
            return None, None

        if not self.activation_matrices:
            self.logger.error("Keine Aktivierungsmatrizen zum Erstellen eines Circuit vorhanden.")
            return None, None

        # Setze die Aktivierungsphasen auf die erste Aktivierungsmatrix
        activation_phases = self.activation_matrices[0]["matrix"]  # qubits x (depth *3)

        # Generiere eine randomisierte Trainings-Phasengatter-Matrix mit gleichen Dimensionen wie activation_phases
        training_phases = [
            [float(np.random.rand()) for _ in range(self.depth * 3)] for _ in range(self.qubits)
        ]

        self.logger.info(f"Generierte Trainings-Phasengatter-Matrix: {training_phases}")

        # Erstelle einen Circuit mit qubits und depth
        circuit = Circuit(
            qubits=self.qubits,
            depth=self.depth,
            training_phases=training_phases,
            activation_phases=activation_phases,
            shots=10000
        )

        # Führe den Circuit aus
        circuit.measure_all()
        circuit.run()
        counts = circuit.get_counts()

        self.logger.info(f"Circuit ausgeführt. Messresultate: {counts}")

        return circuit, training_phases

    def optimize(self, optimizer, optimized_param, shots, iterations, end_value=None):
        """
        Führt den Optimierungsprozess aus, um die Phasenmatrix basierend auf den Messungsergebnissen anzupassen.

        :param optimizer: Typ des Optimierers (z.B. 'adam', 'sgd')
        :type optimizer: str
        :param optimized_param: Parameter für den Optimierer als Dictionary
        :type optimized_param: dict
        :param shots: Anzahl der Shots für die Simulation
        :type shots: int
        :param iterations: Maximale Anzahl an Iterationen für die Optimierung
        :type iterations: int
        :param end_value: Optionaler Schwellenwert für die Wahrscheinlichkeit des Zielzustands
        :type end_value: float, optional
        :return: None
        """
        try:
            from lly.core.circuit import Circuit
        except ImportError as e:
            self.logger.error(f"Fehler beim Importieren der Circuit-Klasse: {e}")
            return

        for idx, activation_matrix in enumerate(self.activation_matrices):
            self.logger.info(f"Starte Optimierung für Matrize {idx} ({activation_matrix.get('name', 'Unbenannt')}).")

            target_state = activation_matrix.get('target_state', [1] * self.qubits)  # Standardziel: Alle Qubits auf 1
            activation_phases = activation_matrix["matrix"]  # qubits x (depth *3)

            # Initiale Trainings-Phasengatter-Matrix
            training_phases = [
                [float(np.random.rand()) for _ in range(self.depth * 3)] for _ in range(self.qubits)
            ]

            # Erstelle einen Circuit mit den aktuellen Phasen
            circuit = Circuit(
                qubits=self.qubits,
                depth=self.depth,
                training_phases=training_phases,
                activation_phases=activation_phases,
                shots=shots
            )

            # Initialisiere den Optimierer
            ml = ML(
                qubits=self.qubits,
                optimizer_type=optimizer,
                params=optimized_param,
                target_state=target_state
            )

            for iteration in range(iterations):
                self.logger.info(f"Iteration {iteration + 1}/{iterations} für Matrize {idx}.")

                # Führe den Circuit aus und erhalte Messresultate
                circuit.measure_all()
                circuit.run()
                counts = circuit.get_counts()

                # Berechne die Wahrscheinlichkeit des Zielzustands
                target_key = ''.join(str(bit) for bit in target_state)
                target_count = counts.get(target_key, 0)
                total_counts = sum(counts.values())
                probability = target_count / total_counts if total_counts > 0 else 0
                self.logger.info(f"Wahrscheinlichkeit des Zielzustands {target_key}: {probability:.4f}")

                # Prüfe, ob der Schwellenwert erreicht wurde
                if end_value is not None and probability >= end_value:
                    self.logger.info(f"Schwellenwert erreicht: {probability:.4f} >= {end_value}")
                    break

                # Führe die Optimierung für dieses Qubit durch
                optimized_phases = ml.run(counts, training_phases)

                # Aktualisiere die Trainingsphasen
                training_phases = optimized_phases

                # Aktualisiere den Circuit mit den neuen Phasen
                circuit = Circuit(
                    qubits=self.qubits,
                    depth=self.depth,
                    training_phases=training_phases,
                    activation_phases=activation_phases,
                    shots=shots
                )

            # Speichere die optimierten Phasen zurück in die Aktivierungsmatrix
            activation_matrix["training_phases"] = training_phases

            self.logger.info(f"Optimierung abgeschlossen für Matrize {idx}.")

        # Speichere die aktualisierten Daten zurück in die JSON-Datei
        self.save_data()

    def save_data(self):
        """
        Speichert die aktuellen Aktivierungsmatrizen zurück in die JSON-Datei.
        """
        data = {
            "activation_matrices": self.activation_matrices
        }
        try:
            with open(self.data_path, 'w') as file:
                json.dump(data, file, indent=4)
            self.logger.info(f"Aktualisierte Daten erfolgreich in {self.data_path} gespeichert.")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Daten: {e}")
