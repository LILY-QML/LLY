import json
import os
import logging
import numpy as np
import time
from collections import defaultdict
from lly.ml.ml import ML  
from lly.visualization.visual import Visual
from lly.core.thread import ThreadMonitor  # Importiere ThreadMonitor
import threading  # Importiere threading, um threading.get_ident() zu verwenden

class DML:
    def __init__(self, data_path):
        """
        Initializes the DML class.

        :param data_path: Path to the data.json file
        """
        self.data_path = data_path
        self.activation_matrices = []
        self.qubits = 0
        self.depth = 0
        self.logger = self.setup_logger()
        self.load_data()
        self.check()  # Automatically execute the check upon class instantiation
        self.training_phases = None  # Initialisierung

        # Initialize the Visual instance
        self.visual = Visual()

        # Initialize ThreadMonitor instance
        self.thread_monitor = ThreadMonitor()

        # Initialize Circuit as None
        self.circuit = None

    def setup_logger(self):
        """
        Sets up a simple logger.
        """
        logger = logging.getLogger('DMLLogger')
        logger.setLevel(logging.INFO)
        # Prevent duplicate logs
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def load_data(self):
        """
        Loads activation matrices from the JSON file.
        """
        if not os.path.exists(self.data_path):
            self.logger.error(f"File not found: {self.data_path}")
            return

        with open(self.data_path, 'r') as file:
            try:
                data = json.load(file)
                self.activation_matrices = data.get("activation_matrices", [])
                self.logger.info(f"{len(self.activation_matrices)} activation matrices loaded.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON file: {e}")

    def check(self):
        """
        Performs checks on the activation matrices.

        - Checks if all activation matrices are identical.
        - Checks if the number of columns in each matrix is divisible by 3.
        - Sets the values for `qubits` and `depth`.

        :return: Dictionary with the results of the checks
        """
        if not self.activation_matrices:
            self.logger.error("No activation matrices to check.")
            return {"status": "Error", "message": "No activation matrices available."}

        # Check if all matrices are identical
        first_matrix = self.activation_matrices[0]["matrix"]
        all_equal = all(matrix["matrix"] == first_matrix for matrix in self.activation_matrices)
        self.logger.info(f"All matrices identical: {all_equal}")

        # Check if the number of columns is divisible by 3
        num_qubits = len(first_matrix)  # Number of rows
        for idx, matrix in enumerate(self.activation_matrices):
            num_columns = len(matrix["matrix"][0])  # Number of columns
            if num_columns % 3 != 0:
                self.logger.warning(f"Matrix {idx} has {num_columns} columns, not divisible by 3.")
                return {
                    "status": "Error",
                    "message": f"Matrix {idx} has {num_columns} columns, not divisible by 3."
                }

        # Set the values for `qubits` and `depth`
        self.qubits = num_qubits  # Number of rows
        self.depth = len(first_matrix[0]) // 3  # Number of columns divided by 3

        self.logger.info(f"Qubits set to: {self.qubits}")
        self.logger.info(f"Depth set to: {self.depth}")

        return {
            "status": "Success",
            "all_matrices_identical": all_equal,
            "qubits": self.qubits,
            "depth": self.depth
        }

    def create(self):
        """
        Creates a circuit with the given parameters and a randomized training phase gate matrix.

        :return: Tuple (Circuit, training_phases)
        """
        try:
            from lly.core.circuit import Circuit
        except ImportError as e:
            self.logger.error(f"Error importing Circuit class: {e}")
            return None, None

        if not self.activation_matrices:
            self.logger.error("No activation matrices available to create a Circuit.")
            return None, None

        activation_phases = self.activation_matrices[0]["matrix"]

        training_phases = [
            [float(np.random.rand()) for _ in range(self.depth * 3)] for _ in range(self.qubits)
        ]

        self.logger.info(f"Generated training phase gate matrix: {training_phases}")

        # Zuweisung der Trainingsphasen zur Instanzvariable
        self.training_phases = training_phases

        # Speicherung des Circuits als Attribut der Klasse
        self.circuit = Circuit(
            qubits=self.qubits,
            depth=self.depth,
            training_phases=self.training_phases,
            activation_phases=activation_phases,
            shots=10000
        )

        self.circuit.measure_all()
        self.circuit.run()
        counts = self.circuit.get_counts()

        self.logger.info(f"Circuit executed. Measurement results: {counts}")

        # Setze die Aktivierungsphasen separat
        self.visual.set_activation_phases("Matrix1", activation_phases)

        # Setze die initialen Verteilungen (counts)
        self.visual.set_initial_distribution("Matrix1", counts)

        self.visual.set_final_distribution("Matrix1", "InitialCircuit", counts)

        return self.circuit, self.training_phases

    def optimize(self, optimizer, optimized_param, shots, iterations, end_value=None):
        """
        Führt den Optimierungsprozess durch, um die Phasenmatrix basierend auf den Messresultaten anzupassen.
        """
        try:
            from lly.core.circuit import Circuit
        except ImportError as e:
            self.logger.error(f"Error importing Circuit class: {e}")
            return

        for idx, activation_matrix in enumerate(self.activation_matrices):
            self.logger.info(f"Starting optimization for Matrix {idx} ({activation_matrix.get('name', 'Unnamed')}).")

            target_state = activation_matrix.get('target_state', [1] * self.qubits)
            activation_phases = activation_matrix["matrix"]

            # Sicherstellen, dass self.training_phases gesetzt ist
            if self.training_phases is None:
                self.logger.error("Training phases sind nicht gesetzt. Bitte führe zuerst die `create`-Methode aus.")
                return

            training_phases = self.training_phases.copy()

            ml = ML(
                qubits=self.qubits,
                optimizer_type=optimizer,
                params=optimized_param,
                target_state=target_state
            )

            self.visual.set_initial_distribution(f"Matrix{idx}_Optimizer{optimizer}", self.circuit.get_counts())

            thread_id = threading.get_ident()
            self.thread_monitor.log_thread_start(thread_id, f"Optimization for Matrix {idx}")

            try:
                for iteration in range(iterations):
                    self.logger.info(f"Iteration {iteration + 1}/{iterations} für Matrix {idx}.")

                    # Verwenden des gespeicherten Circuit-Objekts
                    self.circuit.measure_all()
                    self.circuit.run()
                    counts = self.circuit.get_counts()

                    target_key = ''.join(str(bit) for bit in target_state)
                    target_count = counts.get(target_key, 0)
                    total_counts = sum(counts.values())
                    probability = target_count / total_counts if total_counts > 0 else 0
                    self.logger.info(f"Wahrscheinlichkeit des Zielzustands {target_key}: {probability:.4f}")

                    loss = 1 - probability
                    self.logger.info(f"Verlust: {loss:.4f}")

                    self.visual.record_probability(f"Matrix{idx}_Optimizer{optimizer}", "Optimizer", iteration + 1, probability)
                    self.visual.record_loss(f"Matrix{idx}_Optimizer{optimizer}", "Optimizer", iteration + 1, loss)

                    if end_value is not None and probability >= end_value:
                        self.logger.info(f"Schwellenwert erreicht: {probability:.4f} >= {end_value}")
                        break

                    optimized_phases = ml.run(counts, training_phases)

                    if optimized_phases is None:
                        self.logger.error("Optimizer returned None. Abbruch der Optimierung.")
                        break

                    training_phases = optimized_phases

                    self.logger.info(f"Iteration {iteration + 1}: Aktualisierte Trainingsphasen: {training_phases}")

                    # Aktualisieren des gespeicherten Trainingsphases
                    self.training_phases = training_phases

                    # Aktualisieren des Circuits mit den neuen Trainingsphasen
                    self.circuit = Circuit(
                        qubits=self.qubits,
                        depth=self.depth,
                        training_phases=self.training_phases,
                        activation_phases=activation_phases,
                        shots=shots
                    )
            finally:
                self.thread_monitor.log_thread_end(thread_id)

            self.logger.info(f"Optimierung abgeschlossen für Matrix {idx}.")

            self.circuit.measure_all()
            self.circuit.run()
            final_counts = self.circuit.get_counts()

            self.visual.set_final_distribution(f"Matrix{idx}_Optimizer{optimizer}", "Optimizer", final_counts)

            final_matrix = np.array(training_phases).reshape(self.qubits, self.depth, 3)
            self.visual.record_heatmap_data(f"Matrix{idx}_Optimizer{optimizer}", "Optimizer", final_matrix)

        self.visual.generate_pdf("ThreadActivityReport.pdf")

    def measure(self, training_matrix, shots, activation_matrix):
        """
        Measures the training matrix with the given activation matrix.

        :param training_matrix: The training matrix to be measured
        :type training_matrix: list[list[float]]
        :param shots: Number of shots for the measurement
        :type shots: int
        :param activation_matrix: The activation matrix to be used for measurement
        :type activation_matrix: list[list[float]]
        :return: Measurement results as a dictionary
        :rtype: dict
        """
        try:
            from lly.core.circuit import Circuit
        except ImportError as e:
            self.logger.error(f"Error importing Circuit class: {e}")
            return None

        if len(training_matrix) != len(activation_matrix) or len(training_matrix[0]) != len(activation_matrix[0]):
            self.logger.error("The dimensions of the training matrix and the activation matrix do not match.")
            return None

        self.logger.info(f"Starting measurement with {shots} shots.")
        self.logger.info(f"Training matrix: {training_matrix}")
        self.logger.info(f"Activation matrix: {activation_matrix}")

        # Create the Circuit
        self.circuit = Circuit(
            qubits=self.qubits,
            depth=self.depth,
            training_phases=training_matrix,
            activation_phases=activation_matrix,
            shots=shots
        )

        self.circuit.measure_all()
        self.circuit.run()

        counts = self.circuit.get_counts()
        self.logger.info(f"Measurement results: {counts}")

        target_state = [1] * self.qubits  # Define your target state appropriately
        target_key = ''.join(str(bit) for bit in target_state)
        target_count = counts.get(target_key, 0)
        total_counts = sum(counts.values())
        probability = target_count / total_counts if total_counts > 0 else 0
        loss = 1 - probability

        self.visual.record_probability("Measurement", "Measurement", 1, probability)
        self.visual.record_loss("Measurement", "Measurement", 1, loss)

        # Record phase values (assuming phase values are part of the measurement)
        for qubit_idx in range(self.qubits):
            for depth_idx in range(self.depth * 3):
                phase = training_matrix[qubit_idx][depth_idx]
                self.visual.record_phase_value("Measurement", "Measurement", phase)

        self.visual.set_final_distribution("Measurement", "Measurement", counts)

        # Record heatmap data
        final_matrix = np.array(training_matrix).reshape(self.qubits, self.depth, 3)
        self.visual.record_heatmap_data("Measurement", "Measurement", final_matrix)

        # Optionally, generate the PDF here if measurements are standalone
        # self.visual.generate_pdf("00")

        return counts

    def save_training_matrix(self, training_matrix, folder_path):
        """
        Saves the current training matrix to a specified folder.

        :param training_matrix: The training matrix to be saved
        :type training_matrix: list[list[float]]
        :param folder_path: The folder where the file will be saved
        :type folder_path: str
        """
        import datetime
        import json

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generate a unique filename with date and time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"training_matrix_{timestamp}.json"
        file_path = os.path.join(folder_path, file_name)

        # Save the training matrix as a JSON file
        try:
            with open(file_path, 'w') as file:
                json.dump(training_matrix, file, indent=4)
            self.logger.info(f"Training matrix successfully saved at: {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving training matrix: {e}")

# ----------------------------------------------------------------------------
# Beispielnutzung (optional, falls benötigt)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Beispiel-Pfad zur data.json Datei
    data_path = "example/dml/var/data.json"

    # Erstelle eine DML-Instanz
    dml = DML(data_path=data_path)

    # Erstelle einen Circuit und Trainingsphasen
    circuit, training_phases = dml.create()

    if circuit is not None:
        # Beispiel-Messdaten nach der Erstellung des Circuits
        measurement = circuit.get_counts()

        # Speichere die Trainingsmatrix
        dml.save_training_matrix(training_phases, folder_path="example/dml/saved_matrices")

        # Führe die Optimierung durch
        dml.optimize(
            optimizer='adam',
            optimized_param={'learning_rate': 0.001},
            shots=10000,
            iterations=100,
            end_value=0.99
        )

        # Führe eine zusätzliche Messung durch (optional)
        # counts = dml.measure(training_matrix=training_phases, shots=10000, activation_matrix=dml.activation_matrices[0]["matrix"])

    # Generiere einen PDF-Bericht der Thread-Aktivitäten (falls benötigt)
    # dml.thread_monitor.generate_pdf_report()

    # Optional: Ausgabe der Thread-Daten im Terminal
    # print("\nCurrent Thread Activity Overview:")
    # print(dml.thread_monitor)
