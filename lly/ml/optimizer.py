# optimizer.py

import os
import json
import logging
import datetime
import numpy as np
import tensorflow as tf

from utils.qubit import Qubit

# Mapping von string "XYOptimizer" zu Keras-Optimizer-Klasse
_OPTIMIZER_MAPPING = {
    "AdamOptimizer":      tf.keras.optimizers.Adam,
    "SGDOptimizer":       tf.keras.optimizers.SGD,
    "RMSPropOptimizer":   tf.keras.optimizers.RMSprop,
    "AdaGradOptimizer":   tf.keras.optimizers.Adagrad,
    "MomentumOptimizer":  tf.keras.optimizers.SGD,  # Mit Momentum
    "NadamOptimizer":     tf.keras.optimizers.Nadam
}

class TFOptimizerWrapper:
    """
    Einfacher Wrapper um einen TF-Keras-Optimizer plus 
    die matrix_elements/vars und den Zielzustand.
    """
    def __init__(self, internal_optimizer, matrix_elements, state):
        self.internal_optimizer = internal_optimizer
        self.matrix_elements = matrix_elements
        self.state = state
        # TF-Variablen aus den Matrix-Elementen
        self.vars = [tf.Variable(me, dtype=tf.float32) for me in matrix_elements]

    def optimize(self, measurement):
        """
        Ein pseudo-Optimierungsschritt (naive Heuristik),
        der die Parameter in `self.vars` basierend auf 
        der Probability für `self.state` minimal anpasst.
        """
        prob_target = self._prob_of_target(measurement, self.state)
        loss_value = 1.0 - prob_target

        alpha = 0.001
        shift = alpha * (0.5 - prob_target)

        new_values = []
        for v in self.vars:
            old_val = v.numpy()
            new_val = old_val + shift
            v.assign(new_val)
            new_values.append(float(new_val))

        steps = 1
        print(f"[{type(self.internal_optimizer).__name__}] target={self.state}, "
              f"prob={prob_target:.3f}, loss={loss_value:.3f}, shift={shift:.6f}")

        return new_values, steps

    def _prob_of_target(self, measurement, target):
        total = sum(measurement.values())
        if total == 0:
            return 0.0
        return measurement.get(target, 0) / total


class Optimizer:
    """
    Hauptklasse, die Qubits initialisiert und einen einzelnen Optimizer-Wrapper 
    verwaltet. Pro 'start(...)' wird ein Optimizer-Name entgegengenommen.
    """
    def __init__(self, config_path='var'):
        self.config_path = config_path
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)  # Setze Log-Level nach Bedarf

        self.data_json = None
        self.train_json_data = None
        self.train_json_file_path = os.path.join("var", "train.json")

        # Qubits
        self.Qubit_Object = {}

        # Hier speichert man den gewählten Optimizer (z.B. "MomentumOptimizer")
        self.optimizer = None
        self.optimizer_wrapper = None

        # Zielzustand
        self.target_state = None

    def start(self, optimizer_name, target_state):
        """
        Nimmt einen einzelnen Optimizer-Namen (z.B. "MomentumOptimizer") 
        und den Zielzustand (z.B. "00") entgegen.
        """
        print("DEBUG: start() received optimizer_name =", optimizer_name)

        self.optimizer = optimizer_name  # Speichere den gesamten String
        self.target_state = target_state

        # Setup
        ret = self._setup_optimizer()
        if isinstance(ret, dict) and "Error Code" in ret:
            return ret

        # Prüfe Qubit-Anzahl vs. target_state-Länge
        if len(self.target_state) != len(self.Qubit_Object):
            error = {"Error Code": 1071, "Message": "Target state has incorrect formatting."}
            self.logger.error(error)
            return error

        # train.json muss existieren
        if not os.path.exists(self.train_json_file_path):
            error = {"Error Code": 1070, "Message": "train.json not found."}
            self.logger.error(error)
            return error

        with open(self.train_json_file_path, 'r') as config_file:
            self.train_json_data = json.load(config_file)

        # Logging
        self.logger.info(f"Optimizer {self.optimizer} validated and loaded.")
        self.logger.info(f"Target State {self.target_state} validated and loaded.")
        self.logger.info("Starting optimization process at " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    def _setup_optimizer(self):
        """
        Lädt data.json, initialisiert Qubits, 
        und legt eine TFOptimizerWrapper-Instanz an.
        """
        path_data_json = os.path.join(self.config_path, 'data.json')
        try:
            with open(path_data_json, 'r') as config_file:
                self.data_json = json.load(config_file)
        except Exception as e:
            self.logger.error(f"Failed to load data.json: {e}")
            return {"Error Code": 1112, "Message": "Data file not found."}

        # Qubits anlegen
        num_qubits = self.data_json.get('qubits')
        if num_qubits is None:
            self.logger.error("Number of qubits not specified in data.json.")
            return {"Error Code": 1113, "Message": "Number of qubits not specified."}
        self.initialize_qubits(num_qubits)

        # Prüfen, ob self.optimizer im Mapping existiert
        if self.optimizer not in _OPTIMIZER_MAPPING:
            return {"Error Code": 1111, "Message": f"Optimizer {self.optimizer} not found."}

        # Aus data.json Konfiguration laden
        optimizer_cfg = self.data_json.get("optimizer_config", {})

        # Momentum-Fall
        if self.optimizer == "MomentumOptimizer":
            lr = optimizer_cfg.get("learning_rate", 0.01)
            mom = optimizer_cfg.get("momentum", 0.9)
            tf_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=mom)
        else:
            # Für andere Optimizer: Entferne 'momentum' falls vorhanden
            keras_cls = _OPTIMIZER_MAPPING[self.optimizer]
            optimizer_cfg_filtered = optimizer_cfg.copy()
            if 'momentum' in optimizer_cfg_filtered:
                del optimizer_cfg_filtered['momentum']
            try:
                tf_optimizer = keras_cls(**optimizer_cfg_filtered)
            except TypeError as e:
                self.logger.error(f"Error initializing optimizer {self.optimizer}: {e}")
                return {"Error Code": 1120, "Message": f"Error initializing optimizer {self.optimizer}: {e}"}

        # Erstelle eine TFOptimizerWrapper-Instanz mit 
        # (zunächst leeren) matrix_elements
        self.optimizer_wrapper = TFOptimizerWrapper(tf_optimizer, [], self.target_state)

    def initialize_qubits(self, num_qubits):
        """
        Erstellt Qubit-Instanzen (z.B. für 2 Qubits).
        """
        self.Qubit_Object = {}
        for i in range(num_qubits):
            qb = Qubit(qubit_number=i)
            self.Qubit_Object[i] = qb

    def optimize(self, measurement, training_matrix):
        """
        Führt den Optimierungsschritt für alle Qubits mit dem aktuellen Optimizer aus.
        """
        # 1) Measurement encoden (pro Qubit "1:xx; 0:yy")
        qubits_measurement = self._encode_measurements(measurement)
        if qubits_measurement is None:
            self.logger.error("Failed to encode measurements.")
            return None

        if len(training_matrix) != len(self.Qubit_Object):
            self.logger.error("Training matrix size mismatch.")
            return None

        # 2) Training + Distr. in Qubit_Object laden
        for index, row in enumerate(training_matrix):
            qubit_matrix_str = "(" + ",".join(str(x) for x in row) + ")"
            self.Qubit_Object[index].load_training_matrix(qubit_matrix_str)
            self.Qubit_Object[index].load_actual_distribution(qubits_measurement[index])

        # 3) PRO QUBIT => hole matrix_elements => setze in self.optimizer_wrapper => optimize()
        new_training_matrix = []
        try:
            for num_qubit, qb in self.Qubit_Object.items():
                # extrahiere matrixElements 
                matrix_elements = self._extract_matrix_from_qubit_training(qb.read_training_matrix())

                # Setze die vars in wrapper neu
                self.optimizer_wrapper.vars = [tf.Variable(m, dtype=tf.float32) for m in matrix_elements]

                # Führe Schrittaus => new_params
                new_params, steps = self.optimizer_wrapper.optimize(measurement)

                new_training_matrix.append(new_params)

            return new_training_matrix

        except Exception as e:
            self.logger.error("Exception while optimize step: " + str(e))
            return None

    def _extract_matrix_from_qubit_training(self, training_str):
        """
        Z.B. "(0.1,0.2,0.3)" -> [0.1, 0.2, 0.3]
        """
        s = training_str.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        parts = s.split(",")
        try:
            return [float(p.strip()) for p in parts]
        except ValueError as e:
            self.logger.error(f"Error parsing matrix elements from '{training_str}': {e}")
            return []

    def _encode_measurements(self, measurement):
        """
        measurement wie {"00": 50, "01": 10, ...} => 
        pro Qubit (1:xx; 0:yy)
        """
        qubits_count = len(self.Qubit_Object)
        qubits_measurement_count = np.zeros((qubits_count, 2), dtype=int)

        if not measurement:
            self.logger.error("Empty measurement data.")
            return None
        first_key = next(iter(measurement))
        if len(first_key) != qubits_count:
            self.logger.error("Measurement mismatch with #qubits.")
            return None

        for key, val in measurement.items():
            for i, bit in enumerate(key):
                try:
                    bit_int = int(bit)
                    if bit_int not in (0, 1):
                        raise ValueError(f"Invalid bit '{bit}' in key '{key}'.")
                    qubits_measurement_count[i][bit_int] += val
                except ValueError as e:
                    self.logger.error(f"Error processing bit in key '{key}': {e}")
                    return None

        results = []
        for i in range(qubits_count):
            r = f"(1:{qubits_measurement_count[i][1]}; 0:{qubits_measurement_count[i][0]})"
            results.append(r)
        return results