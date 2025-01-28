# tests/test_optimizer.py

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Füge das Hauptverzeichnis zum Python-Pfad hinzu, um Importe zu ermöglichen
# Angenommen, der Test befindet sich im 'tests' Verzeichnis und 'opt.py' im 'lly/ml'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lly.ml.opt import Optimizer
from lly.ml.helpers import (
    TensorFlowAdamOptimizer,
    TensorFlowSGDOptimizer,
    TensorFlowRMSpropOptimizer,
    TensorFlowAdagradOptimizer,
    BinaryCrossEntropyLoss,
    MeanSquaredErrorLoss,
    HingeLoss,
    KullbackLeiblerDivergenceLoss,
    HuberLoss
)

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # Definiere alle verfügbaren Optimizer
        self.optimizers = [
            TensorFlowAdamOptimizer(learning_rate=0.01),
            TensorFlowSGDOptimizer(learning_rate=0.01, momentum=0.9),
            TensorFlowRMSpropOptimizer(learning_rate=0.001, rho=0.9, epsilon=1e-07),
            TensorFlowAdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1)
            # Weitere Optimizer können hier hinzugefügt werden
        ]
        
        # Definiere alle verfügbaren Loss-Funktionen
        self.loss_functions = [
            BinaryCrossEntropyLoss(),
            MeanSquaredErrorLoss(),
            HingeLoss(),
            KullbackLeiblerDivergenceLoss(),
            HuberLoss(delta=1.0)
            # Weitere Loss-Funktionen können hier hinzugefügt werden
        ]
        
        # Ziel-Qubit: Test für sowohl 0 als auch 1
        self.target_qubits = [0, 1]

    def test_optimizer_loss_combinations(self):
        for optimizer in self.optimizers:
            for loss_fn in self.loss_functions:
                for target_qubit in self.target_qubits:
                    with self.subTest(optimizer=optimizer.__class__.__name__,
                                      loss_fn=loss_fn.__class__.__name__,
                                      target_qubit=target_qubit):
                        # Zufällige Anzahl von Phasen (1 bis 12)
                        num_phases = np.random.randint(1, 13)
                        initial_phases = np.random.uniform(-10, 10, size=num_phases).astype(np.float32)
                        
                        # Parameter für den Optimizer
                        # Hole die Lernrate aus dem Optimizer-Instanz
                        learning_rate = optimizer.optimizer.learning_rate.numpy()
                        params = {'learning_rate': learning_rate}
                        
                        # Erstellung der Optimizer-Instanz
                        opt_instance = Optimizer(
                            optimizer=optimizer,
                            loss_function=loss_fn,
                            target_qubit=target_qubit,
                            params=params
                        )
                        
                        # Führe die Optimierung durch
                        optimized_phases = opt_instance.run(initial_phases, epochs=1000, tolerance=1e-4)
                        
                        # Berechne die finalen Wahrscheinlichkeiten
                        optimized_probabilities = 1 / (1 + np.exp(-optimized_phases))
                        
                        # Überprüfe, ob die Wahrscheinlichkeiten dem Ziel-Qubit nahekommen
                        if target_qubit == 1:
                            self.assertTrue(np.all(optimized_probabilities > 0.99),
                                            f"Optimierte Wahrscheinlichkeiten sollten nahe bei 1 liegen, aber sind {optimized_probabilities}")
                        else:
                            self.assertTrue(np.all(optimized_probabilities < 0.01),
                                            f"Optimierte Wahrscheinlichkeiten sollten nahe bei 0 liegen, aber sind {optimized_probabilities}")

if __name__ == '__main__':
    unittest.main()
