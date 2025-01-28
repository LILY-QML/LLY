# opt.py

import os
import numpy as np
import tensorflow as tf
from helpers import (
    TensorFlowAdamOptimizer,
    TensorFlowSGDOptimizer,
    TensorFlowRMSpropOptimizer,
    ScipyNelderMeadOptimizer,
    BinaryCrossEntropyLoss,
    MeanSquaredErrorLoss,
    HingeLoss,
    KullbackLeiblerDivergenceLoss,
    HuberLoss
)

# Optional: TensorFlow nur CPU verwenden, falls CUDA Fehler auftreten
# Deaktiviere die GPU-Nutzung, falls keine GPU verfügbar ist oder Probleme auftreten
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Optimizer:
    def __init__(self, optimizer, loss_function, target_qubit, params):
        """
        Initialisiert den Optimizer mit dem angegebenen Optimizer, der Loss Function,
        dem Ziel-Qubit und den zusätzlichen Parametern.

        :param optimizer: Instanz des Optimizers (z.B. TensorFlowAdamOptimizer)
        :param loss_function: Instanz der Loss Function (z.B. BinaryCrossEntropyLoss)
        :param target_qubit: Zielwert des Qubits (0 oder 1)
        :param params: Dictionary mit zusätzlichen Parametern
        """
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.target_qubit = target_qubit
        self.params = params
        self.check_inputs()
        self.create_combination()

    def check_inputs(self):
        """
        Überprüft, ob der Optimizer und die Loss Function korrekt initialisiert wurden
        und ob alle notwendigen Parameter vorhanden sind.
        """
        if not self.optimizer:
            raise ValueError("Optimizer ist nicht definiert.")
        
        if not self.loss_function:
            raise ValueError("Loss Function ist nicht definiert.")
        
        if self.target_qubit not in [0, 1]:
            raise ValueError("target_qubit muss entweder 0 oder 1 sein.")
        
        # Weitere Parameterprüfungen können hier hinzugefügt werden
        required_params = ['learning_rate']
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Erforderlicher Parameter '{param}' fehlt.")

    def create_combination(self):
        """
        Kombiniert den Optimizer und die Loss Function.
        Stellt sicher, dass der Optimizer mit der Loss Function kompatibel ist.
        """
        # In diesem einfachen Beispiel gibt es keine spezielle Kombination,
        # aber hier könnten zusätzliche Schritte hinzugefügt werden.
        print("Optimizer und Loss Function erfolgreich kombiniert.")

    def run(self, phases, epochs=100, tolerance=1e-3):
        """
        Führt den Optimierungsprozess aus, um die Phasen so anzupassen, dass die Wahrscheinlichkeitsverteilung
        näher an den Zielwert kommt.

        :param phases: Initiale Phasenwerte (z.B. numpy Array)
        :param epochs: Maximale Anzahl der Optimierungsdurchläufe
        :param tolerance: Toleranz für den Abbruch der Optimierung
        :return: Optimierte Phasenwerte
        """
        # Konvertiere Phasen zu TensorFlow-Variablen
        phases_tf = tf.Variable(phases, dtype=tf.float32)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Aktualisiere Wahrscheinlichkeiten basierend auf aktuellen Phasen
                updated_probabilities = self.update_probabilities(phases_tf)
                
                # Berechne den Verlust
                target = tf.fill(updated_probabilities.shape, float(self.target_qubit))
                loss = self.loss_function.compute_loss(updated_probabilities, target)
            
            # Berechne Gradienten
            gradients = tape.gradient(loss, [phases_tf])
            
            # Wende Gradienten an
            self.optimizer.apply_gradients(gradients, [phases_tf])
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.6f}")
            
            # Überprüfe, ob der Zielwert erreicht ist
            if loss.numpy() < tolerance:
                print("Zielwert erreicht. Optimierung beendet.")
                break

        return phases_tf.numpy()

    def update_probabilities(self, phases):
        """
        Aktualisiert die Wahrscheinlichkeitsverteilung basierend auf den Phasen.
        Dies ist eine einfache Implementierung unter Verwendung der Sigmoid-Funktion.

        :param phases: Tensor der aktuellen Phasenwerte
        :return: Aktualisierte Wahrscheinlichkeitsverteilung als Tensor
        """
        probabilities = tf.math.sigmoid(phases)
        return probabilities

# Anwendungsbeispiel

if __name__ == "__main__":
    # Auswahl des Optimizers und der Loss Function
    # Hier kannst du verschiedene Optimizer und Loss-Funktionen auswählen
    loss_fn = BinaryCrossEntropyLoss()
    optimizer_instance = TensorFlowAdamOptimizer(learning_rate=0.01)
    
    # Alternativ könntest du einen anderen Optimizer verwenden, z.B.:
    # optimizer_instance = TensorFlowSGDOptimizer(learning_rate=0.01, momentum=0.9)
    # optimizer_instance = TensorFlowRMSpropOptimizer(learning_rate=0.001, rho=0.9, epsilon=1e-07)
    
    # Ziel-Qubit und zusätzliche Parameter
    target_qubit = 1
    params = {'learning_rate': 0.01}
    
    # Initiale Phasen
    initial_phases = np.array([0.0, 0.0], dtype=np.float32)  # Beispielhafte Phasenwerte
    
    # Erstellung der Optimizer-Instanz
    optimizer = Optimizer(
        optimizer=optimizer_instance,
        loss_function=loss_fn,
        target_qubit=target_qubit,
        params=params
    )
    
    print("Optimizer erfolgreich initialisiert und überprüft.")
    
    # Ausführen des Optimierungsprozesses
    optimized_phases = optimizer.run(initial_phases, epochs=1000, tolerance=1e-4)
    
    # Berechne die finalen Wahrscheinlichkeiten
    optimized_probabilities = 1 / (1 + np.exp(-optimized_phases))
    
    print("Optimierte Phasen:", optimized_phases)
    print("Optimierte Wahrscheinlichkeiten:", optimized_probabilities)
