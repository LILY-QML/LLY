# helpers.py

import tensorflow as tf
import numpy as np
from scipy.optimize import minimize

# ------------------------------
# Optimizer Wrapper Klassen
# ------------------------------

class TensorFlowAdamOptimizer:
    def __init__(self, learning_rate=0.001):
        """
        Initialisiert den TensorFlow Adam Optimizer.

        :param learning_rate: Lernrate für den Optimizer
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def apply_gradients(self, gradients, variables):
        """
        Wendet die berechneten Gradienten auf die Variablen an.

        :param gradients: Liste der Gradienten
        :param variables: Liste der zu optimierenden TensorFlow-Variablen
        """
        self.optimizer.apply_gradients(zip(gradients, variables))

class TensorFlowSGDOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Initialisiert den TensorFlow SGD Optimizer.

        :param learning_rate: Lernrate für den Optimizer
        :param momentum: Momentum-Wert
        """
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    
    def apply_gradients(self, gradients, variables):
        """
        Wendet die berechneten Gradienten auf die Variablen an.

        :param gradients: Liste der Gradienten
        :param variables: Liste der zu optimierenden TensorFlow-Variablen
        """
        self.optimizer.apply_gradients(zip(gradients, variables))

class TensorFlowRMSpropOptimizer:
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-07):
        """
        Initialisiert den TensorFlow RMSprop Optimizer.

        :param learning_rate: Lernrate für den Optimizer
        :param rho: RMSprop rho-Parameter
        :param epsilon: RMSprop epsilon-Parameter
        """
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon)
    
    def apply_gradients(self, gradients, variables):
        """
        Wendet die berechneten Gradienten auf die Variablen an.

        :param gradients: Liste der Gradienten
        :param variables: Liste der zu optimierenden TensorFlow-Variablen
        """
        self.optimizer.apply_gradients(zip(gradients, variables))

class ScipyNelderMeadOptimizer:
    def __init__(self, learning_rate=0.01):
        """
        Initialisiert den Scipy Nelder-Mead Optimizer.

        :param learning_rate: Lernrate für den Optimizer (wird als Schrittgröße interpretiert)
        """
        self.learning_rate = learning_rate
    
    def apply_gradients(self, gradients, variables):
        """
        Da Scipy Optimizer funktionbasiert sind, wird hier eine Dummy-Implementierung verwendet.
        Tatsächliche Implementierungen erfordern eine andere Herangehensweise.

        :param gradients: Liste der Gradienten (nicht genutzt)
        :param variables: Liste der zu optimierenden Numpy-Arrays
        """
        raise NotImplementedError("Scipy Optimizer erfordern eine funktionbasierte Implementierung.")

# Weitere Optimizer können hier hinzugefügt werden.

# ------------------------------
# Loss Function Wrapper Klassen
# ------------------------------

class BinaryCrossEntropyLoss:
    def __init__(self):
        pass
    
    def compute_loss(self, predicted, target):
        """
        Berechnet den Binary Cross-Entropy Loss zwischen den vorhergesagten und den Zielwerten.

        :param predicted: Tensor mit vorhergesagten Wahrscheinlichkeiten
        :param target: Tensor mit Zielwerten (0 oder 1)
        :return: Berechneter Loss-Wert als Tensor
        """
        predicted = tf.clip_by_value(predicted, 1e-15, 1 - 1e-15)  # Vermeidet log(0)
        loss = -tf.reduce_mean(target * tf.math.log(predicted) + (1 - target) * tf.math.log(1 - predicted))
        return loss

class MeanSquaredErrorLoss:
    def __init__(self):
        pass
    
    def compute_loss(self, predicted, target):
        """
        Berechnet den Mean Squared Error Loss zwischen den vorhergesagten und den Zielwerten.

        :param predicted: Tensor mit vorhergesagten Werten
        :param target: Tensor mit Zielwerten
        :return: Berechneter Loss-Wert als Tensor
        """
        loss = tf.reduce_mean(tf.square(predicted - target))
        return loss

class HingeLoss:
    def __init__(self):
        pass
    
    def compute_loss(self, predicted, target):
        """
        Berechnet den Hinge Loss zwischen den vorhergesagten und den Zielwerten.

        :param predicted: Tensor mit vorhergesagten Werten
        :param target: Tensor mit Zielwerten (-1 oder 1)
        :return: Berechneter Loss-Wert als Tensor
        """
        loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - target * predicted))
        return loss

class KullbackLeiblerDivergenceLoss:
    def __init__(self):
        pass
    
    def compute_loss(self, predicted, target):
        """
        Berechnet die Kullback-Leibler Divergenz zwischen den vorhergesagten und den Zielwerten.

        :param predicted: Tensor mit vorhergesagten Wahrscheinlichkeiten
        :param target: Tensor mit Zielwerten
        :return: Berechneter Loss-Wert als Tensor
        """
        predicted = tf.clip_by_value(predicted, 1e-15, 1 - 1e-15)
        target = tf.clip_by_value(target, 1e-15, 1 - 1e-15)
        loss = tf.reduce_sum(target * tf.math.log(target / predicted))
        return loss

class HuberLoss:
    def __init__(self, delta=1.0):
        """
        Initialisiert den Huber Loss.

        :param delta: Schwellenwert für die Huber-Transition
        """
        self.delta = delta
    
    def compute_loss(self, predicted, target):
        """
        Berechnet den Huber Loss zwischen den vorhergesagten und den Zielwerten.

        :param predicted: Tensor mit vorhergesagten Werten
        :param target: Tensor mit Zielwerten
        :return: Berechneter Loss-Wert als Tensor
        """
        error = predicted - target
        is_small_error = tf.abs(error) <= self.delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = self.delta * tf.abs(error) - 0.5 * self.delta**2
        loss = tf.where(is_small_error, squared_loss, linear_loss)
        return tf.reduce_mean(loss)

