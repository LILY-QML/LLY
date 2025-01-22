# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Filename: test_circuit.py
# This file contains unit tests for the Circuit class. The tests cover:
# 1) Valid and invalid initialization (dimension checks, etc.).
# 2) Pullback application with different "reductions" and measurement modes.
# 3) Ensuring that no measurements are placed prematurely and that they occur
#    as requested by measure_mode.
#
# Each test method includes brief Sphinx-friendly docstrings explaining
# the purpose of the test.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import unittest
from qiskit.result import Result

# Import the Circuit class from the module where it is defined.
# Example:
# from my_circuit_module import Circuit
#
# Adjust the import path as necessary for your project structure.
from lly.core.circuit import Circuit


class TestCircuit(unittest.TestCase):
    """
    Test suite for the Circuit class, ensuring that both main circuit
    initialization and pullback functionalities work as expected.
    """

    def setUp(self):
        """
        Common setup for all tests. Defines some default training/activation
        phases and other parameters.
        """
        # Example valid phases for a 2-qubit system, depth=2 (=> 6 phase values per qubit)
        self.training_phases_valid = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        ]
        self.activation_phases_valid = [
            [0.3, 0.2, 0.1, 0.6, 0.5, 0.4],
            [1.2, 1.1, 1.0, 0.9, 0.8, 0.7]
        ]
        self.shots = 256

    def test_init_valid(self):
        """
        Test that a Circuit initializes correctly with valid phase dimensions.
        """
        try:
            circuit = Circuit(
                qubits=2,
                depth=2,
                training_phases=self.training_phases_valid,
                activation_phases=self.activation_phases_valid,
                shots=self.shots
            )
            # Check we have the right number of qubits
            self.assertEqual(circuit.qubits, 2)
            self.assertIsNone(circuit.simulation_result, "Result should be None before run()")
        except Exception as e:
            self.fail(f"Unexpected exception during valid initialization: {e}")

    def test_init_invalid_phases_dimensions(self):
        """
        Test that initializing with incorrect phase dimensions raises ValueError.
        """
        # Wrong length in training_phases
        training_phases_invalid = [
            [0.1, 0.2, 0.3, 0.4],  # Only 4 entries instead of 6
            [0.7, 0.8, 0.9, 1.0]
        ]
        with self.assertRaises(ValueError):
            _ = Circuit(
                qubits=2,
                depth=2,
                training_phases=training_phases_invalid,
                activation_phases=self.activation_phases_valid,
                shots=self.shots
            )

    def test_run_simulation(self):
        """
        Test running the circuit simulation with a valid configuration.
        Ensures that get_counts() returns a non-empty dictionary.
        """
        circuit = Circuit(
            qubits=2,
            depth=2,
            training_phases=self.training_phases_valid,
            activation_phases=self.activation_phases_valid,
            shots=self.shots
        )
        # Add measurement to all qubits manually
        circuit.measure_all()

        result = circuit.run()
        self.assertIsInstance(result, Result, "run() should return a Qiskit Result object.")

        counts = circuit.get_counts()
        self.assertIsInstance(counts, dict, "get_counts() should return a dictionary.")
        self.assertGreater(len(counts), 0, "Counts dictionary should not be empty.")

    def test_apply_pullback_valid(self):
        """
        Test applying pullback with valid 'reductions' tuple and ensure
        simulation runs correctly.
        """
        circuit = Circuit(
            qubits=2,
            depth=2,
            training_phases=self.training_phases_valid,
            activation_phases=self.activation_phases_valid,
            shots=self.shots
        )
        # Pullback: 2 new qubits, (r_main=2, r_pull=2, phase1=0.15, phase2=0.22, phase3=0.33)
        # measure_mode='all'
        circuit.apply_pullback(
            pullback_qubits=2,
            reductions=(2, 2, 0.15, 0.22, 0.33),
            measure_mode="all"
        )

        # Now we have 4 qubits total
        self.assertEqual(circuit.qubits, 4, "Total qubits should be 4 after applying pullback.")

        # Run simulation
        circuit.run()
        counts = circuit.get_counts()
        self.assertGreater(len(counts), 0, "Resulting counts from pullback circuit should not be empty.")

    def test_apply_pullback_invalid_reductions(self):
        """
        Test that apply_pullback raises ValueError if 'reductions' has wrong size
        or if r_main / r_pull exceed limits.
        """
        circuit = Circuit(
            qubits=2,
            depth=2,
            training_phases=self.training_phases_valid,
            activation_phases=self.activation_phases_valid,
            shots=self.shots
        )

        # Wrong size for reductions (should have 5 elements)
        with self.assertRaises(ValueError):
            circuit.apply_pullback(
                pullback_qubits=2,
                reductions=(2, 2, 0.15, 0.22),  # Only 4 elements
                measure_mode="all"
            )

        # r_main or r_pull bigger than possible
        with self.assertRaises(ValueError):
            circuit.apply_pullback(
                pullback_qubits=2,
                # r_main=3 (invalid for 2 main qubits), r_pull=2, then phases
                reductions=(3, 2, 0.15, 0.22, 0.33),
                measure_mode="all"
            )

    def test_measure_modes(self):
        """
        Test measuring different modes ('main', 'pullback', 'custom').
        We only verify that the circuit creation doesn't fail and returns valid counts.
        """
        circuit = Circuit(
            qubits=2,
            depth=1,  # short depth for faster test
            training_phases=self.training_phases_valid,
            activation_phases=self.activation_phases_valid,
            shots=self.shots
        )

        # Example: Add 2 pullback qubits, but measure only 'main' qubits
        circuit.apply_pullback(
            pullback_qubits=2,
            reductions=(2, 2, 0.1, 0.2, 0.3),
            measure_mode="main"
        )
        # We expect classical bits = 2 (since measure_mode="main")
        self.assertEqual(circuit.circuit.num_clbits, 2)

        # Run and check
        circuit.run()
        counts_main = circuit.get_counts()
        self.assertGreater(len(counts_main), 0, "'main' measurement counts should not be empty.")

        # Now let's test 'pullback'
        circuit2 = Circuit(
            qubits=2,
            depth=1,
            training_phases=self.training_phases_valid,
            activation_phases=self.activation_phases_valid,
            shots=self.shots
        )
        circuit2.apply_pullback(
            pullback_qubits=2,
            reductions=(2, 2, 0.1, 0.2, 0.3),
            measure_mode="pullback"
        )
        # measure_mode="pullback" => we measure only the 2 newly added qubits
        self.assertEqual(circuit2.circuit.num_clbits, 2, "Should have exactly 2 classical bits for pullback mode.")
        circuit2.run()
        counts_pull = circuit2.get_counts()
        self.assertGreater(len(counts_pull), 0, "'pullback' measurement counts should not be empty.")

        # Test 'custom' measurement
        circuit3 = Circuit(
            qubits=3,
            depth=1,
            # Provide valid 3-qubit phases
            training_phases=[
                [0.1, 0.2, 0.3],     # 1 depth * 3 phases = 3 for qubit 0
                [0.4, 0.5, 0.6],     # qubit 1
                [0.7, 0.8, 0.9]      # qubit 2
            ],
            activation_phases=[
                [1.1, 1.2, 1.3],
                [1.4, 1.5, 1.6],
                [1.7, 1.8, 1.9]
            ],
            shots=self.shots
        )
        # Now let's apply pullback: (r_main=2, r_pull=1, plus phases)
        # measure_mode="custom": measure only qubit 0 and 4
        # qubit 4 will be (old_qubits=3 + pullback_qubits=1) => index 3 for the new qubit
        circuit3.apply_pullback(
            pullback_qubits=1,
            reductions=(2, 1, 0.12, 0.13, 0.14),
            measure_mode="custom",
            measure_qubits=[0, 3]
        )
        self.assertEqual(circuit3.circuit.num_clbits, 2)
        circuit3.run()
        counts_custom = circuit3.get_counts()
        self.assertGreater(len(counts_custom), 0, "'custom' measurement counts should not be empty.")


# If you want to run this test file directly, uncomment the lines below:
# if __name__ == '__main__':
#     unittest.main()
