# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
This module contains the `Circuit` class, which encapsulates the logic for
building and executing quantum circuits with customizable "pullback" qubits.

The workflow includes:
1. Initializing a main circuit with a specified number of qubits and a certain
   depth, using provided phase matrices for training and activation.
2. Optionally adding a pullback circuit (extra qubits) and coupling them
   (entangling) with the main qubits.
3. Applying additional phase gates on these pullback qubits before entangling.
4. Measuring qubits according to various modes (all, main-only, pullback-only,
   or a custom list of qubits).

This file is tailored for usage with Sphinx documentation. Every function is
documented using Sphinx-style docstrings, and there are extensive in-line
comments to explain the code flow in detail.
"""

from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
import copy

class Circuit:
    """
    A class that represents a quantum circuit with optional "pullback" qubits
    and flexible measurement options.

    :param qubits: Number of qubits in the main circuit (without pullback).
    :type qubits: int
    :param depth: Depth of the circuit, which determines how many times the
        sequence of gates is repeated.
    :type depth: int
    :param training_phases: A 2D list (or matrix) of floats, defining the
        training phases to be applied on each qubit. Must have the shape
        ``(qubits, depth * 3)``.
    :type training_phases: list[list[float]]
    :param activation_phases: A 2D list (or matrix) of floats, defining the
        activation phases to be applied on each qubit. Must have the shape
        ``(qubits, depth * 3)``.
    :type activation_phases: list[list[float]]
    :param shots: Number of shots (repetitions) to be used in the quantum
        simulator backend.
    :type shots: int
    :param aer_backend: The backend name to be retrieved from ``qiskit_aer.Aer``.
        By default, this is ``'aer_simulator'``.
    :type aer_backend: str
    """

    def __init__(self, qubits, depth, training_phases, activation_phases, shots, aer_backend='aer_simulator'):
        """
        Constructor for the Circuit class.

        Initializes a quantum circuit with the specified number of qubits, depth,
        phase matrices, and the number of shots for simulation. This constructor
        does not include measurements by default, allowing you to measure only
        when needed (e.g., after adding pullback qubits).

        :param qubits: Number of qubits in the main circuit (without pullback).
        :type qubits: int
        :param depth: Depth of the circuit, which determines how many times the
            sequence of gates is repeated.
        :type depth: int
        :param training_phases: 2D list of floats for training phases per qubit.
            Must match shape requirements.
        :type training_phases: list[list[float]]
        :param activation_phases: 2D list of floats for activation phases per qubit.
            Must match shape requirements.
        :type activation_phases: list[list[float]]
        :param shots: Number of shots (repetitions) in simulation.
        :type shots: int
        :param aer_backend: Name of the Aer backend to use, defaults to 'aer_simulator'.
        :type aer_backend: str
        """
        self.qubits = qubits
        self.depth = depth
        self.training_phases = training_phases
        self.activation_phases = activation_phases
        self.shots = shots
        self.aer_backend = aer_backend
        self.simulation_result = None  # This will hold simulation results after running

        # Create an empty QuantumCircuit with `qubits` quantum bits
        # Note: We do not create classical registers here yet, as we want
        # to measure optionally at a later stage.
        self.circuit = QuantumCircuit(qubits)

        # Initialize the gates for the main circuit. This will place
        # phase gates (p) and Hadamard gates (h) according to the provided
        # training and activation phase matrices.
        self.initialize_gates()

    def initialize_gates(self):
        """
        Initializes the gates of the circuit based on the training and
        activation phase matrices.

        Each qubit receives a sequence of phase gates derived from
        ``training_phases`` and ``activation_phases``. Additionally,
        Hadamard gates are inserted at specific positions.

        The number of phase gates per qubit depends on:
        - The depth of the circuit (`self.depth`).
        - The fact that each "L-gate" stage consists of 3 sub-phases.

        :raises ValueError: If the dimensions of `training_phases` or
            `activation_phases` are not as expected.
        """
        # The required number of rows in the phase matrices corresponds
        # to the number of qubits.
        required_phase_entries = self.qubits

        # Check that each matrix has exactly `required_phase_entries` rows
        if len(self.training_phases) != required_phase_entries or len(self.activation_phases) != required_phase_entries:
            raise ValueError(
                f"Training and activation phases must each have {required_phase_entries} rows."
            )

        # Check that each row in the phase matrices has the correct length,
        # which should be `self.depth * 3`.
        if any(len(row) != self.depth * 3 for row in self.training_phases) or \
           any(len(row) != self.depth * 3 for row in self.activation_phases):
            raise ValueError(
                f"Each phase entry must have a length of {self.depth * 3}."
            )

        # For every qubit, and for each depth layer, apply the L-gates
        # using the method `apply_l_gate()`.
        for qubit in range(self.qubits):
            for d in range(self.depth):
                self.apply_l_gate(qubit, d)

    def apply_l_gate(self, qubit, depth_index):
        """
        Applies the L-gate sequence for a single qubit and a specific depth index.

        The L-gate is composed of three phases:
        1. Training phase (TP)
        2. Activation phase (AP)
        3. Potentially repeated with a Hadamard gate in between.

        :param qubit: The index of the qubit to apply the gate on.
        :type qubit: int
        :param depth_index: An integer indicating which depth layer is being
            processed (0-based).
        :type depth_index: int
        """
        # Each L-gate consists of 3 sub-steps per depth layer
        for i in range(3):
            # The index in the phase matrix is computed by multiplying
            # the depth index by 3, then adding `i`.
            index = depth_index * 3 + i

            # Retrieve the training phase and activation phase for this
            # qubit and sub-step.
            tp_phase = self.training_phases[qubit][index]
            ap_phase = self.activation_phases[qubit][index]

            # Apply the training phase as a phase gate.
            self.circuit.p(tp_phase, qubit)

            # Apply the activation phase as a phase gate.
            self.circuit.p(ap_phase, qubit)

            # Insert a Hadamard gate after certain sub-phases. In this
            # design, we do it for i == 1 and i == 2, to match the
            # original pattern.
            if i == 1 or i == 2:
                self.circuit.h(qubit)

    def remove_measurements(self, qc):
        """
        Creates a copy of the given quantum circuit without any
        measurement (measure) instructions.

        This is useful if you want to ensure that all measurements
        happen only at the end of the circuit, rather than possibly
        being scattered throughout.

        :param qc: The original quantum circuit that may contain
            measurement instructions.
        :type qc: QuantumCircuit

        :return: A new quantum circuit, identical to `qc` except for
            the absence of measure instructions.
        :rtype: QuantumCircuit
        """
        # Create a new QuantumCircuit with the same number of qubits
        # as the original one
        new_qc = QuantumCircuit(qc.num_qubits)

        # Iterate over each instruction in the original circuit.
        # If the instruction is a measurement, skip it. Otherwise,
        # append it to the new circuit.
        for instr, qargs, cargs in qc.data:
            if instr.name != 'measure':
                new_qc.append(instr, qargs, cargs)

        return new_qc

    def run(self):
        """
        Runs the quantum circuit simulation using the specified Aer backend
        and returns the result.

        :return: The simulation result, which provides measurement counts,
            probabilities, etc.
        :rtype: qiskit.result.Result
        """
        # Get the requested backend from the Aer provider
        simulator = Aer.get_backend(self.aer_backend)

        # Transpile the circuit for the specific simulator
        compiled_circuit = transpile(self.circuit, simulator)

        # Execute the compiled circuit with the specified number of shots
        self.simulation_result = simulator.run(compiled_circuit, shots=self.shots).result()

        return self.simulation_result

    def get_counts(self):
        """
        Retrieves the measurement counts from the last simulation run.

        :raises RuntimeError: If the circuit has not been executed yet via `run()`.

        :return: A dictionary containing measurement outcomes and their
            respective frequencies.
        :rtype: dict
        """
        # Check if the circuit was executed
        if self.simulation_result is not None:
            return self.simulation_result.get_counts(self.circuit)
        else:
            raise RuntimeError("The circuit has not been executed yet.")

    def measure_all(self):
        """
        A helper method that measures all qubits in the current circuit.

        This method adds a classical register of size ``self.circuit.num_qubits``
        and measures each quantum bit into the corresponding classical bit.
        """
        n = self.circuit.num_qubits
        # Create a classical register with `n` bits (matching the number of qubits)
        self.circuit.add_register(n)

        # Measure all qubits (0..n-1) into all classical bits (0..n-1)
        self.circuit.measure(range(n), range(n))

    def copy(self):
        """
        Creates a deep copy of the Circuit instance. This includes copying
        the quantum circuit, the phase matrices, and other attributes.

        :return: A deep copy of the current Circuit instance.
        :rtype: Circuit
        """
        return copy.deepcopy(self)

    def __str__(self):
        """
        Returns a textual representation of the circuit (drawn in ASCII form).

        :return: A string showing the ASCII-art representation of the circuit.
        :rtype: str
        """
        return self.circuit.draw(output='text').__str__()

    def __repr__(self):
        """
        Returns a string representation of the circuit (for debugging and
        interactive use). This is typically the same as :meth:`__str__`.

        :return: A string representation of the circuit drawing.
        :rtype: str
        """
        return self.__str__()

    def to_dict(self):
        """
        Returns a dictionary representation of the circuit, intended for JSON
        or other serialization formats.

        :return: A dictionary with (currently) a single key, "circuit",
            containing an ASCII representation of the circuit.
        :rtype: dict
        """
        return {
            "circuit": self.circuit.draw(output='text')
        }

    def apply_pullback(self,
                       pullback_qubits,
                       reductions,
                       measure_mode="all",
                       measure_qubits=None):
        """
        Extends the current circuit by adding a number of "pullback" qubits
        and applying additional gates (including user-specified phase gates)
        before entangling them with the main qubits.

        This method also supports measuring a subset (or all) of the qubits
        according to the provided mode.

        .. note::
           The format for ``reductions`` is expected to be:

           .. code-block:: python

              (r_main, r_pull, phase1, phase2, phase3)

           Where:
           
           * ``r_main`` is the number of main qubits (starting from index 0)
             that will be entangled.
           * ``r_pull`` is the number of pullback qubits (among those newly
             added) that will be entangled.
           * ``phase1, phase2, phase3`` are the phase gate values (floats)
             to apply on **each** pullback qubit *before* doing the
             entangling (e.g. CNOT).

        :param pullback_qubits: The number of additional qubits to add to
            the current circuit as pullback qubits.
        :type pullback_qubits: int
        :param reductions: A 5-element tuple defining how to handle the
            pullback entanglement and phases. Specifically:
            ``(r_main, r_pull, phase1, phase2, phase3)``.
        :type reductions: tuple
        :param measure_mode: Determines which qubits to measure at the end.
            Valid options are ``"all"``, ``"main"``, ``"pullback"``, or
            ``"custom"``.
            * ``"all"`` measures every qubit in the extended circuit.
            * ``"main"`` measures only the original main qubits.
            * ``"pullback"`` measures only the newly added pullback qubits.
            * ``"custom"`` measures only the qubits specified by
              ``measure_qubits``.
        :type measure_mode: str
        :param measure_qubits: A list of specific qubit indices to measure
            if ``measure_mode="custom"``. Ignored otherwise.
        :type measure_qubits: list or None

        :raises ValueError: If any of the conditions on ``reductions`` or
            the chosen measurement mode are violated.
        """
        # We expect exactly 5 values in `reductions`
        if len(reductions) != 5:
            raise ValueError(
                "reductions must contain exactly 5 values: (r_main, r_pull, phase1, phase2, phase3)."
            )

        # Unpack the 5 values from the `reductions` tuple
        r_main, r_pull, phase1, phase2, phase3 = reductions

        # Keep track of how many qubits we currently have
        old_qubits = self.circuit.num_qubits

        # Basic validation to ensure the chosen r_main and r_pull are feasible
        if r_main > old_qubits:
            raise ValueError(
                f"r_main ({r_main}) cannot exceed the number of main qubits ({old_qubits})."
            )
        if r_pull > pullback_qubits:
            raise ValueError(
                f"r_pull ({r_pull}) cannot exceed the number of pullback qubits ({pullback_qubits})."
            )

        # Remove any existing measurement gates from the old circuit,
        # so that we only measure once at the end.
        old_circ_no_measure = self.remove_measurements(self.circuit)

        # The total number of qubits after adding pullback qubits
        total_qubits = old_qubits + pullback_qubits

        # Determine how many classical bits we need, based on `measure_mode`
        if measure_mode == "all":
            classical_bits = total_qubits
        elif measure_mode == "main":
            classical_bits = old_qubits
        elif measure_mode == "pullback":
            classical_bits = pullback_qubits
        elif measure_mode == "custom":
            if not measure_qubits:
                raise ValueError(
                    "When measure_mode='custom', you must provide a list of qubits in measure_qubits."
                )
            classical_bits = len(measure_qubits)
        else:
            raise ValueError(
                "measure_mode must be one of: 'all', 'main', 'pullback', 'custom'."
            )

        # Create a new QuantumCircuit with enough qubits and classical bits
        new_circuit = QuantumCircuit(total_qubits, classical_bits)

        # Step 1: Compose the old circuit (with no measurement instructions) into
        #         the new circuit. The old qubits map to indices [0..old_qubits-1].
        new_circuit.compose(
            old_circ_no_measure,
            qubits=range(old_qubits),
            inplace=True
        )

        # Step 2: For each pair of (r_main, r_pull), apply the specified phase gates
        #         on the pullback qubits BEFORE performing CNOT to entangle them
        #         with the main qubits.
        for i in range(r_main):
            for j in range(r_pull):
                main_qubit = i  # among the first old_qubits
                pull_qubit = old_qubits + j  # among the newly added pullback qubits

                # Apply the three phase gates to the pullback qubit
                new_circuit.p(phase1, pull_qubit)
                new_circuit.p(phase2, pull_qubit)
                new_circuit.p(phase3, pull_qubit)

                # Then apply an entangling gate, e.g., a CNOT
                new_circuit.cx(main_qubit, pull_qubit)

        # Step 3: Perform measurements based on measure_mode
        if measure_mode == "all":
            # Measure all qubits [0..total_qubits-1] into classical bits [0..total_qubits-1]
            new_circuit.measure(range(total_qubits), range(total_qubits))

        elif measure_mode == "main":
            # Measure only the original (main) qubits [0..old_qubits-1]
            new_circuit.measure(range(old_qubits), range(old_qubits))

        elif measure_mode == "pullback":
            # Measure only the new (pullback) qubits [old_qubits..old_qubits+r_pullback-1]
            new_circuit.measure(
                range(old_qubits, old_qubits + pullback_qubits),
                range(pullback_qubits)
            )

        elif measure_mode == "custom":
            # Measure only the specified qubits in `measure_qubits`
            for i, q_idx in enumerate(measure_qubits):
                new_circuit.measure(q_idx, i)

        # Update our circuit to the new extended one
        self.circuit = new_circuit

        # Update the `self.qubits` count to reflect the new total
        self.qubits = total_qubits

        # Reset any previous simulation result, because the circuit changed
        self.simulation_result = None


# ----------------------------------------------------------------------------
# Example usage (if this file is run as a script)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Training and activation phases for a 2-qubit main circuit with depth=2
    # => each qubit has 2 * 3 = 6 phase values
    training_phases = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Phases for qubit 0
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]   # Phases for qubit 1
    ]

    activation_phases = [
        [0.3, 0.2, 0.1, 0.6, 0.5, 0.4],  # Phases for qubit 0
        [1.2, 1.1, 1.0, 0.9, 0.8, 0.7]   # Phases for qubit 1
    ]

    # Create a Circuit instance with 2 main qubits, depth=2, and the above phases
    circuit = Circuit(
        qubits=2,
        depth=2,
        training_phases=training_phases,
        activation_phases=activation_phases,
        shots=1024
    )

    print("Initial main circuit (2 qubits, no measurement yet):")
    print(circuit)

    # Now we apply the pullback with 2 additional qubits, specifying
    # a 5-element tuple for `reductions`:
    # (r_main, r_pull, phase1, phase2, phase3)
    # In this example, we are entangling 2 main qubits with 2 pullback qubits,
    # and applying three phase gates (0.15, 0.22, 0.33) on each pullback qubit
    # before the CNOT.
    circuit.apply_pullback(
        pullback_qubits=2,
        reductions=(2, 2, 0.15, 0.22, 0.33),
        measure_mode="all"  # measure all qubits at the end
    )

    print("\nExtended circuit (4 qubits total, measurement at the end):")
    print(circuit)

    # Finally, run the simulation
    result = circuit.run()
    print("\nMeasurement results:")
    print(circuit.get_counts())
