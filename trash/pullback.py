import os
import json
import psutil
import time
import datetime
from core.circuit import Circuit
from ml.optimizer import AdvancedOptimizer
from visualization.visual import Visual

def probability_of_target(counts, target):
    total = sum(counts.values())
    return counts.get(target, 0) / total if total else 0.0

def pad_state(state, total_length):
    """
    Pads the binary state to the left with zeros to match the required length.
    """
    return state.zfill(total_length)

def get_most_likely_outcome(counts, total_length):
    """
    Gets the most likely outcome from the counts and pads it to the total length.
    """
    if not counts:
        return None
    most_likely = max(counts, key=counts.get)
    return pad_state(most_likely, total_length)

def main():
    # Configuration
    main_qubits = 4
    pullback_qubits = 2
    total_qubits = main_qubits + pullback_qubits
    depth = 1
    shots = 1000

    # Training and activation phases
    training_phases = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2]
    ]
    activation_phases = [
        [0.3, 0.2, 0.1],
        [0.6, 0.5, 0.4],
        [0.9, 0.8, 0.7],
        [1.2, 1.1, 1.0]
    ]

    # Circuit initialization
    circuit = Circuit(
        qubits=main_qubits,
        depth=depth,
        training_phases=training_phases,
        activation_phases=activation_phases,
        shots=shots
    )
    circuit.measure_all()
    circuit.run()
    counts = circuit.get_counts()

    # Determine target state
    target_state = get_most_likely_outcome(counts)
    if not target_state or len(target_state) != total_qubits:
        print(f"Invalid target state: {target_state}. Expected length: {total_qubits}. Exiting.")
        return
    print(f"Target State (Pullback): {target_state}")

    # Apply pullback
    pullback_training_phases = [[0.15, 0.22, 0.33] for _ in range(pullback_qubits)]
    circuit.apply_pullback(
        pullback_qubits=pullback_qubits,
        reductions=(2, 2, 0.15, 0.22, 0.33),
        measure_mode="all"
    )

    # Visualization
    visual = Visual()
    visual.set_initial_distribution(counts)

    # Optimizer Initialization
    optimizer = AdvancedOptimizer(
        config_path="var",
        advanced_mode=True,
        advanced_config={"learning_rate": 0.001, "epsilon": 1e-7}
    )
    optimizer.start("AdamOptimizer", target_state)

    # Optimization Loop
    max_iterations = 100
    for iteration in range(1, max_iterations + 1):
        circuit.run()
        counts = circuit.get_counts()
        prob_tgt = probability_of_target(counts, target_state)
        visual.record_probability("AdamOptimizer", iteration, prob_tgt)

        print(f"Iteration {iteration}, Target State Probability: {prob_tgt * 100:.2f}%")

        if prob_tgt >= 0.99:
            print(f"Target state reached with {prob_tgt * 100:.2f}% probability.")
            break

        updated_training_phases = optimizer.optimize(counts, training_phases)
        if not updated_training_phases:
            print(f"Error in iteration {iteration}. Exiting optimization loop.")
            break
        training_phases = updated_training_phases

    # Finalize and Save
    visual.generate_pdf(target_state)
    print("Optimization completed. Report generated.")



if __name__ == "__main__":
    main()
