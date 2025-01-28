import os  # Importing the os module is essential for os.getpid()
import psutil  # Import psutil for CPU usage monitoring
import time
import datetime
import numpy as np

from core.circuit import Circuit
from ml.optimizer import Optimizer, AdvancedOptimizer
from visualization.visual import Visual


def probability_of_target(counts, target):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get(target, 0) / total


def get_most_likely_outcome(counts):
    if not counts:
        return None
    return max(counts, key=counts.get)


def main():
    # Target State Configuration
    training_phases_input = [
        [0.05, 0.15, 0.2],
        [0.1, 0.25, 0.3]
    ]
    activation_phases_input = [
        [0.02, 0.05, 0.08],
        [0.07, 0.09, 0.12]
    ]
    shots = 10000

    # Create the initial state with a Circuit
    input_circuit = Circuit(
        qubits=2,
        depth=1,
        training_phases=training_phases_input,
        activation_phases=activation_phases_input,
        shots=shots
    )
    input_circuit.measure_all()
    input_circuit.run()
    input_counts = input_circuit.get_counts()

    # Determine the target state
    target_state = get_most_likely_outcome(input_counts)
    if target_state is None:
        print("No target state found.")
        return
    print("Target State:", target_state)

    # Initialize the Visualization tool
    visual = Visual()
    visual.set_initial_distribution(input_counts)

    # Optimizer Configurations
    optimizers_to_try = ["MomentumOptimizer", "AdamOptimizer"]
    training_phases = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    activation_phases = [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]]
    max_iterations = 100000

    # Configuration for the AdvancedOptimizer
    advanced_config = {
        "learning_rate": 0.001,
        "epsilon": 1e-7
    }

    # Training loop for each optimizer
    for opt_name in optimizers_to_try + ["AdvancedOptimizer"]:
        print(f"\n=== Starting Training with {opt_name} ===")

        # Select the appropriate optimizer class
        optimizer_class = AdvancedOptimizer if opt_name == "AdvancedOptimizer" else Optimizer

        # Initialize the optimizer with the appropriate configuration
        if opt_name == "AdvancedOptimizer":
            optimizer = optimizer_class(config_path="var", advanced_mode=True, advanced_config=advanced_config)
            display_name = "AdvancedOptimizer"
        else:
            optimizer = optimizer_class(config_path="var")
            display_name = opt_name

        # Record the start time and CPU usage before starting the optimizer
        start_time = time.time()
        process = psutil.Process(os.getpid())
        cpu_usage_start = process.cpu_percent(interval=None)  # Non-blocking call

        # Start the optimizer
        ret = optimizer.start(display_name if opt_name != "AdvancedOptimizer" else "AdamOptimizer", target_state)

        # Check for errors in starting the optimizer
        if ret is not None and "Error Code" in ret:
            print(f"Error starting optimizer {opt_name}:", ret)
            continue

        current_training_matrix = [phase.copy() for phase in training_phases]
        high_prob_threshold = 0.9  # Minimum allowed probability once 90% is reached
        reached_high_prob = False  # Flag for restricting probability drops

        # Initialize a placeholder for the final matrix
        final_matrix = None

        for iteration in range(1, max_iterations + 1):
            # Build the Circuit
            circuit = Circuit(
                qubits=2,
                depth=1,
                training_phases=current_training_matrix,
                activation_phases=activation_phases,
                shots=shots
            )
            circuit.measure_all()
            circuit.run()

            # Calculate the probability of the target state
            new_counts = circuit.get_counts()
            prob_tgt = probability_of_target(new_counts, target_state)
            visual.record_probability(opt_name, iteration, prob_tgt)

            print(f"Iteration {iteration}, Target State Probability: {prob_tgt * 100:.2f}%")

            # Check thresholds
            if prob_tgt >= 0.99:
                print(f"Stopping: Target state reached with {prob_tgt * 100:.2f}% probability.")
                break
            if prob_tgt >= 0.9:
                reached_high_prob = True
            if reached_high_prob and prob_tgt < high_prob_threshold:
                print(f"Stopping: Probability dropped below 90% ({prob_tgt * 100:.2f}%).")
                break

            # Perform optimization
            updated_matrix = optimizer.optimize(new_counts, current_training_matrix)
            if updated_matrix is None:
                print(f"Error in iteration {iteration} for optimizer {opt_name}.")
                break
            current_training_matrix = updated_matrix

            # Optional: Implement logic to record loss if available
            # Example: visual.record_loss(opt_name, iteration, loss_value)

        # Record the end time and CPU usage after training
        end_time = time.time()
        cpu_usage_end = process.cpu_percent(interval=None)
        cpu_usage = cpu_usage_end  # Simple CPU usage snapshot

        # Record thread details
        visual.record_thread_detail(
            optimizer_name=opt_name,
            start_time=start_time,
            end_time=end_time,
            cpu_usage=cpu_usage
        )

        # Final Circuit after Optimization
        final_circuit = Circuit(
            qubits=2,
            depth=1,
            training_phases=current_training_matrix,
            activation_phases=activation_phases,
            shots=shots
        )
        final_circuit.measure_all()
        final_circuit.run()
        final_counts = final_circuit.get_counts()
        visual.set_final_distribution(opt_name, final_counts)

        # Record Heatmap Data (final matrix)
        # Assuming the optimizer can provide the final matrix. If not, create a matrix based on final_counts.
        # Here, we'll simulate a final matrix for demonstration purposes.
        final_matrix = np.random.rand(5, 5)  # Replace with actual matrix if available
        visual.record_heatmap_data(opt_name, final_matrix)

        print(f"=== Training with {opt_name} completed ===")

    # Generate the PDF report
    visual.generate_pdf(target_state)
    print("Done.")


if __name__ == "__main__":
    main()
