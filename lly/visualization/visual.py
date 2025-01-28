import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import datetime
import numpy as np
import time
from collections import defaultdict

class Visual:
    """
    This class collects data from multiple optimizers and input matrices, tracks their training progress,
    and generates a detailed PDF report with charts, tables, and metrics.
    """

    def __init__(self):
        # Data Structures
        self.iteration_probs = defaultdict(lambda: defaultdict(list))  # {input_matrix: {optimizer_name: [(iteration, prob), ...]}}
        self.loss_data = defaultdict(lambda: defaultdict(list))        # {input_matrix: {optimizer_name: [(iteration, loss), ...]}}
        self.initial_distributions = {}                                # {input_matrix: counts}
        self.final_distributions = defaultdict(dict)                   # {input_matrix: {optimizer_name: counts}}
        self.optimizer_metrics = defaultdict(dict)                     # {input_matrix: {optimizer_name: metrics}}
        self.pdf_filename = "training_report.pdf"
        self.threshold_drops = defaultdict(lambda: defaultdict(list))  # {input_matrix: {optimizer_name: [iteration, ...]}}
        self.stability_regions = defaultdict(lambda: defaultdict(list))  # {input_matrix: {optimizer_name: [(start_iter, end_iter), ...]}}
        self.heatmap_data = defaultdict(dict)                          # {input_matrix: {optimizer_name: np.array([...])}}
        self.convergence_iterations = defaultdict(lambda: defaultdict(lambda: {'90%': 'N/A', '99%': 'N/A'}))  # {input_matrix: {optimizer_name: {'90%': iter, '99%': iter}}}
        self.phase_values = defaultdict(lambda: defaultdict(list))      # {input_matrix: {optimizer_name: [phase_value, ...]}}

        # Separates for activation phases
        self.activation_distributions = {}  # {input_matrix: activation_phases}

    def set_initial_distribution(self, input_matrix, counts):
        """
        Sets the initial distribution (counts) for a given input matrix.
        Expects 'counts' to be a dictionary.
        """
        if isinstance(counts, dict):
            self.initial_distributions[input_matrix] = counts
        else:
            self.logger_error("set_initial_distribution expects a dictionary for counts.")

    def set_activation_phases(self, input_matrix, activation_phases):
        """
        Sets the activation phases for a given input matrix.
        Expects 'activation_phases' to be a list.
        """
        if isinstance(activation_phases, list):
            self.activation_distributions[input_matrix] = activation_phases
        else:
            self.logger_error("set_activation_phases expects a list for activation_phases.")

    def set_final_distribution(self, input_matrix, optimizer_name, counts):
        self.final_distributions[input_matrix][optimizer_name] = counts

    def record_probability(self, input_matrix, optimizer_name, iteration, probability):
        self.iteration_probs[input_matrix][optimizer_name].append((iteration, probability))

        # Check for threshold drops (e.g., below 90%)
        if probability < 0.9:
            self.threshold_drops[input_matrix][optimizer_name].append(iteration)

        # Check for convergence iterations
        if self.convergence_iterations[input_matrix][optimizer_name]['90%'] == 'N/A' and probability >= 0.9:
            self.convergence_iterations[input_matrix][optimizer_name]['90%'] = iteration
        if self.convergence_iterations[input_matrix][optimizer_name]['99%'] == 'N/A' and probability >= 0.99:
            self.convergence_iterations[input_matrix][optimizer_name]['99%'] = iteration

    def record_loss(self, input_matrix, optimizer_name, iteration, loss):
        self.loss_data[input_matrix][optimizer_name].append((iteration, loss))

    def record_optimizer_metrics(self, input_matrix, optimizer_name, metrics):
        self.optimizer_metrics[input_matrix][optimizer_name] = metrics

    def record_heatmap_data(self, input_matrix, optimizer_name, matrix):
        # Store only the final matrix for the heatmap
        self.heatmap_data[input_matrix][optimizer_name] = matrix

    def record_phase_value(self, input_matrix, optimizer_name, phase_value):
        self.phase_values[input_matrix][optimizer_name].append(phase_value)

    def generate_pdf(self, target_state):
        # Filenames for plots
        png_prob_filename = "probability_over_time.png"
        png_loss_filename = "loss_over_time.png"
        png_peak_prob_filename = "peak_probabilities.png"
        png_distribution_before_after = "prob_distribution_before_after.png"
        png_phase_values = "phase_values.png"
        png_final_bar_chart = "final_probability_bar_chart.png"
        png_heatmap_filename = "matrix_heatmap.png"
        png_convergence_speed = "convergence_speed.png"

        # Create plots
        self._create_probability_plot(target_state, png_prob_filename)
        self._create_loss_plot(png_loss_filename)
        self._create_peak_probability_table(png_peak_prob_filename)
        self._create_distribution_before_after(target_state, png_distribution_before_after)
        self._create_phase_values_plot(png_phase_values)
        self._create_final_bar_chart(png_final_bar_chart, target_state)
        self._create_heatmap(png_heatmap_filename)
        self._create_convergence_speed_plot(png_convergence_speed)

        # Generate PDF
        pdf = FPDF()
        
        # Title Page
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(0, 10, f"Training Report - Target State: {target_state}", ln=True)
        
        # Add creation date
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Generated on: {now_str}", ln=True)
        
        # Insert Probability Over Time Plot
        if os.path.exists(png_prob_filename):
            pdf.image(png_prob_filename, w=180)
            pdf.ln(10)
        
        # Insert Peak Probability Plot
        if os.path.exists(png_peak_prob_filename):
            pdf.image(png_peak_prob_filename, w=180)
            pdf.ln(10)
        
        # Describe the first two plots
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Input Data and Probability Over Time:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "The above plot illustrates how the probability of the target state increases over training iterations for each input matrix and optimizer. The accompanying bar chart shows the peak probability achieved by each matrix.")
        pdf.ln(5)
        
        # Second Page
        pdf.add_page()
        
        # Insert Distribution Before and After Training
        distribution_files = [f for f in os.listdir('.') if f.startswith("prob_distribution_before_after") and f.endswith(".png")]
        for file in distribution_files:
            if os.path.exists(file):
                pdf.image(file, w=180)
                pdf.ln(10)
        
        # Insert Phase Values Plot
        if os.path.exists(png_phase_values):
            pdf.image(png_phase_values, w=180)
            pdf.ln(10)
        
        # Insert Final Probability Bar Chart
        if os.path.exists(png_final_bar_chart):
            pdf.image(png_final_bar_chart, w=180)
            pdf.ln(10)
        
        # Describe the second page plots
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Probability Distributions and Phase Values:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "The above bar charts display the probability distributions before and after training for each input matrix and optimizer, along with the phase values recorded after each training iteration. The final probability bar chart summarizes the probabilities achieved by each matrix and optimizer combination after training.")
        pdf.ln(5)
        
        # Third Page
        pdf.add_page()
        
        # Insert Heatmap
        heatmap_files = [f for f in os.listdir('.') if f.startswith("matrix_heatmap") and f.endswith(".png")]
        for file in heatmap_files:
            if os.path.exists(file):
                pdf.image(file, w=180)
                pdf.ln(10)
        
        # Insert Convergence Speed Plot
        if os.path.exists(png_convergence_speed):
            pdf.image(png_convergence_speed, w=180)
            pdf.ln(10)
        
        # Describe the third page plots
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Heatmap of Final Matrices and Convergence Speed:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "The heatmap provides a visual representation of the final matrices for each input matrix and optimizer. The convergence speed plot shows the number of iterations required to achieve 90% and 99% probability thresholds.")
        pdf.ln(5)
        
        # Summary Page
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Arial", size=12)
        summary_text = "This report summarizes the training performance of various optimizers across different input matrices. It includes their final probabilities, loss values, convergence speeds, and phase values."
        pdf.multi_cell(0, 10, summary_text)
        pdf.ln(5)
        
        # Key Metrics Table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Key Metrics:", ln=True)
        pdf.set_font("Arial", size=12)
        summary_table = [["Input Matrix", "Optimizer", "Final Probability", "Iterations to 90%", "Iterations to 99%"]]
        for input_matrix in self.final_distributions.keys():
            for optimizer in self.final_distributions[input_matrix].keys():
                counts = self.final_distributions[input_matrix][optimizer]
                total = sum(counts.values())
                final_prob = counts.get(target_state, 0) / total if total > 0 else 0.0
                iter_90 = self.convergence_iterations[input_matrix][optimizer].get('90%', 'N/A')
                iter_99 = self.convergence_iterations[input_matrix][optimizer].get('99%', 'N/A')
                summary_table.append([input_matrix, optimizer, f"{final_prob:.3f}", str(iter_90), str(iter_99)])
        # Add summary table content
        for row in summary_table:
            for item in row:
                pdf.cell(35, 10, item, border=1)
            pdf.ln(10)
        
        # Best Optimizer Snapshot
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Best Optimizer Snapshot:", ln=True)
        pdf.set_font("Arial", size=12)
        if self.final_distributions:
            # Determine the best optimizer across all input matrices
            best_optimizer = None
            best_input_matrix = None
            best_prob = 0.0
            for input_matrix, optimizers in self.final_distributions.items():
                for optimizer, counts in optimizers.items():
                    total = sum(counts.values())
                    prob = counts.get(target_state, 0) / total if total > 0 else 0.0
                    if prob > best_prob:
                        best_prob = prob
                        best_optimizer = optimizer
                        best_input_matrix = input_matrix
            snapshot_text = f"The best optimizer is {best_optimizer} for input matrix '{best_input_matrix}' with a final probability of {best_prob:.3f}."
            pdf.multi_cell(0, 10, snapshot_text)
            specific_prob_filename = f"{png_prob_filename.replace('.png', '')}_{best_input_matrix}_{best_optimizer}.png"
            if os.path.exists(specific_prob_filename):
                pdf.image(specific_prob_filename, w=180)
        pdf.ln(5)
        
        # Legends and Descriptions
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Legends and Descriptions:", ln=True)
        pdf.set_font("Arial", size=12)
        legend_text = "All charts include detailed captions and legends for better interpretability."
        pdf.multi_cell(0, 10, legend_text)
        
        # Save PDF
        pdf.output(self.pdf_filename)
        print(f"PDF '{self.pdf_filename}' created.")
        
        # Remove temporary PNG files
        for file in [png_prob_filename, png_loss_filename, png_peak_prob_filename, png_distribution_before_after,
                    png_phase_values, png_final_bar_chart, png_heatmap_filename, png_convergence_speed]:
            # Also include distribution and heatmap files with matrix and optimizer names
            specific_files = [f for f in os.listdir('.') if f.startswith(file.split('.')[0])]
            for specific_file in specific_files:
                if os.path.exists(specific_file):
                    os.remove(specific_file)

    def _create_probability_plot(self, target_state, filename):
        plt.figure(figsize=(10, 6))
        for input_matrix, optimizers in self.iteration_probs.items():
            for optimizer_name, data in optimizers.items():
                iterations, probs = zip(*data)
                plt.plot(iterations, probs, label=f"{input_matrix} - {optimizer_name}")
        plt.title(f"Probability of Target State '{target_state}' Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_loss_plot(self, filename):
        plt.figure(figsize=(10, 6))
        for input_matrix, optimizers in self.loss_data.items():
            for optimizer_name, data in optimizers.items():
                iterations, losses = zip(*data)
                plt.plot(iterations, losses, label=f"{input_matrix} - {optimizer_name}")
        plt.title("Loss Function Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_peak_probability_table(self, filename):
        # Calculate peak probabilities
        peak_data = []
        for input_matrix, optimizers in self.iteration_probs.items():
            for optimizer_name, data in optimizers.items():
                peak_prob = max(prob for _, prob in data)
                peak_data.append([input_matrix, optimizer_name, f"{peak_prob:.3f}"])
        
        # Create a bar chart for peak probabilities
        labels = [f"{im} - {opt}" for im, opt, _ in peak_data]
        peak_probs = [float(prob) for _, _, prob in peak_data]
        x = np.arange(len(labels))
        plt.figure(figsize=(12, 6))
        plt.bar(x, peak_probs, color='skyblue')
        plt.xlabel('Input Matrix - Optimizer')
        plt.ylabel('Peak Probability')
        plt.title('Peak Probability for Each Input Matrix and Optimizer')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_distribution_before_after(self, target_state, filename):
        # This function creates bar charts for each input matrix and optimizer showing
        # probability distributions before and after training.
        for input_matrix, counts_before in self.initial_distributions.items():
            for optimizer_name, counts_after in self.final_distributions[input_matrix].items():
                if not isinstance(counts_before, dict) or not isinstance(counts_after, dict):
                    self.logger_error(f"Expected dictionaries for counts, got {type(counts_before)} and {type(counts_after)}")
                    continue

                labels = list(counts_before.keys())
                initial_probs = [counts_before.get(label, 0) / sum(counts_before.values()) for label in labels]
                final_probs = [counts_after.get(label, 0) / sum(counts_after.values()) for label in labels]

                x = np.arange(len(labels))  # label locations
                width = 0.35  # bar width

                plt.figure(figsize=(10, 6))
                plt.bar(x - width/2, initial_probs, width, label='Before Training')
                plt.bar(x + width/2, final_probs, width, label='After Training')
                plt.xlabel('State')
                plt.ylabel('Probability')
                plt.title(f"Probability Distribution Before and After Training for {input_matrix} - {optimizer_name}")
                plt.xticks(x, labels)
                plt.legend()
                plt.tight_layout()
                specific_filename = f"{filename.replace('.png', '')}_{input_matrix}_{optimizer_name}.png"
                plt.savefig(specific_filename)
                plt.close()

    def _create_phase_values_plot(self, filename):
        plt.figure(figsize=(10, 6))
        for input_matrix, optimizers in self.phase_values.items():
            for optimizer_name, phases in optimizers.items():
                iterations = range(1, len(phases) + 1)
                plt.plot(iterations, phases, label=f"{input_matrix} - {optimizer_name}")
        plt.title("Phase Values After Each Training Iteration")
        plt.xlabel("Training Iteration")
        plt.ylabel("Phase Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_final_bar_chart(self, filename, target_state):
        # Aggregate final probabilities
        labels = []
        final_probs = []
        for input_matrix, optimizers in self.final_distributions.items():
            for optimizer_name, counts in optimizers.items():
                total = sum(counts.values())
                prob = counts.get(target_state, 0) / total if total > 0 else 0.0
                labels.append(f"{input_matrix} - {optimizer_name}")
                final_probs.append(prob)
        
        x = np.arange(len(labels))
        plt.figure(figsize=(12, 6))
        plt.bar(x, final_probs, color='salmon')
        plt.xlabel('Input Matrix - Optimizer')
        plt.ylabel('Final Probability')
        plt.title('Final Probability per Input Matrix and Optimizer')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_heatmap(self, filename):
        # This function creates heatmaps for each input matrix and optimizer
        for input_matrix, optimizers in self.heatmap_data.items():
            for optimizer_name, matrix in optimizers.items():
                plt.figure(figsize=(8, 6))
                im = plt.imshow(matrix, cmap='viridis', aspect='auto')
                plt.title(f"Heatmap of Final Matrix for {input_matrix} - {optimizer_name}")
                plt.xlabel("Dimensions")
                plt.ylabel("Dimensions")
                plt.colorbar(im)
                plt.tight_layout()
                specific_filename = f"{filename.replace('.png', '')}_{input_matrix}_{optimizer_name}.png"
                plt.savefig(specific_filename)
                plt.close()

    def _create_convergence_speed_plot(self, filename):
        # This function creates convergence speed plots for each input matrix and optimizer
        for input_matrix, optimizers in self.convergence_iterations.items():
            for optimizer_name, iterations in optimizers.items():
                labels = ['90% Convergence', '99% Convergence']
                iter_values = [
                    iterations['90%'] if iterations['90%'] != 'N/A' else 0,
                    iterations['99%'] if iterations['99%'] != 'N/A' else 0
                ]
                
                x = np.arange(len(labels))
                width = 0.35

                plt.figure(figsize=(8, 6))
                plt.bar(x, iter_values, width, color=['blue', 'green'])
                plt.xlabel('Convergence Threshold')
                plt.ylabel('Iterations')
                plt.title(f'Convergence Speed for {input_matrix} - {optimizer_name}')
                plt.xticks(x, labels)
                plt.tight_layout()
                specific_filename = f"{filename.replace('.png', '')}_{input_matrix}_{optimizer_name}.png"
                plt.savefig(specific_filename)
                plt.close()

    def logger_error(self, message):
        """
        Simple error logger. You can expand this as needed.
        """
        print(f"Error: {message}")
