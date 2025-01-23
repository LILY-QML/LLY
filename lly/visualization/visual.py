import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import datetime
import numpy as np
import psutil
import threading
import time
from collections import defaultdict

class Visual:
    """
    This class collects data from multiple optimizers, tracks their training progress,
    and generates a detailed PDF report with charts, tables, and metrics.
    """

    def __init__(self):
        self.iteration_probs = {}  # {optimizer_name: [(iteration, prob), ...]}
        self.loss_data = {}        # {optimizer_name: [(iteration, loss), ...]}
        self.initial_distribution = None
        self.final_distributions = {}
        self.optimizer_metrics = {}
        self.pdf_filename = "training_report.pdf"
        self.metrics_summary = {}
        self.thread_details = {}    # {optimizer_name: {'start_time': ..., 'end_time': ..., 'cpu_usage': ...}}
        self.threshold_drops = defaultdict(list)  # {optimizer_name: [iteration, ...]}
        self.stability_regions = defaultdict(list)  # {optimizer_name: [(start_iter, end_iter), ...]}
        self.heatmap_data = {}     # {optimizer_name: np.array([...])}
        self.thread_sync = []      # List of (optimizer_name, start_time, end_time)
        self.convergence_iterations = {}  # {optimizer_name: {'90%': iter, '99%': iter}}

    def set_initial_distribution(self, counts):
        self.initial_distribution = counts

    def set_final_distribution(self, optimizer_name, counts):
        self.final_distributions[optimizer_name] = counts

    def record_probability(self, optimizer_name, iteration, probability):
        if optimizer_name not in self.iteration_probs:
            self.iteration_probs[optimizer_name] = []
        self.iteration_probs[optimizer_name].append((iteration, probability))

        # Check for threshold drops (e.g., below 90%)
        if probability < 0.9:
            self.threshold_drops[optimizer_name].append(iteration)

        # Check for convergence iterations
        if optimizer_name not in self.convergence_iterations:
            self.convergence_iterations[optimizer_name] = {}
        if '90%' not in self.convergence_iterations[optimizer_name] and probability >= 0.9:
            self.convergence_iterations[optimizer_name]['90%'] = iteration
        if '99%' not in self.convergence_iterations[optimizer_name] and probability >= 0.99:
            self.convergence_iterations[optimizer_name]['99%'] = iteration

    def record_loss(self, optimizer_name, iteration, loss):
        if optimizer_name not in self.loss_data:
            self.loss_data[optimizer_name] = []
        self.loss_data[optimizer_name].append((iteration, loss))

    def record_optimizer_metrics(self, optimizer_name, metrics):
        self.optimizer_metrics[optimizer_name] = metrics

    def record_thread_detail(self, optimizer_name, start_time, end_time, cpu_usage):
        self.thread_details[optimizer_name] = {
            'start_time': start_time,
            'end_time': end_time,
            'cpu_usage': cpu_usage
        }

    def record_heatmap_data(self, optimizer_name, matrix):
        # Store only the final matrix for the heatmap
        self.heatmap_data[optimizer_name] = matrix

    def generate_pdf(self, target_state):
        png_prob_filename = "probability_over_time.png"
        png_loss_filename = "loss_over_time.png"
        png_bar_chart_filename = "optimizer_comparison.png"
        png_heatmap_filename = "matrix_heatmap.png"
        png_convergence_speed = "convergence_speed.png"
        png_thread_timeline = "thread_timeline.png"

        # Create plots
        self._create_probability_plot(target_state, png_prob_filename)
        self._create_loss_plot(png_loss_filename)
        self._create_optimizer_comparison(png_bar_chart_filename)
        self._create_heatmap(png_heatmap_filename)
        self._create_convergence_speed_plot(png_convergence_speed)
        self._create_thread_timeline_plot(png_thread_timeline)

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(0, 10, f"Training Report - Target State: {target_state}", ln=True)

        # Add creation date
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Generated on: {now_str}", ln=True)

        # Insert probability plot
        if os.path.exists(png_prob_filename):
            pdf.image(png_prob_filename, w=180)
            pdf.ln(10)

        # Insert loss plot
        if os.path.exists(png_loss_filename):
            pdf.image(png_loss_filename, w=180)
            pdf.ln(10)

        # Describe the plots
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Loss and Probability Plots:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "The above plots show the evolution of the target state probability and the loss function over iterations for each optimizer.")
        pdf.ln(5)

        # Training Progress Table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Training Progress Overview:", ln=True)
        pdf.set_font("Arial", size=12)
        table_data = [["Optimizer", "Initial Probability", "Max Probability", "Final Probability", "Iterations", "Convergence Time (s)"]]
        for optimizer, probs in self.iteration_probs.items():
            initial_prob = self.initial_distribution.get(target_state, 0) / sum(self.initial_distribution.values()) if self.initial_distribution else 0
            max_prob = max([prob for _, prob in probs])
            final_prob = probs[-1][1]
            iterations = len(probs)
            convergence_time = (self.thread_details.get(optimizer, {}).get('end_time', 0) - self.thread_details.get(optimizer, {}).get('start_time', 0))
            table_data.append([
                optimizer,
                f"{initial_prob:.3f}",
                f"{max_prob:.3f}",
                f"{final_prob:.3f}",
                str(iterations),
                f"{convergence_time:.2f}"
            ])
        # Add table content
        for row in table_data:
            for item in row:
                pdf.cell(40, 10, item, border=1)
            pdf.ln(10)

        pdf.ln(5)

        # Thread Details
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Thread Details:", ln=True)
        pdf.set_font("Arial", size=12)
        for optimizer, details in self.thread_details.items():
            start_time = datetime.datetime.fromtimestamp(details['start_time']).strftime("%Y-%m-%d %H:%M:%S")
            end_time = datetime.datetime.fromtimestamp(details['end_time']).strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = details['cpu_usage']
            pdf.multi_cell(0, 10, f"{optimizer}: Start Time = {start_time}, End Time = {end_time}, CPU Usage = {cpu_usage:.2f}%")
        pdf.ln(5)

        # Optimizer Metrics and Final Distributions
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Optimizer Metrics and Final Distributions:", ln=True)
        pdf.set_font("Arial", size=12)
        for optimizer_name, dist in self.final_distributions.items():
            metrics = self.optimizer_metrics.get(optimizer_name, {})
            total = sum(dist.values())
            prob_target = dist.get(target_state, 0) / total if total > 0 else 0.0
            pdf.multi_cell(0, 10, f"{optimizer_name}: Final Probability = {prob_target:.3f}, Metrics = {metrics}")
            pdf.multi_cell(0, 10, f"Distribution: {dist}")
            pdf.ln(5)

        # Best Optimizer
        if self.final_distributions:
            best_optimizer, best_dist = max(
                self.final_distributions.items(),
                key=lambda x: x[1].get(target_state, 0) / sum(x[1].values()) if sum(x[1].values()) > 0 else 0
            )
            best_prob = best_dist.get(target_state, 0) / sum(best_dist.values()) if sum(best_dist.values()) > 0 else 0.0
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Best Optimizer: {best_optimizer} with Probability {best_prob:.3f}", ln=True)

        pdf.ln(5)

        # Insert Convergence Speed Plot
        if os.path.exists(png_convergence_speed):
            pdf.image(png_convergence_speed, w=180)
            pdf.ln(10)

        # Threshold Drops
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Threshold Drops:", ln=True)
        pdf.set_font("Arial", size=12)
        for optimizer, iterations in self.threshold_drops.items():
            pdf.multi_cell(0, 10, f"{optimizer}: Probabilities dropped below 90% at iterations {iterations}")
        pdf.ln(5)

        # Stability Regions
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Stability Regions:", ln=True)
        pdf.set_font("Arial", size=12)
        for optimizer, regions in self.stability_regions.items():
            pdf.multi_cell(0, 10, f"{optimizer}: Stable over iterations {regions}")
        pdf.ln(5)

        # Insert Heatmap
        if os.path.exists(png_heatmap_filename):
            pdf.image(png_heatmap_filename, w=180)
            pdf.ln(10)

        # Insert Thread Timeline
        if os.path.exists(png_thread_timeline):
            pdf.image(png_thread_timeline, w=180)
            pdf.ln(10)

        # Summary Page
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Arial", size=12)
        summary_text = "This report summarizes the training performance of various optimizers, including their final probabilities, loss values, convergence speeds, and resource utilizations."
        pdf.multi_cell(0, 10, summary_text)
        pdf.ln(5)

        # Key Metrics Table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Key Metrics:", ln=True)
        pdf.set_font("Arial", size=12)
        summary_table = [["Optimizer", "Final Probability", "Iterations to 90%", "Iterations to 99%", "CPU Usage (%)"]]
        for optimizer in self.final_distributions.keys():
            final_prob = self.final_distributions[optimizer].get(target_state, 0) / sum(self.final_distributions[optimizer].values())
            iter_90 = self.convergence_iterations.get(optimizer, {}).get('90%', 'N/A')
            iter_99 = self.convergence_iterations.get(optimizer, {}).get('99%', 'N/A')
            cpu = self.thread_details.get(optimizer, {}).get('cpu_usage', 0.0)
            summary_table.append([optimizer, f"{final_prob:.3f}", str(iter_90), str(iter_99), f"{cpu:.2f}"])
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
            snapshot_text = f"The best optimizer is {best_optimizer} with a final probability of {best_prob:.3f}."
            pdf.multi_cell(0, 10, snapshot_text)
            if os.path.exists(png_prob_filename):
                pdf.image(png_prob_filename, w=180)
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
        for file in [png_prob_filename, png_loss_filename, png_bar_chart_filename, png_heatmap_filename, png_convergence_speed, png_thread_timeline]:
            if os.path.exists(file):
                os.remove(file)

    def _create_probability_plot(self, target_state, filename):
        plt.figure(figsize=(10, 6))
        for optimizer_name, data in self.iteration_probs.items():
            iterations, probs = zip(*data)
            plt.plot(iterations, probs, label=optimizer_name)
        plt.title(f"Probability of Target State '{target_state}' Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def _create_loss_plot(self, filename):
        if not self.loss_data:
            return
        plt.figure(figsize=(10, 6))
        for optimizer_name, data in self.loss_data.items():
            iterations, losses = zip(*data)
            plt.plot(iterations, losses, label=optimizer_name)
        plt.title("Loss Function Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def _create_optimizer_comparison(self, filename):
        if not self.optimizer_metrics:
            return
        labels = list(self.optimizer_metrics.keys())
        avg_probs = [np.mean([prob for _, prob in self.iteration_probs[opt]]) for opt in labels]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, avg_probs, color='skyblue')
        plt.title("Average Probability per Optimizer")
        plt.xlabel("Optimizer")
        plt.ylabel("Average Probability")
        plt.savefig(filename)
        plt.close()

    def _create_heatmap(self, filename):
        if not self.heatmap_data:
            return
        num_optimizers = len(self.heatmap_data)
        fig, axes = plt.subplots(num_optimizers, 1, figsize=(10, 6 * num_optimizers))
        if num_optimizers == 1:
            axes = [axes]
        for ax, (optimizer, matrix) in zip(axes, self.heatmap_data.items()):
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(f"Heatmap of Final Matrix for {optimizer}")
            ax.set_xlabel("Dimensions")
            ax.set_ylabel("Dimensions")
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _create_convergence_speed_plot(self, filename):
        if not self.convergence_iterations:
            return
        optimizers = list(self.convergence_iterations.keys())
        iter_90 = [self.convergence_iterations[opt].get('90%', np.nan) for opt in optimizers]
        iter_99 = [self.convergence_iterations[opt].get('99%', np.nan) for opt in optimizers]
        x = np.arange(len(optimizers))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, iter_90, width, label='90% Convergence')
        plt.bar(x + width/2, iter_99, width, label='99% Convergence')
        plt.xlabel('Optimizer')
        plt.ylabel('Iterations')
        plt.title('Iterations to Achieve 90% and 99% Probability')
        plt.xticks(x, optimizers)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def _create_thread_timeline_plot(self, filename):
        if not self.thread_details:
            return
        plt.figure(figsize=(10, 6))
        for idx, (optimizer, details) in enumerate(self.thread_details.items()):
            start = details['start_time']
            end = details['end_time']
            plt.hlines(idx, start, end, colors='b', linewidth=4)
            plt.plot(start, idx, 'go')  # Start point
            plt.plot(end, idx, 'ro')    # End point
            plt.text(end, idx, f' {optimizer}', verticalalignment='bottom')
        plt.xlabel('Time')
        plt.ylabel('Optimizer')
        plt.title('Thread Timeline')
        plt.yticks(range(len(self.thread_details)), list(self.thread_details.keys()))
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    # Additional methods to record stability regions can be added here

# Example Usage
if __name__ == "__main__":
    import random

    visual = Visual()

    # Example data
    visual.set_initial_distribution({"00": 50, "01": 20, "10": 15, "11": 15})
    visual.set_final_distribution("SGDOptimizer", {"00": 90, "01": 5, "10": 3, "11": 2})
    visual.set_final_distribution("AdamOptimizer", {"00": 95, "01": 3, "10": 1, "11": 1})

    # Recording probabilities
    visual.record_probability("SGDOptimizer", 1, 0.5)
    visual.record_probability("SGDOptimizer", 2, 0.6)
    visual.record_probability("SGDOptimizer", 3, 0.85)
    visual.record_probability("SGDOptimizer", 4, 0.92)
    visual.record_probability("SGDOptimizer", 5, 0.88)

    visual.record_probability("AdamOptimizer", 1, 0.55)
    visual.record_probability("AdamOptimizer", 2, 0.65)
    visual.record_probability("AdamOptimizer", 3, 0.75)
    visual.record_probability("AdamOptimizer", 4, 0.95)
    visual.record_probability("AdamOptimizer", 5, 0.99)

    # Recording losses
    visual.record_loss("SGDOptimizer", 1, 0.5)
    visual.record_loss("SGDOptimizer", 2, 0.4)
    visual.record_loss("SGDOptimizer", 3, 0.35)
    visual.record_loss("SGDOptimizer", 4, 0.3)
    visual.record_loss("SGDOptimizer", 5, 0.25)

    visual.record_loss("AdamOptimizer", 1, 0.45)
    visual.record_loss("AdamOptimizer", 2, 0.35)
    visual.record_loss("AdamOptimizer", 3, 0.25)
    visual.record_loss("AdamOptimizer", 4, 0.2)
    visual.record_loss("AdamOptimizer", 5, 0.15)

    # Recording optimizer metrics
    visual.record_optimizer_metrics("SGDOptimizer", {"learning_rate": 0.01, "momentum": 0.9})
    visual.record_optimizer_metrics("AdamOptimizer", {"learning_rate": 0.001, "epsilon": 1e-7})

    # Thread Details (Example)
    current_time = time.time()
    visual.record_thread_detail("SGDOptimizer", start_time=current_time, end_time=current_time + 30, cpu_usage=15.5)
    visual.record_thread_detail("AdamOptimizer", start_time=current_time + 5, end_time=current_time + 28, cpu_usage=12.3)

    # Heatmap Data (Only final matrix)
    final_matrix_sgd = np.random.rand(5, 5)
    final_matrix_adam = np.random.rand(5, 5)
    visual.record_heatmap_data("SGDOptimizer", final_matrix_sgd)
    visual.record_heatmap_data("AdamOptimizer", final_matrix_adam)

    # Generate PDF
    visual.generate_pdf("00")
