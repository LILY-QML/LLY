# visual.py
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import datetime

class Visual:
    """
    Diese Klasse sammelt Daten über mehrere Optimizer, 
    deren Trainingsschritte und Wahrscheinlichkeiten 
    und erstellt am Schluss ein PDF mit Diagramm und Zusammenfassung.
    """

    def __init__(self):
        # Dictionary zum Speichern der Zielzustands-Wahrscheinlichkeit:
        # { "SGDOptimizer": [(iteration, prob), (iteration, prob), ...],
        #   "AdamOptimizer": [...],
        #   ... }
        self.iteration_probs = {}

        # Dictionaries für Verteilungen: pro Optimizer
        self.initial_distribution = None     # Anfangsverteilung (wenn für alle gleich)
        self.final_distributions = {}        # {optimizer_name: { ...counts... }}

        # Name der PDF-Datei
        self.pdf_filename = "training_report.pdf"

    def set_initial_distribution(self, counts):
        """Speichert die gemeinsame Anfangs-Verteilung (für alle Optimizer)."""
        self.initial_distribution = counts

    def set_final_distribution(self, optimizer_name, counts):
        """Speichert die End-Verteilung für einen bestimmten Optimizer."""
        self.final_distributions[optimizer_name] = counts

    def record_probability(self, optimizer_name, iteration, probability):
        """
        Zeichnet die Zielzustands-Wahrscheinlichkeit pro Iteration auf
        - pro Optimizer (z.B. "AdamOptimizer", "SGDOptimizer").
        """
        if optimizer_name not in self.iteration_probs:
            self.iteration_probs[optimizer_name] = []
        self.iteration_probs[optimizer_name].append((iteration, probability))

    def generate_pdf(self, target_state):
        """
        Erzeugt ein PDF, das den Verlauf der Wahrscheinlichkeit 
        für jeden Optimizer zeigt und die Anfangs-/End-Verteilungen ausgibt,
        sowie einen Hinweis, welcher Optimizer zum Schluss 
        die höchste Probability erreicht hat.
        """
        # 1) Erstes Diagramm (PNG), in dem alle Optimizer-Linien gezeichnet werden
        png_filename = "prob_evolution.png"
        self._create_plot(target_state, png_filename)

        # 2) Erstelle ein PDF
        pdf = FPDF()
        pdf.add_page()

        # Titel
        pdf.set_font("Arial", size=16)
        pdf.cell(0, 10, f"Training Report - Target State: {target_state}", ln=True)

        # Datum
        pdf.set_font("Arial", size=12)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 10, f"Generated at: {now_str}", ln=True)

        # Diagramm einfügen
        if os.path.exists(png_filename):
            pdf.image(png_filename, w=180)  # w=180 mm (breit), H automatisch
            pdf.ln(10)

        # Anfangs-Verteilung
        if self.initial_distribution is not None:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Initial Distribution (common):", ln=True)
            pdf.set_font("Arial", size=12)
            line_str = ", ".join([f"{k}: {v}" for k, v in self.initial_distribution.items()])
            pdf.multi_cell(0, 10, line_str)
            pdf.ln(5)

        # End-Verteilung pro Optimizer
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Final Distributions per Optimizer:", ln=True)
        pdf.set_font("Arial", size=12)

        # Wir berechnen nebenbei, welcher Optimizer das beste End-p hat
        best_optimizer = None
        best_prob = -1.0

        for optimizer_name, dist in self.final_distributions.items():
            # Bilde Summen
            total = sum(dist.values())
            prob_target = dist.get(target_state, 0) / total if total != 0 else 0.0
            if prob_target > best_prob:
                best_prob = prob_target
                best_optimizer = optimizer_name

            line_str = (f"{optimizer_name}: " + 
                        ", ".join([f"{k}: {v}" for k, v in dist.items()]) +
                        f" [p={prob_target:.3f}]")
            pdf.multi_cell(0, 10, line_str)
            pdf.ln(2)

        # Beste(r) Optimizer
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Best Optimizer at final iteration: {best_optimizer} (prob={best_prob:.3f})", ln=True)

        pdf.output(self.pdf_filename)
        print(f"\nPDF '{self.pdf_filename}' wurde erstellt.")

        # Optional: PNG löschen
        if os.path.exists(png_filename):
            os.remove(png_filename)

    def _create_plot(self, target_state, png_filename):
        """
        Zeichnet den Verlauf der Zielzustands-Wahrscheinlichkeit 
        für alle Optimizer in EINEM Diagramm.
        Linien statt Punkte, X-Achse breiter.
        """
        import matplotlib.pyplot as plt

        if not self.iteration_probs:
            return  # Keine Daten -> kein Plot

        plt.figure(figsize=(10, 4))  # breiteres Format

        # Wir erzeugen pro Optimizer eine eigene Linie
        for optimizer_name, data in self.iteration_probs.items():
            # data = [(iteration, prob), ...]
            if not data:
                continue
            iterations = [x[0] for x in data]
            probs = [x[1] for x in data]

            plt.plot(iterations, probs, 
                     label=optimizer_name, 
                     linestyle='-', marker='', linewidth=2)

        plt.title(f"Probability of target state '{target_state}' over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Probability")
        plt.grid(True)
        plt.ylim([0, 1.0])
        plt.legend()
        plt.savefig(png_filename, bbox_inches='tight')
        plt.close()
