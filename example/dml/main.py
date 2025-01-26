# example/dml/main.py

import os
import sys

# Füge den Pfad zum Hauptverzeichnis hinzu, damit die `lly`-Module gefunden werden können
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, parent_dir)

from src.dml import DML  # Importiere die DML-Klasse aus src/dml.py

def main():
    # Pfad zur data.json Datei
    data_json_path = os.path.join(current_dir, 'var', 'data.json')

    # Initialisiere die DML-Klasse (führt automatisch `check` aus)
    dml = DML(data_path=data_json_path)

    # Erstelle den Circuit und die Trainings-Phasengatter-Matrix
    circuit, training_phases_matrix = dml.create()

    if circuit is None:
        print("Fehler beim Erstellen des Circuit.")
        return

    # Ausgabe des Circuit
    print("\nErstellter Circuit:")
    print(circuit)  # Passe dies an die tatsächliche Implementierung der Circuit-Klasse an

    # Ausgabe der generierten Trainings-Phasengatter-Matrix
    print("\nGenerierte Trainings-Phasengatter-Matrix:")
    for qubit_idx, qubit_gates in enumerate(training_phases_matrix):
        print(f"Qubit {qubit_idx + 1}:")
        for gate_idx, gate in enumerate(qubit_gates):
            print(f"  Gate {gate_idx + 1}: {gate}")

    # Führe die Optimierung durch
    optimizer_type = 'adam'  # Beispiel: 'adam'
    optimizer_params = {'learning_rate': 0.001}  # Beispiel-Parameter für Adam
    shots = 10000
    iterations = 100
    end_value = 0.95  # Optionaler Schwellenwert

    print("\nStarte Optimierungsprozess...")
    dml.optimize(
        optimizer=optimizer_type,
        optimized_param=optimizer_params,
        shots=shots,
        iterations=iterations,
        end_value=end_value
    )

    # Nach der Optimierung, lade die optimierten Phasen erneut
    dml.load_data()
    circuit, training_phases_matrix = dml.create()

    if circuit is None:
        print("Fehler beim Erstellen des Circuit nach der Optimierung.")
        return

    # Ausgabe des optimierten Circuit
    print("\nOptimierter Circuit:")
    print(circuit)  # Passe dies an die tatsächliche Implementierung der Circuit-Klasse an

    # Ausgabe der optimierten Trainings-Phasengatter-Matrix
    print("\nOptimierte Trainings-Phasengatter-Matrix:")
    for qubit_idx, qubit_gates in enumerate(training_phases_matrix):
        print(f"Qubit {qubit_idx + 1}:")
        for gate_idx, gate in enumerate(qubit_gates):
            print(f"  Gate {gate_idx + 1}: {gate}")

if __name__ == "__main__":
    main()
