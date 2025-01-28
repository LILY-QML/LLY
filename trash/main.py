# example/dml/main.py

import os
import sys
import threading  # Importiere threading, um den Hauptthread zu identifizieren

# Füge den Pfad zum Hauptverzeichnis hinzu, damit die `lly`-Module gefunden werden können
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, parent_dir)

# Debug-Ausgabe der sys.path (optional, zur Fehlerbehebung)
print("sys.path:")
for p in sys.path:
    print(p)

# Jetzt, nachdem sys.path angepasst wurde, importiere die Module
try:
    from lly.core.thread import ThreadMonitor  # Importiere ThreadMonitor aus lly.core.thread
    from src.dml import DML  # Importiere die DML-Klasse aus src/dml.py
except ModuleNotFoundError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    # Initialisiere den ThreadMonitor
    thread_monitor = ThreadMonitor()
    
    # Identifiziere den Hauptthread
    main_thread_id = threading.get_ident()
    
    # Logge den Start des Hauptthreads
    thread_monitor.log_thread_start(main_thread_id, "Main Execution")
    
    try:
        # Pfad zur data.json Datei
        data_json_path = os.path.join(current_dir, 'var', 'data.json')
    
        # Initialisiere die DML-Klasse (führt automatisch `check` aus)
        dml = DML(data_path=data_json_path)
    
        # Erstelle den Circuit und die Trainings-Phasengatter-Matrix
        circuit, training_phases_matrix = dml.create()
    
        # Setze die Trainingsphasen in der DML-Instanz
        dml.training_phases = training_phases_matrix  # **Wichtig: Setze die Trainingsphasen**
    
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
    
        # Messung der Trainingsmatrix mit einer Aktivierungsmatrix
        print("\nFühre Messung durch...")
        if dml.activation_matrices:
            activation_matrix = dml.activation_matrices[0]["matrix"]  # Beispiel: Erste Aktivierungsmatrix
            shots = 1000  # Anzahl der Shots für die Messung
            measurement_results = dml.measure(training_matrix=training_phases_matrix, shots=shots, activation_matrix=activation_matrix)
    
            # Ausgabe der Messresultate
            print("\nMessresultate:")
            for state, count in measurement_results.items():
                print(f"State: {state}, Count: {count}")
    
        # Speichern der Trainingsmatrix in einem Ordner
        save_folder = os.path.join(current_dir, 'saved_matrices')
        print(f"\nSpeichere Trainingsmatrix in: {save_folder}")
        dml.save_training_matrix(training_matrix=training_phases_matrix, folder_path=save_folder)
    
        # Führe die Optimierung durch
        optimizer_type = 'adam'  # Beispiel: 'adam'
        optimizer_params = {'learning_rate': 0.001}  # Beispiel-Parameter für Adam
        shots = 1000  # Anzahl der Shots für die Optimierung (angepasst von 10 auf 1000)
        iterations = 10
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
    
        # Setze die optimierten Trainingsphasen in der DML-Instanz
        dml.training_phases = training_phases_matrix  # **Wichtig: Setze die optimierten Trainingsphasen**
    
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
    
    finally:
        # Logge das Ende des Hauptthreads
        thread_monitor.log_thread_end(main_thread_id)
        
        # Generiere den PDF-Bericht der Thread-Aktivitäten
        thread_monitor.generate_pdf_report()
    
        # Optional: Ausgabe der Thread-Daten im Terminal
        print("\nCurrent Thread Activity Overview:")
        print(thread_monitor)

if __name__ == "__main__":
    main()
