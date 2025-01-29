import os
import sys
import threading
import argparse
import logging
from logging.handlers import RotatingFileHandler

# ------------------------------------------------------------------------------
# Log-Verzeichnis einrichten
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'log')
os.makedirs(log_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Logger-Konfiguration: normal_logger
#     -> Schreibt alles, was nicht explizit als Training gekennzeichnet ist,
#        in log/log.log
# ------------------------------------------------------------------------------
normal_logger = logging.getLogger('normal_logger')
normal_logger.setLevel(logging.DEBUG)  # alles ab DEBUG-Level ins File

normal_handler = RotatingFileHandler(
    os.path.join(log_dir, 'log.log'),
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5
)
normal_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
normal_handler.setFormatter(normal_formatter)
normal_logger.addHandler(normal_handler)

# ------------------------------------------------------------------------------
# Logger-Konfiguration: training_logger
#     -> Alles, was mit Training/Optimierung zusammenhängt,
#        in log/training.log
# ------------------------------------------------------------------------------
training_logger = logging.getLogger('training_logger')
training_logger.setLevel(logging.DEBUG)  # alles ab DEBUG-Level ins File

training_handler = RotatingFileHandler(
    os.path.join(log_dir, 'training.log'),
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5
)
training_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
training_handler.setFormatter(training_formatter)
training_logger.addHandler(training_handler)

# ------------------------------------------------------------------------------
# Optional: Log auch auf der Konsole ausgeben (hier nur normal_logger)
# ------------------------------------------------------------------------------
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(message)s')
)
normal_logger.addHandler(console_handler)

# ------------------------------------------------------------------------------
# Pfad für Module ergänzen
# ------------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, parent_dir)

normal_logger.debug("Aktuelle sys.path:")
for p in sys.path:
    normal_logger.debug(p)

# ------------------------------------------------------------------------------
# Module importieren
# ------------------------------------------------------------------------------
try:
    from lly.core.thread import ThreadMonitor
    from src.dml import DML
except ModuleNotFoundError as e:
    normal_logger.error(f"ImportError: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Testmodus-Funktion
# ------------------------------------------------------------------------------
def run_test_mode():
    normal_logger.info("=== Testmodus aktiviert ===")

    test_data_path = os.path.join(current_dir, 'var', 'test_data.json')

    if not os.path.exists(test_data_path):
        test_data = {
            "activation_matrices": [
                {
                    "name": "TestMatrix1",
                    "matrix": [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6]
                    ],
                    "target_state": [1, 0]
                }
            ]
        }
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        with open(test_data_path, 'w') as f:
            import json
            json.dump(test_data, f, indent=4)
        normal_logger.info(f"Erstellte Testdaten unter {test_data_path}")

    dml = DML(data_path=test_data_path)

    if not dml.activation_matrices:
        normal_logger.error("Fehler: Aktivierungsmatrizen wurden nicht korrekt geladen.")
        return
    else:
        normal_logger.info("Test 1 bestanden: Aktivierungsmatrizen wurden korrekt geladen.")

    circuit, training_phases = dml.create()
    if circuit is None or training_phases is None:
        normal_logger.error("Fehler: Circuit konnte nicht erstellt werden.")
        return
    else:
        normal_logger.info("Test 2 bestanden: Circuit wurde erfolgreich erstellt.")

    measurement_results = dml.measure(
        training_matrix=training_phases,
        shots=100,
        activation_matrix=dml.activation_matrices[0]["matrix"]
    )
    if not measurement_results:
        normal_logger.error("Fehler: Messung hat keine Ergebnisse geliefert.")
        return
    else:
        normal_logger.info("Test 3 bestanden: Messung wurde erfolgreich durchgeführt.")

    test_save_folder = os.path.join(current_dir, 'test_saved_matrices')
    dml.save_training_matrix(training_matrix=training_phases, folder_path=test_save_folder)
    saved_files = os.listdir(test_save_folder)
    if not saved_files:
        normal_logger.error("Fehler: Trainingsmatrix wurde nicht gespeichert.")
        return
    else:
        normal_logger.info(f"Test 4 bestanden: Trainingsmatrix wurde erfolgreich in '{test_save_folder}' gespeichert.")

    # --------------------------------------------------------------------------
    # Ab hier: Logs zur Optimierung / "Training" werden in den training_logger geschrieben
    # --------------------------------------------------------------------------
    training_logger.info("Starte eine schnelle Optimierung für Testzwecke...")
    dml.optimize(
        optimizer='adam',
        optimized_param={'learning_rate': 0.001},
        shots=1000,
        iterations=10000,
        end_value=0.95
    )
    training_logger.info("Test 5 abgeschlossen: Optimierung durchgeführt.")

    pdf_report = dml.visual.pdf_filename
    if os.path.exists(pdf_report):
        training_logger.info(f"Test 6 bestanden: PDF-Bericht '{pdf_report}' wurde erfolgreich erstellt.")
    else:
        normal_logger.error("Fehler: PDF-Bericht wurde nicht erstellt.")

    try:
        os.remove(test_data_path)
        import shutil
        shutil.rmtree(test_save_folder)
        normal_logger.info("Bereinigung abgeschlossen: Testdaten und temporäre Dateien wurden entfernt.")
    except Exception as e:
        normal_logger.warning(f"Warnung: Fehler bei der Bereinigung der Testdaten: {e}")

    normal_logger.info("=== Testmodus abgeschlossen ===")

# ------------------------------------------------------------------------------
# Hauptprogramm
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DML Hauptprogramm mit optionalem Testmodus.")
    parser.add_argument('-t', '--test', action='store_true', help='Führe das Programm im Testmodus aus.')
    args = parser.parse_args()

    if args.test:
        run_test_mode()
        sys.exit(0)

    thread_monitor = ThreadMonitor()
    main_thread_id = threading.get_ident()
    thread_monitor.log_thread_start(main_thread_id, "Main Execution")

    try:
        data_json_path = os.path.join(current_dir, 'var', 'data.json')
        dml = DML(data_path=data_json_path)
        circuit, training_phases_matrix = dml.create()

        if circuit is None:
            normal_logger.error("Fehler beim Erstellen des Circuit.")
            sys.exit(1)

        normal_logger.info("\nErstellter Circuit:")
        normal_logger.info(circuit)

        normal_logger.info("\nGenerierte Trainings-Phasengatter-Matrix:")
        for qubit_idx, qubit_gates in enumerate(training_phases_matrix):
            normal_logger.info(f"Qubit {qubit_idx + 1}:")
            for gate_idx, gate in enumerate(qubit_gates):
                normal_logger.info(f"  Gate {gate_idx + 1}: {gate}")

        if dml.activation_matrices:
            activation_matrix = dml.activation_matrices[0]["matrix"]
            shots = 1000
            measurement_results = dml.measure(
                training_matrix=training_phases_matrix,
                shots=shots,
                activation_matrix=activation_matrix
            )

            normal_logger.info("\nMessresultate:")
            for state, count in measurement_results.items():
                normal_logger.info(f"State: {state}, Count: {count}")

        save_folder = os.path.join(current_dir, 'saved_matrices')
        normal_logger.info(f"\nSpeichere Trainingsmatrix in: {save_folder}")
        dml.save_training_matrix(training_matrix=training_phases_matrix, folder_path=save_folder)

        optimizer_type = 'adam'
        optimizer_params = {'learning_rate': 0.001}
        shots = 1000
        iterations = 10000
        end_value = 0.95

        # ----------------------------------------------------------------------
        # Ab hier: Logs zur Optimierung / "Training" gehen in den training_logger
        # ----------------------------------------------------------------------
        training_logger.info("\nStarte Optimierungsprozess...")
        dml.optimize(
            optimizer=optimizer_type,
            optimized_param=optimizer_params,
            shots=shots,
            iterations=iterations,
            end_value=end_value
        )

        dml.load_data()
        circuit, training_phases_matrix = dml.create()

        if circuit is None:
            normal_logger.error("Fehler beim Erstellen des Circuit nach der Optimierung.")
            sys.exit(1)

        training_logger.info("\nOptimierter Circuit:")
        training_logger.info(circuit)

        training_logger.info("\nOptimierte Trainings-Phasengatter-Matrix:")
        for qubit_idx, qubit_gates in enumerate(training_phases_matrix):
            training_logger.info(f"Qubit {qubit_idx + 1}:")
            for gate_idx, gate in enumerate(qubit_gates):
                training_logger.info(f"  Gate {gate_idx + 1}: {gate}")

    finally:
        thread_monitor.log_thread_end(main_thread_id)
        thread_monitor.generate_pdf_report()
        normal_logger.info("\nCurrent Thread Activity Overview:")
        normal_logger.info(thread_monitor)

if __name__ == "__main__":
    main()
