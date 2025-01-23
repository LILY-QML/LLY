from core.circuit import Circuit
from ml.optimizer import Optimizer
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
    # ================
    # 1) Ausgangs-Circuit für Target-State
    # ================
    training_phases_input = [
        [0.05, 0.15, 0.2],
        [0.1,  0.25, 0.3]
    ]
    activation_phases_input = [
        [0.02, 0.05, 0.08],
        [0.07, 0.09, 0.12]
    ]
    shots = 1000

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

    # Zielzustand bestimmen
    target_state = get_most_likely_outcome(input_counts)
    if target_state is None:
        print("Kein Zielzustand gefunden.")
        return
    print("Zielzustand:", target_state)

    # ================
    # 2) Visual-Objekt anlegen
    # ================
    visual = Visual()
    visual.set_initial_distribution(input_counts)

    # ================
    # 3) Mehrere Optimizer gegeneinander laufen lassen
    # ================
    optimizers_to_try = [
        "MomentumOptimizer",
        "AdamOptimizer"
    ]

    # Dieselben Start-Trainingsphasen
    training_phases = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    activation_phases = [
        [0.15, 0.25, 0.35],
        [0.45, 0.55, 0.65]
    ]
    max_iterations = 10000  # Kurze Demo

    # Für jeden Optimizer eine separate Instanz erstellen
    optimizer_instances = {}
    for opt_name in optimizers_to_try:
        print(f"\n=== Starte Training mit {opt_name} ===")
        my_optimizer = Optimizer(config_path="var")
        ret = my_optimizer.start(opt_name, target_state)  # Hier übergeben wir einen einzelnen String
        if ret is not None and "Error Code" in ret:
            print(f"Fehler beim Starten des Optimizers {opt_name}:", ret)
            continue

        optimizer_instances[opt_name] = my_optimizer

        current_training_matrix = training_phases.copy()  # Kopie erstellen, falls nötig

        # Schleife
        for iteration in range(1, max_iterations + 1):
            # Circuit aufbauen
            circuit = Circuit(
                qubits=2,
                depth=1,
                training_phases=current_training_matrix,
                activation_phases=activation_phases,
                shots=shots
            )
            circuit.measure_all()
            circuit.run()

            new_counts = circuit.get_counts()
            prob_tgt = probability_of_target(new_counts, target_state)

            # Im Visual => record_probability
            visual.record_probability(opt_name, iteration, prob_tgt)

            print(f"Iteration {iteration}, Wahrscheinlichkeit des Zielzustands: {prob_tgt * 100:.2f}%")

            # Abbrechen, wenn die Wahrscheinlichkeit über 98% liegt
            if prob_tgt >= 0.98:
                print(f"Abbruch: Wahrscheinlichkeit des Zielzustands {prob_tgt * 100:.2f}% überschreitet 98%.")
                break

            # Optimierung
            updated_matrix = my_optimizer.optimize(new_counts, current_training_matrix)
            if updated_matrix is None:
                print(f"Fehler bei der Optimierung in Iteration {iteration} für Optimizer {opt_name}.")
                break
            current_training_matrix = updated_matrix

        # Nach dem Training: finaler Circuit
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
        # Im Visual => final_distribution für diesen Optimizer
        visual.set_final_distribution(opt_name, final_counts)

    # ================
    # 4) PDF erzeugen
    # ================
    visual.generate_pdf(target_state)
    print("Fertig.")

if __name__ == "__main__":
    main()
