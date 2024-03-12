"""
- Naam: Jelle Sakkers
- Vak: Computational Science
- Studie: Bachelor Informatica
- Datum 21-2-2024
"""

import numpy as np
import matplotlib.pyplot as plt

from ca import TableWalkThrough, CASim


def run_simulations(simulator):
    """
    Run simulations using the specified simulator and rule_builder.
    """
    def initialize_simulation():
        """
        Initialize transient_lens and seen dictionaries.
        """
        rule_builder = TableWalkThrough(0.0, 2, 4)
        simulator_range = np.arange(0.0, 1.01, 0.10)
        simulator.height = 10 ** 4
        return rule_builder, simulator_range

    def simulate():
        """
        Run the simulation and track transient lengths.
        """
        transient_lengths = []

        for _ in range(10):
            seen = {}
            transient_len = 0

            for _ in range(simulator.height):
                key = hash_key(simulator.config[simulator.t])
                if key in seen:
                    transient_len = seen[key]
                    break
                seen[key] = simulator.t
                simulator.step()
            
            transient_lengths.append(transient_len)
            simulator.reset()

        average_transient_len = np.mean(transient_lengths)
        return average_transient_len

    def hash_key(config):
        """
        Convert the NumPy array to a hashable representation.
        """
        return tuple(config)

    def unhash_key(config):
        """
        Convert a byte representation of data to an integer.
        """
        return np.array(list(config))

    builder, simulator_range = initialize_simulation()
    transient_lens = []

    for lambda_prime in simulator_range:
        builder.set_t(lambda_prime)
        simulator.rule_set = builder.walk_through()
        simulator.reset()
        average_transient_len = simulate()
        transient_lens.append(average_transient_len)

    plt.plot(simulator_range, transient_lens, linewidth=0.5, label='Transient Lengths')
    plt.scatter(simulator_range, transient_lens, color='red', marker='s', label='Points')
    plt.xlabel('$λ$')
    plt.ylabel('$transient\ length$')
    plt.title("Average Transient Length")
    plt.legend()
    plt.show()
    return transient_lens

def main():
    sim = CASim()
    run_simulations(sim)

if __name__ == "__main__":
    main()
    
    
   
