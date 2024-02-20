import numpy as np
import matplotlib.pyplot as plt

from ca import TableWalkThrough, CASim


def run_simulations(simulator, rule_builder):
    """
    Run simulations using the specified simulator and rule_builder.
    """
    def initialize_simulation():
        """
        Initialize transient_lens and seen dictionaries.
        """
        transient_lens = []
        seen = {}
        return transient_lens, seen

    def simulate():
        """
        Run the simulation and track transient lengths.
        """
        transient_lens, seen = initialize_simulation()
        transient_len = 0

        for _ in range(simulator.height):
            key = hash_key(simulator.config[simulator.t])
            if key in seen:
                transient_len = seen[key]
                break
            seen[key] = simulator.t
            simulator.step()
        return transient_len

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

    # Set up simulation parameters
    simulation_range = np.arange(0.10, 1.0, 0.125)
    simulator.height = 10 ** 4

    transient_lens = []

    # Run simulations for specified range
    for threshold in simulation_range:
        simulator.rule_set = rule_builder.walk_through(threshold)
        simulator.reset()
        transient_len = simulate()
        transient_lens.append(transient_len)
    return transient_lens
