import numpy as np
import matplotlib.pyplot as plt
from ca import TableWalkThrough

def run_experiments():
    k_value = 2
    r_value = 1
    num_experiments = 100

    table_walker = TableWalkThrough()
    
    entropies = []

    for _ in range(num_experiments):
        initial_rule_set = table_walker.build_initial_rule_set_to_sq()
        table_walker.rule_set = initial_rule_set

        table_walker.table_walk_through()

        entropy = table_walker.shannon_entropy()
        entropies.append(entropy)

    # Plot the results
    plt.hist(entropies, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Shannon Information Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    run_experiments()

