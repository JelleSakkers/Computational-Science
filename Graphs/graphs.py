###############################################################################
#                                                                             #
#                              GRAPHS ASSIGNMENT                              #
#                      INTRODUCTION COMPUTATIONAL SCIENCE                     #
#                                    2022                                     #
#                                                                             #
###############################################################################
#                                SCRIPT DETAILS                               #
###############################################################################
# This script accompanies the Graphs Jupyter Notebook and is used for         #
# functions that are a little complex for a Notebook, or require an execution #
# time that is not suitable for a Notebook.                                   #
#                                                                             #
# Run python3.8 graphs.py -h for a full description of options to run this    #
# script.                                                                     #
#                                                                             #
# You are free (and even encouraged) to write your own helper functions in    #
# this file. Just make sure that you keep the provided functions (with the    #
# same parameters).                                                           #
###############################################################################
#                               STUDENT DETAILS                               #
###############################################################################
# Name: Jelle Sakkers                                                         #
# UvANetID: 14619946                                                          #
###############################################################################
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Use this PRNG to generate random numbers
rng = np.random.default_rng()


def example_graph() -> None:
    """This is an example of how to create a random graph with N = 10⁵ and
    <k> = 1 and print all of the first nodes its neighbors.

    For more information, take a look at the NetworkX documentation:
    https://networkx.org/documentation/stable/reference/
    """
    # Create a graph  with N = 10⁵  and <k> = 1.0
    G = nx.fast_gnp_random_graph(10 ** 5, 1.0 / 10 ** 5)

    # Print all neighbors of the first node
    for neighbor in G.neighbors(0):
        print(neighbor)


def scalefree_graph(N: int, avg_k: float, gamma: float = 2.5) -> nx.Graph:
    """The package NetworkX does not have a method to directly create a
    scale-free graph itself, but we can create our own using this function.
    It draws a number of degrees from the power law distribution and uses them
    to create a random graph.

    :param N:     Amount of nodes in the graph
    :param avg_k: Average degree in the graph
    :param gamma: Exponent for power law sequence
    :return:      The scale-free graph
    """
    # Generate sequence of expected degrees, one for each vertex, with
    # p(k) ~ k^(-γ)
    degrees = nx.utils.powerlaw_sequence(N, gamma)
    # rescale the expected degrees to match the given average degree
    degrees = degrees / np.mean(degrees) * avg_k

    return nx.expected_degree_graph(degrees, selfloops=False)


def sand_avalanche(time_steps: int, sand_lost: float, scalefree: bool = False) -> np.ndarray:
    """
    Simulate the sand avalanche process on a random or scale-free network.

    :param time_steps: The number of time steps to simulate.
    :param sand_lost: The fraction of sand grains lost in transfer each time step.
    :param scalefree: If True, use a scale-free graph, otherwise use a random graph.
    :return: A 1D numpy array containing the number of toppled buckets per time step.
    """
    N = 10 ** 3
    K = 2
    Pk = K / N

    if scalefree:
        G = nx.scale_free_graph(N, K)
    else:
        G = nx.erdos_renyi_graph(N, Pk)

    bucket = np.zeros(N, dtype=int)
    degrees = np.array([G.degree[node] for node in G.nodes])
    aval = np.zeros(time_steps)

    def random_stable_nodes():
        i = np.where(bucket < degrees)[0]
        if len(i) == 0:
            return None
        return np.random.choice(i)

    def neighbor_stable_nodes(n):
        return [neighbor for neighbor in list(G.neighbors(n)) \
                if bucket[neighbor] < degrees[neighbor]]

    def count_toppled_buckets(node):
        return sum(1 for neighbor in G.neighbors(node) if \
                bucket[neighbor] >= degrees[neighbor])

    def add_by_chance(n):
        if rng.uniform(0, 1) > sand_lost:
            bucket[n] += 1

    for t in range(time_steps):
        node = random_stable_nodes()
        if node == None:
            break
        add_by_chance(node)
        if bucket[node] >= degrees[node]:
            neighbors = neighbor_stable_nodes(node)
            for neighbor in neighbors:
                add_by_chance(neighbor)
        aval[t] = count_toppled_buckets(node)
    return aval


def plot_avalanche_distribution(scalefree: bool, bins: int = 20, show: bool = False) -> None:
    """This function should run the simulation described in section 1.3 and
    plot an avalanche distribution based on the data retrieved.

    :param scalefree: If true a scale-free network is used, otherwise a random
                      network is used.
    :param bins:      Number of bins for the histogram.
    :param show:      If true the plot is also shown in addition to being
                      stored as png.
    """
    avalanche_sizes = sand_avalanche(10 ** 4, 10 ** -4, scalefree=scalefree)

    unique_sizes, counts = np.unique(avalanche_sizes, return_counts=True)
    probabilities = counts / len(avalanche_sizes)
 
    plt.figure()
    plt.hist(avalanche_sizes, bins=bins, density=False, \
            label='Simulated Distribution', alpha=0.7)

    plt.xlabel('number of toppled buckets $s$')
    plt.ylabel('probability $p$')
    plt.title(f'Avalanche Distribution ({scalefree if scalefree else random})')
    plt.grid(True)

    # Set logarithmic y-scale
    plt.yscale('log')

    # Use different filename if random or scale-free is used.
    filename = f"1-3{'b' if scalefree else 'a'}.png"
    plt.savefig(filename)
    if show:
        plt.show()

#########################
# HELPER FUNCTIONS HERE #
#########################


def susceptible_infected(N: int, avg_k: float, i: float, time_steps: int,
                         scalefree: bool = False, avg_degree: bool = False,
                         start_infected: float = 0.001) -> np.ndarray:
    """This function should run the simulation described in section 2.1 using a
    random network.

    :param N:              The amount of nodes in the network.
    :param avg_k:          The average degree of the nodes.
    :param i:              The probability that a node will infect a neighbor.
    :param time_steps:     The amount of time steps his simulate.
    :param scalefree:      If true a scale-free network should be used, if
                           false a random network should be used. Can be
                           ignored until question 2.4a.
    :param avg_degree:     If true the average degree of the infected nodee
                           per time step should be returned instead of the
                           amount of infected nodes. Can be ignored until
                           question 2.4c.
    :param start_infected: Fraction of nodes that will be infected from the
                           start.
    :return:               1D numpy array containing the amount of infected
                           nodes per time step. (So not normalised.)
    """
    def create_network():
        return nx.erdos_renyi_graph(N, avg_k / N)

    def retrieve_infected(neighbors):
        return [neighbor for neighbor in neighbors \
                if neighbor in infected_nodes]

    def retrieve_susceptible(neighbors):
        return infected_nodes - {node for node in neighbors}

    def update_infection():
        count = 0
        for node in infected_nodes:
            neighbors = list(G.neighbors(node))
            infected_amount = len(retrieve_infected(neighbors))
            infection_prob = prob(infected_amount)
            susceptible_nodes = retrieve_susceptible(neighbors)
            for node in susceptible_nodes:
                if np.random.rand() < infection_prob:
                    infected_nodes.add(node)
                    count += 1
        return count

    def prob(r):
        not_infected = (1 - i) ** r
        infected = 1 - not_infected
        return infected
    
    G = create_network()
    infected_nodes = set(np.random.choice(range(N), \
            int(start_infected * N), replace=False))
    infected = np.zeros(time_steps)
    
    for t in range(time_steps):
        infected[t] = update_infection()
    return infected

def plot_normalised_prevalence_random(start: bool, show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings described in 2.1 b. It should then produce a plot with the
    normalised prevalence over time for the two different settings. Make sure
    to average your result over multiple runs.

    :param start: If true the simulation should be run for the first 50 time
                  steps, if false it should be run for 500 time steps.

    :param show:  If true the plot is also shown in addition to being stored
                  as png.
    """
    fig = plt.figure(figsize=(7, 5))
    t = susceptible_infected(10 ** 5, 5.0, 0.01, 50)
    print(t)

    fig.savefig(f"2-1b-{'start' if start else 'full'}.png")
    if show:
        plt.show()


def plot_approximate_R0(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings described in 2.1 b. It should then produce a plot with the
    approximate R0 over time for the two different settings. Make sure
    to average your result over multiple runs.

    :param show: If true the plot is also shown in addition to being stored
                 as png.
    """
    fig = plt.figure(figsize=(7, 5))

    # YOUR PLOTTING CODE HERE

    fig.savefig("2-1h.png")
    if show:
        plt.show()


def euler(f1: callable, f2: callable, I0: float, S0: float, t0: float,
          t_end: float, dt: float) -> tuple:
    """Euler method to estimate I(t) and S(t) given f1(I, S) and f2(I, S)
    with initial value I0 and S0 from t=t0, until t=t_end with a time step of
    dt.
    :param f1:    dI/dt, callable as f1(I, S).
    :param f2:    dS/dt, callable as f2(I, S).
    :param I0:    Initial value of I(t0).
    :param S0:    Initial value of S(t0).
    :param t0:    First time step to calculate.
    :param t_end: Last time step to calculate.
    :param dt:    Time between time steps.
    :return:      3 numpy arrays, corresponding to t, I(t) and S(t)
                  respectively.
    """
    t = np.arange(t0, t_end, dt)
    I = []
    S = []

    # YOUR CODE HERE

    return t, np.array(I), np.array(S)


def approximate_ODE(case: int, t0: float, t_end: float, dt: float) -> tuple:
    """Should run the euler function to approximate the ODE described in
    section 2.2 with the settings described in 2.1 b."""
    if case == 1:
        b = ...  # enter your solution of b in case (i) here
    elif case == 2:
        b = ...  # enter your solution of b in case (ii) here
    else:
        raise ValueError(f"Case must be 1 or 2, not {case}")

    # ODE functions described in section 2.2
    def f1(I, S):
        return (1 - (1 - b) ** I) * S

    def f2(I, S):
        return -(1 - (1 - b) ** I) * S

    # Approximate the solution of the ODE using Euler
    t, I, _S = ...  # YOUR CODE HERE

    return t, I


def plot_compare_ode_simulation(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings described in 2.1 b and approximate the same result using the
    ODE described in section 2.2 and the Euler function. It should then produce
    a plot with all four curves in one figure.

    :param show: If true the plot is also shown in addition to being stored
                 as png.
    """
    fig = plt.figure(figsize=(7, 5))

    # YOUR PLOTTING CODE HERE

    fig.savefig("2-3a.png")
    if show:
        plt.show()


def plot_normalised_prevalence_scalefree(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings of the case for which a giant component exists described in
    2.1 b. The simulation should be repeated using both a random and scale-free
    network. It should then produce a plot with the normalised prevalence over
    time for the two different networks. Make sure to average your result over
    multiple runs.

    :param show:  If true the plot is also shown in addition to being stored
                  as png.
    """
    fig = plt.figure(figsize=(7, 5))

    # YOUR PLOTTING CODE HERE

    fig.savefig(f"2-4a.png")
    if show:
        plt.show()


def plot_average_degree(show: bool = False) -> None:
    """This function should run the simulation described in section 2.1 with
    the settings of the case for which a giant component exists described in
    2.1 b and keep track of the average degrees of the newly infected nodes at
    every time step. The simulation should be repeated using both a random and
    scale-free network. It should then produce a plot with the average degree
    over time for the two different networks. Make sure to average your result
    over multiple runs.

    :param show:  If true the plot is also shown in addition to being stored
                  as png.
    """
    fig = plt.figure(figsize=(7, 5))

    # YOUR PLOTTING CODE HERE

    fig.savefig(f"2-4c.png")
    if show:
        plt.show()


if __name__ == '__main__':
    import argparse
    from argparse import RawTextHelpFormatter
    from sys import stderr

    # Handle command-line arguments
    assignments = ['1.3', '1.3a', '1.3b', '2.1', '2.1b', '2.1h',
                   '2.3', '2.3a', '2.4', '2.4a', '2.4c']
    description = "Run simulations and generate plots for graphs exercise."

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description=description)
    parser.add_argument('-s', '--show', action='store_true',
                        help="Also show the plots in a GUI in addition to "
                             "storing them as png.")
    parser.add_argument('assignments', type=str, nargs='*',
                        choices=assignments, metavar='ASSIGNMENT',
                        help="Assignments to execute. If left out, "
                             "all assignments will be executed.\n"
                             "Choices are:\n" +
                             '\n'.join([f"'{a}'" for a in assignments]))

    args = parser.parse_args()

    if not args.assignments:
        parser.print_usage(stderr)
        print(f"Run {parser.prog} -h for more info", file=stderr)
        print("\nRunning all assignments...", file=stderr)

    # Run selected assignments
    # Plot random avalanche distribution
    if not args.assignments or '1.3' in args.assignments \
            or '1.3a' in args.assignments:
        print("Running assignment 1.3a", file=stderr)
        plot_avalanche_distribution(False, show=args.show)

    # Plot scale-free avalanche distribution
    if not args.assignments or '1.3' in args.assignments \
            or '1.3b' in args.assignments:
        print("Running assignment 1.3b", file=stderr)
        plot_avalanche_distribution(True, show=args.show)

    # Plot normalised prevalence
    if not args.assignments or '2.1' in args.assignments \
            or '2.1b' in args.assignments:
        print("Running assignment 2.1b", file=stderr)
        plot_normalised_prevalence_random(True, show=args.show)
        plot_normalised_prevalence_random(False, show=args.show)

    # Plot approximated R0
    if not args.assignments or '2.1' in args.assignments \
            or '2.1h' in args.assignments:
        print("Running assignment 2.1h", file=stderr)
        plot_approximate_R0(show=args.show)

    # Plot comparison between simulation and ODE
    if not args.assignments or '2.3' in args.assignments \
            or '2.3a' in args.assignments:
        print("Running assignment 2.3a", file=stderr)
        plot_compare_ode_simulation(show=args.show)

    # Plot comparison between scale-free and random network
    if not args.assignments or '2.4' in args.assignments \
            or '2.4a' in args.assignments:
        print("Running assignment 2.4a", file=stderr)
        plot_normalised_prevalence_scalefree(show=args.show)

    # Plot comparison between scale-free and random network
    if not args.assignments or '2.4' in args.assignments \
            or '2.4c' in args.assignments:
        print("Running assignment 2.4c", file=stderr)
        plot_average_degree(show=args.show)
