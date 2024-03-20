

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

    suscepted_indices  = set()
    infected_indices   = set()
    infected_snapshot  = []

    def create_network():
        if scalefree:
            return nx.scale_free_graph(N, avg_k)
        else:
            return nx.erdos_renyi_graph(N, avg_k / N)

    def initialize_infected_nodes():
        return set(np.random.choice(range(N), \
                int(start_infected * N), replace=False))

    def initialize_suscepted_nodes():
        infected_indices = initialize_infected_nodes()
        suscepted_indices = set(G.nodes())
        return suscepted_indices.difference(infected_indices)

    def search_infected_neighbors(suscepted_node_index):
        neighbor_indices = set(G.neighbors(suscepted_node_index))
        return neighbor_indices.intersection(infected_indices)

    def infection_prob(r):
        p = 1 - (1 - i) ** r
        return 1 if p <= np.random.rand() else 0

    def choose_suscepted_node():
        return np.random.choice(list(suscepted_indices))

    def infect_node(node_index):
        infected_neighbors_len = len(search_infected_neighbors(node_index))
        if infection_prob(infected_neighbors_len):
            infected_indices.add(node_index)
            suscepted_indices.remove(node_index)

    def create_snapshot():
        infected_snapshot.append(len(infected_indices))

    G = create_network()
    suscepted_indices = initialize_suscepted_nodes()

    for _ in range(time_steps):
        suscepted_node_index = choose_suscepted_node()
        while search_infected_neighbors(suscepted_node_index) == set():
            suscepted_node_index = choose_suscepted_node()
        infect_node(suscepted_node_index)
        create_snapshot()
    return infected_snapshot





        

