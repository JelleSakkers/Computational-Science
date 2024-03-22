

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

    infected_indices  = set()
    suscepted_indices = set()
    infected_snapshot = []

    # Create the graph based on the network type
    if scalefree:
        G = nx.scale_free_graph(N, avg_k)
    else:
        G = nx.erdos_renyi_graph(N, avg_k / N)

    # Initialize infected nodes
    infected_indices = rng.random.choice(range(N), 
            int(start_infected * N), replace=False)
    # Intialize suscepted nodes
    suscepted_indices = infected_indices.difference(G.nodes())
    
    for t in range(time_steps):
        # Retrieve a random stable node
       suscepted_node_idx = rng.random.choice(suscepted_indices)
        # Keep going until a stable node with infected neighbors is found
        while any((infected_neighbors := neighbor in infected_indices) \
            for neighbor in G.neighbors(suscepted_node_idx)):
            # Find a new candidate
            suscepted_node_idx = rng.random.choice(suscepted_indices)
        # Calculate infection probability based on amount of infected neighbors
        infection_prob = 1 - (1 - i) ** len(infected_neighbors)
        if rng.rand() <= infection_prob:
            # Add node to infected, and remove from suscepted
            infected_indices.add(suscepted_node_idx)
            suscepted_indices.remove(suscepted_node_idx)
        # Take a snapshot of the current amount of infected nodes
        infected_snapshot.append(len(infected_indices))
    return infected_indices



        
    

