import numpy as np
import matplotlib.pyplot as plt

from pyics import Model


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""
    if n == 0:
        return [0]
    res = []
    while n:
        res += [int(n % k)]
        n //= k
    return res[::-1]


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param('r', 2)
        self.make_param('k', 4)
        self.make_param('width', 128)
        self.make_param('height', 200)
        self.make_param('lambda_prime', 0.20)
        self.make_param('rule', 30, setter=self.setter_rule)

        self.rule_builder = TableWalkThrough(self.lambda_prime, self.r, self.k)
    
    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""
        # rule_set_size = self.k ** (2 * self.r + 1)
        # conv = decimal_to_base_k(self.rule, self.k)
        # self.rule_set = [0] * (rule_set_size - len(conv)) + conv
        self.rule_set = self.rule_builder.walk_through()

    def check_rule(self, inp):
        """Returns the new state based on the input states."""
        i = sum(inp[i] * (self.k ** i) for i in range(len(inp)))
        return self.rule_set[-(1 + int(i))]

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        a = np.random.randint(low=0, high=self.k, size=self.width)
        b = np.bincount([self.width // 2], [self.k - 1], self.width)
        return a

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""
        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)

class TableWalkThrough():
    def __init__(self, t, r, k):   
        self.lambda_prime = 0
        self.sq = None 
        
        self.t = t
        self.k = k
        self.r = r 
        
        self.rule_set = np.zeros(self.get_rule_size())
    
    def __select_method__(self, method='increase'):
        """Select the grow method of Langton's parameter."""
        method_func = self.increase_x

        if method == "increase":
            method_func = self.increase_x
        elif method == "decrease":
            method_func = self.decrease_x
        return method_func

    def __walk_through__(self):
        """Intermediate steps."""
        print("Before walk-through:")
        print("Initial rule set:", self.rule_set)

        x = self.calculate_x_parameter()
        print("Langton's parameter (before update):", x)

        print("After updating rule table:")
        print("Rule set:", self.rule_set)
        return x

    def get_rule_size(self):
        """Calculate neighborhood size. """
        return self.k ** (2 * self.r + 1)

    def get_quiescent_state(self):
        """Choose a quiescent state randomly."""
        return np.random.choice(range(self.k))

    def count_transitions_to_state(self):
        """Count transitions to the specified state in the rule set."""
        return np.count_nonzero(self.rule_set == self.sq)

    def calculate_x_parameter(self):
        """Calculate the Langto's parameter based on the count
        of transitions to the quiescent state."""
        k = self.get_rule_size()
        n = self.count_transitions_to_state()
        return (k - n) / k

    def increase_x(self):
        """Increase X: Replace transitions to Sq with transitions to other states."""
        i = np.random.choice(np.where(self.rule_set == self.sq)[0], size=1)
        self.rule_set[i] = np.random.choice(np.delete(range(self.k), self.sq), size=len(i))

    def decrease_x(self):
        """Decrease X: Replace transitions not to Sq with transitions to Sq."""
        i = np.random.choice(np.where(range(self.k) != self.sq)[0], size=1)
        self.rule_set[i] = self.sq

    def build_initial_rule_set_to_sq(self):
        """Build an initial rule set with transitions entirely to the quiescent state,
        and start by building ro sq."""
        self.sq = self.get_quiescent_state()
        self.rule_set = np.full(self.get_rule_size(), self.sq, dtype=int)
        return self.rule_set

    def walk_through(self):
        """Perform the table walk-through method to update the transition tables."""
        if self.lambda_prime < self.t:
            self.increase_x()
        elif self.lambda_prime > self.t:
            self.decrease_x()
        else:
            return self.rule_set
        self.lambda_prime = self.calculate_x_parameter()


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
        simulator.rule_set = rule_builder.walk_through('increase', threshold)
        simulator.reset()
        transient_len = simulate()
        transient_lens.append(transient_len)
    return transient_lens

if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)

    cx.start()

    # print(run_simulations(sim, rule_builder))

