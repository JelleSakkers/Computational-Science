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

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)

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
        rule_set_size = self.k ** (2 * self.r + 1)
        conv = decimal_to_base_k(self.rule, self.k)
        self.rule_set = [0] * (rule_set_size - len(conv)) + conv

    def check_rule(self, inp):
        """Returns the new state b
        ased on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
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


class TableWalkThrough(CASim):
    def __init__(self):
        CASim.__init__(self)

        self.rule_set = np.zeros(self.get_rule_size())

    def __select_method__(self, method='increase'):
        """Select the grow method of Langton's parameter."""
        method_func = self.increase_x

        if method == "increase":
            method_func = self.increase_x
        elif method == "decrease":
            method_func = self.decrease_x
        return method_func

    def __walk_through__(self, method_func, sq):
        """Intermediate steps."""
        print("Before walk-through:")
        print("Initial rule set:", self.rule_set)

        x = self.calculate_x_parameter(sq)
        print("Langton's parameter (before update):", x)

        method_func(sq)
        print("After updating rule table:")
        print("Rule set:", self.rule_set)
        return x

    def get_rule_size(self):
        """Calculate neighborhood size. """
        return self.k ** (2 * self.r + 1)

    def get_quiescent_state(self):
        """Choose a quiescent state randomly."""
        return np.random.choice(range(self.k))

    def count_transitions_to_state(self, sq):
        """Count transitions to the specified state in the rule set."""
        return np.count_nonzero(self.rule_set == sq)

    def calculate_x_parameter(self, sq):
        """Calculate the Langto's parameter based on the count
        of transitions to the quiescent state."""
        k = self.get_rule_size()
        n = self.count_transitions_to_state(sq)
        return (k - n) / k

    def increase_x(self, sq):
        """Increase X: Replace transitions to Sq with transitions to other states."""
        i = np.random.choice(np.where(self.rule_set == sq)[0], size=1)
        self.rule_set[i] = np.random.choice(np.delete(range(self.k), sq), size=len(i))

    def decrease_x(self, sq):
        """Decrease X: Replace transitions not to Sq with transitions to Sq."""
        i = np.random.choice(np.where(range(self.k) != sq)[0], size=1)
        self.rule_set[i] = sq

    def build_initial_rule_set_to_sq(self, sq):
        """Build an initial rule set with transitions entirely to the quiescent state,
        and start by building ro sq."""
        self.rule_set = np.full(self.get_rule_size(), sq, dtype=int)

    def walk_through(self, method, t):
        """Perform the table walk-through method to update the transition tables."""
        method_func = self.__select_method__(method)
        sq = self.get_quiescent_state()
        self.build_initial_rule_set_to_sq(sq)
        lambda_prime = 0

        while lambda_prime < t:
            lambda_prime = self.__walk_through__(method_func, sq)
        return self.rule_set

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
                break
            seen[key] = simulator.config[simulator.t]
            simulator.step()
            transient_len += 1
        return transient_len

    def hash_key(config):
        """
        Convert the NumPy array to a hashable byte representation.
        """
        return config.tobytes()

    def dehash_key(config):
        """
        Convert a byte representation of data to an integer.
        """
        return int.from_bytes(config, byteorder='big')

    # Set up simulation parameters
    simulation_range = np.arange(0.10, 1.01, 0.10)
    simulator.height = 10 ** 4

    transient_lengths = []

    # Run simulations for specified range
    for threshold in [0.10]:
        simulator.rule_set = rule_builder.walk_through('increase', threshold)
        simulator.reset()
        transient_len = simulate()
        transient_lengths.append(transient_len)
    return transient_lengths 

if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)

    cx.start()
