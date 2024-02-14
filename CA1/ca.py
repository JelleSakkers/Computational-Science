import numpy as np
import matplotlib.pyplot as plt

from pyics import Model


def create_class_dict():
    data = np.genfromtxt('rule_class_wolfram.csv', delimiter=',')
    class_dict = {}

    for i in range(1, 5):
        values = data[data[:, 1] == i][:, 0]
        class_dict[i] = values
    return class_dict


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
        self.neighbourhood = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 10000)
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
        [2, 2, 2] -> 0 an1d [0, 0, 1] -> 2."""
        rule_set_size = self.k ** (2 * self.r + 1)
        conv = decimal_to_base_k(self.rule, self.k)
        self.rule_set = [0] * (rule_set_size - len(conv)) + conv

    def check_rule(self, inp):
        """Returns the new state b
        ased on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
        idx = sum(inp[i] * (self.k ** i) for i in range(len(inp)))
        return self.rule_set[-(1 + int(idx))]

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        a = np.bincount([self.width // 2], [self.k - 1], self.width)
        b = np.random.randint(low=0, high=self.k, size=self.width)
        return b

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""
        self.t = 0
        self.neighbourhood = []
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

        self.neighbourhood.append(self.config[self.t])


class Cycle(CASim):
    def __init__(self, sim):
        """Initialize Cycle class with a reference to the given simulation."""
        self.sim = sim

    def __detect_init__(self):
        """Initialize the detection process with the given table."""
        self.seen = {}
        self.cycles = []

    def __experiments_init__(self):
        """Initialize the experiment variables. Rules is initialize from 1 till 256. """
        self.rules = [i for i in range(0, 256)]
        self.t_max = 10 ** 4
        self.avg_cycles = []

    def __detect__(self, i, v):
        """Detect cycles in the simulation and update cycle information."""
        c_start = self.seen[v]
        c_length = i - c_start
        self.cycles.append(c_length)

    def __plot__(self):
        """ Plot the average cycle lengths over 256 Wolfram Rules"""
        plt.xlabel('rule')
        plt.ylabel('average cycle length')
        plt.title('Average cycle length by class (random value start)')

    def detect(self, table):
        """Detect cycles in the provided table."""
        self.__detect_init__()

        for i, v in enumerate(table):
            v = tuple(v)
            if v in self.seen:
                self.__detect__(i, v)
            self.seen[v] = i

    def stats(self):
        """Calculate the average cycle lenght. Append zero when
        a empty list is encounterd."""
        self.avg_cycles.append(np.mean(np.array(self.cycles))) if self.cycles else self.avg_cycles.append(0)

    def run_experiments(self):
        """Run experiments for 256 Wolfram Rules."""
        self.__experiments_init__()
        self.sim.t = self.t_max

        for rule in self.rules:
            self.sim.rule = rule
            self.sim.reset()

            for _ in range(self.t_max):
                self.sim.step()
            self.detect(self.sim.neighbourhood)
            self.stats()

        self.plot_experiments()

    def plot_experiments(self):
        """Plot the 256 Wolfram experiment results. Every rule is
        categorised and colored based one of the four Wolfram classes."""
        self.__plot__()

        class_dict = create_class_dict()
        class_colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow'}

        ys = []
        colors = []

        for rule, length in enumerate(self.avg_cycles):
            class_v = next(key for key, array in class_dict.items() if rule in array)
            ys.append(length)
            colors.append(class_colors[class_v])

        legend_labels = {class_num: f'Wolfram class {class_num}' for class_num in class_colors.keys()}

        plt.ylim(0, 100)

        plt.legend(handles=[plt.Line2D([0], [0], color=class_colors[color], label=label) for color, label in legend_labels.items()])
        plt.bar(np.linspace(0, 255, 256), ys, color=colors, width=0.2)

        plt.xticks(np.arange(0, 256, 50))
        plt.savefig('test.png')


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)

    cycle = Cycle(sim)
    cycle.run_experiments()

