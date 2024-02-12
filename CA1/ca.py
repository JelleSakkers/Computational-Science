import numpy as np

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


class Cycle:
    def __init__(self):
        self.seen = {}
        self.cycles = []

    def __detect_init__(self, table):
        self.table = table

    def __detect__(self, i, v):
        c_start = self.seen[v]
        c_length = i - c_start
        self.cycles.append((c_start, c_length))

    def __plot__(self):
        pass

    def detect(self, table):
        self.__detect_init__(table)
        
        for i, v in enumerate(self.table):
            v = tuple(v)
            if v in self.seen:
                self.__detect__(i, v)
            else:
                self.seen[v] = i
        print(self.cycles)

    def stats(self):
        pass

    def run(self, cx):  
        rules = [i for i in range(1, 2**8 + 1)]
        
        for rule in rules:
            cx.setter_rule(rule)
            cx.s

    def plot_experiment(self):
        pass


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
        idx = sum(inp[i] * (self.k ** i) for i in range(len(inp)))
        return self.rule_set[-(1 + int(idx))]
    
    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        a = np.bincount([self.width // 2], [self.k - 1], self.width)
        b = np.random.randint(low=0, high=self.k, size=self.width)
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
        cycle = Cycle() 
        table = []
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
            table.append(values)
            self.config[self.t, patch] = self.check_rule(values)
        
        cycle.detect(table)

if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
