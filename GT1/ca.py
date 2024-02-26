import numpy as np
import random
import matplotlib.pyplot as plt

from pyics import Model


rewards = {'CC':(3, 3), 'CD':(0, 5), 'DC':(5, 0), 'DD':(1, 1)}
mem = ['CCCCCC', 'CCCCCD', 'CCCCDC', 'CCCCDD', 'CCCDCC', 'CCCDCD',
       'CCCDDC', 'CCCDDD', 'CCDCCC', 'CCDCCD', 'CCDCDC', 'CCDCDD',
       'CCDDCC', 'CCDDCD', 'CCDDDC', 'CCDDDD', 'CDCCCC', 'CDCCCD'
       'CDCCDC', 'CDCCDD', 'CDCDCC', 'CDCDCD', 'CDCDDC', 'CDCDDD',
       'CDDCCC', 'CDDCCD', 'CDDCDC', 'CDDCDD', 'CDDDCC', 'CDDDCD',
       'CDDDDC', 'CDDDDD', 'DCCCCC', 'DCCCCD', 'DCCCDC', 'DCCCDD',
       'DCCDCC', 'DCCDCD', 'DCCDDC', 'DCCDDD', 'DCDCCC', 'DCDCCD',
       'DCDCDC', 'DCDCDD', 'DCDDCC', 'DCDDCD', 'DCDDDC', 'DCDDDD',
       'DDCCCC', 'DDCCCD', 'DDCCDC', 'DDCCDD', 'DDCDCC', 'DDCDCD',
       'DDCDDC', 'DDCDDD', 'DDDCCC', 'DDDCCD', 'DDDCDC', 'DDDCDD',
       'DDDDCC', 'DDDDCD', 'DDDDDC', 'DDDDDD']


class Individual():
    def __init__(self):
        self.points = 0

    def fitness():
        pass


class Population(CAsim):
    def __init__(self, 
                 generations=None,
                 population_size=None,
                 crossover_prob=None,
                 mutation_prob=None):
        
        CASim.__init__(self)
        
        self.generations = generations
        self.pop_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.pop = []

    def initialize_pop(self):
        genes = ['C', 'D']
        chromosome = ''

        for _ in range(self.pop_size):
            chromosome += random.choice(genes)
            self.pop += [chromosome]
            chromosome = ''

    def grade_fitness(self):
        pass

    def crossover(self):
        pass

    def breed(self):
        pass

    def evolve(self):
        pass

    def prisoners_dilemma():
        pass


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


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)

    cx.start()
