import numpy as np
import random
import matplotlib.pyplot as plt

from pyics import Model


# payoff matrix for Prisoner's Dilemma
rewards = {'CC': (3, 3), 'CD': (0, 5), 'DC': (5, 0), 'DD': (1, 1)}

# first two letters are the decisions made by player a and b three
# games ago. The second two letters are decisions two games ago, etcetera.
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


class Population(CAsim):
    def __init__(self,
                 seq_len=67,
                 generations=None,
                 population_size=None,
                 crossover_prob=None,
                 mutation_prob=None):

        CASim.__init__(self)
        self.__initialize_prob_params__(crossover_prob, mutation_prob)
        self.__initialize_pop_params__(seq_len, population_size)

    def __initialize_prob_params__(self, crossover_prob, mutation_prob):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def __initialize_pop_params__(self, seq_len, population_size):
        self.seq_len = seq_len
        self.pop_size = population_size
        self.pop = np.random.choice(
            ['C', 'D'], size=(self.pop_size, self.seq_len))

    def get_fitness(self, points, c=2):
        fitness = np.zeros(self.pop_size)
        
        fitness_average = np.mean(fitness)
        fitness_max = np.max(fitness)

        def calculate_a(c, average, maximum):
            return (c - 1) * (average / (maximum - average))

        def calculate_b(c, average, maximum):
            return average * (average - (c * average)) / (maximum - average)
        
        # calculate parameters a and b described by Goldberg
        a = calculate_a(c, fitness_average, fitness_max)
        b = calculate_b(c, fitness_average, fitness_max)
        
        for i in range(self.pop_size):
            # scale raw fitness scores
            fitness[i] = a * points[i] + b
        return fitness

    def extend_population(self, fitness):
        fitness_prob = fitness / np.sum(fitness)
        parents_idx = np.random.choice(np.arange(self.pop_size), size=2, p=fitness_prob)
        parent_1, parent_2 = self.pop[parents_idx]

        # determine if crossover should take place
        if random.uniform(0, 1) < self.crossover_prob:
            crossover_pos = random.randrange(67)
            offspring_1 = parent_1[:crossover_pos] + parent_2[crossover_pos:]
            offspring_2 = parant_2[:crossover_pos] + parent_1[crossover_pos:]
        else:
            offspring_1 = parent_1
            offspring_2 = parent_2
        
    def run_prisoners_dilemma(self, num_runs, num_generations):
        for i in range(num_runs):
            points = self.run_tournament()
            fitness = self.get_fitness(points)

            for j in range(1, num_generations):
                pass

    def run_tournament(self):
        def eo_against_eo():
            """Helper function to reduce indentation level."""
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    yield i, j

        points = np.zeros(self.pop_size)

        for i, j in eo_against_eo():
            mem_a = self.pop[i][64] + self.pop[j][64] + self.pop[i][65] + \
                self.pop[j][65] + self.pop[i][66] + self.pop[j][66]
            mem_b = self.pop[j][64] + self.pop[i][64] + self.pop[j][65] + \
                self.pop[i][65] + self.pop[j][66] + self.pop[i][66]
            mem_idx_a = mem.index(mem_a)
            mem_idx_b = mem.index(mem_b)
            outcome = rewards[mem_a[mem_idx_a] + mem_b[mem_idx_b]]
            points[i] += outcome[0]
            points[j] += outcome[1]
        return points


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
