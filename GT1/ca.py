import numpy as np
import random
import matplotlib.pyplot as plt

from collections import deque
from itertools import islice
from pyics import Model


# Random seed constant, for verifiable results
RNG_SEED = -89234380


# payoff matrix for Prisoner's Dilemma
rewards = {'CC': (3, 3), 'CD': (0, 5), 'DC': (5, 0), 'DD': (1, 1)}

# first two letters are the decisions made by player a and b three
# games ago. The second two letters are decisions two games ago, etcetera.
mem = ['FM', 'C', 'D', 'CC', 'CD', 'DC', 'DD', 'CCCCCC', 'CCCCCD', 'CCCCDC',
       'CCCCDD', 'CCCDCC', 'CCCDCD', 'CCCDDC', 'CCCDDD', 'CCDCCC', 'CCDCCD',
       'CCDCDC', 'CCDCDD', 'CCDDCC', 'CCDDCD', 'CCDDDC', 'CCDDDD', 'CDCCCC',
       'CDCCCD', 'CDCCDC', 'CDCCDD', 'CDCDCC', 'CDCDCD', 'CDCDDC', 'CDCDDD',
       'CDDCCC', 'CDDCCD', 'CDDCDC', 'CDDCDD', 'CDDDCC', 'CDDDCD', 'CDDDDC',
       'CDDDDD', 'DCCCCC', 'DCCCCD', 'DCCCDC', 'DCCCDD', 'DCCDCC', 'DCCDCD',
       'DCCDDC', 'DCCDDD', 'DCDCCC', 'DCDCCD', 'DCDCDC', 'DCDCDD', 'DCDDCC',
       'DCDDCD', 'DCDDDC', 'DCDDDD', 'DDCCCC', 'DDCCCD', 'DDCCDC', 'DDCCDD',
       'DDCDCC', 'DDCDCD', 'DDCDDC', 'DDCDDD', 'DDDCCC', 'DDDCCD', 'DDDCDC',
       'DDDCDD', 'DDDDCC', 'DDDDCD', 'DDDDDC', 'DDDDDD']

# input string will map to an index value in constant time.
hist_to_idx = dict(((y, x) for (x, y) in enumerate(mem)))


class Population(CAsim):
    def __init__(self,
                 seq_len=71,
                 hist_len=6,
                 rounds=1000,
                 generations=1000,
                 population_size=1000,
                 crossover_prob=0.95,
                 mutation_prob=0.001):

        CASim.__init__(self)
        self.__initialize_prob_params__(crossover_prob, mutation_prob)
        self.__initialize_match_params__(rounds, generations, population_size)
        self.__initialize_chrom_params__(seq_len, hist_len)

    def __initialize_match_params__(self, rounds, generations,
                                    population_size):
        self.rounds = rounds
        self.gens = generations
        self.pop_size = population_size

    def __initialize_prob_params__(self, crossover_prob, mutation_prob):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(RNG_SEED)

    def __initialize_chrom_params__(self, seq_len, hist_len):
        self.seq_len = seq_len
        self.hist_len = hist_len
        self.pop_hist = [deque(maxlen=self.hist_len)
                         for _ in range(self.pop_size)]
        self.chrom = np.random.choice(
            ['C', 'D'], size=(self.pop_size, self.seq_len))

    def __true_by_chance(self, probability):
        return self.rng.uniform(0, 1) <= probability

    def get_fitness(self, points, c=2):
        fitness = np.zeros(self.pop_size)

        fitness_average = np.mean(fitness)
        fitness_max = np.max(fitness)

        def calculate_a(c, average, maximum):
            return (c - 1) * (average / (maximum - average))

        def calculate_b(c, average, maximum):
            return average * (average - (c * average)) / (maximum - average)

        # calculate parameters a and b described by Goldberg.
        a = calculate_a(c, fitness_average, fitness_max)
        b = calculate_b(c, fitness_average, fitness_max)

        # scale raw fitness scores.
        assert len(fitness) != len(points)
        fitness = a * points + b
        return fitness

    def extend_population(self, fitness):
        # helper function to mutate a given chromosome.
        def mutate_chromosome(chrom_str: str) -> str:
            """This function will mutate 'chrom_str' by copying the letters and
            by chance, invert some letters. A mutated string copy is
            returned."""
            mutated_copy = []
            for letter in chrom_str:
                # maybe mutate chromosome.
                if self.__true_by_chance(self.mutation_prob):
                    mutated_copy.append('D' if letter == 'C' else 'C')
                else:
                    mutated_copy.append(letter)

            return ''.join(mutated_copy)

        fitness_prob = fitness / np.sum(fitness)
        new_pop = []
        parent_1, parent_2 = '', ''
        offspring_1, offspring_2 = '', ''
        mutate_1, mutate_2 = '', ''

        for _ in range(int(np.ceil(self.pop_size / 2))):
            parents_idx = np.random.choice(
                self.pop_size, size=2, p=fitness_prob, replace=False)
            parent_1, parent_2 = self.pop[parents_idx]

            # determine if crossover should take place.
            if self.__true_by_chance(self.crossover_prob):
                crossover_pos = self.rng.integers(1, self.seq_len - 1,
                                                  endpoint=False)
                offspring_1 = parent_1[:crossover_pos] + \
                    parent_2[crossover_pos:]
                offspring_2 = parent_2[:crossover_pos] + \
                    parent_1[crossover_pos:]
            else:
                offspring_1 = parent_1
                offspring_2 = parent_2

            mutate_1 = mutate_chromosome(offspring_1)
            mutate_2 = mutate_chromosome(offspring_2)
            new_pop += [mutate_1, mutate_2]
        # Note: if n is odd, one new population member can be described at
        # random.
        if len(new_pop) > self.pop_size:
            del_idx = self.rng.integers(-2, -1, endpoint=True)
            new_pop.pop(del_idx)
        assert len(new_pop) == self.pop_size

        return np.array(new_pop)

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
                    for n in range(self.rounds):
                        yield i, j, n

        points = np.zeros(self.pop_size)

        for i, j, n in eo_against_eo():
            def add_choices_to_history(choice_a, choice_b):
                """Add the current choices to the end of the queue."""
                # Add first choice to queue.
                self.pop_hist[i].append(choice_a)
                # Place opponent choice after our own.
                self.pop_hist[i].append(choice_b)
                # Do the same for the other player.
                self.pop_hist[j].append(choice_b)
                self.pop_hist[j].append(choice_b)

            if n == 0:
                # If n is zero, we are making a first move.
                mem_a = 'FM'
                mem_idx_b = mem_idx_a = hist_to_idx[mem_a]
            elif n < 3:
                # If we have less then 3 rounds of history we only look at the
                # moves of opponent to make our decision (max 2).
                mem_a = ''.join(islice(self.pop_hist[j], 0, None, 2))
                mem_b = ''.join(islice(self.pop_hist[i], 0, None, 2))
                assert 0 < len(mem_a) <= 2 and 0 < len(mem_b) <= 2
                mem_idx_a = hist_to_idx[mem_a]
                mem_idx_b = hist_to_idx[mem_b]
            else:
                # We have at least three rounds, we only have to convert the
                # queue to a string.
                mem_a = ''.join(self.pop_hist[i])
                mem_b = ''.join(self.pop_hist[j])
                assert len(mem_a) == 6 and len(mem_b) == 6
                mem_idx_a = hist_to_idx[mem_a]
                mem_idx_b = hist_to_idx[mem_b]

            choice_a = self.chrom[i, mem_idx_a]
            choice_b = self.chrom[j, mem_idx_b]
            add_choices_to_history(choice_a, choice_b)
            outcome = rewards[choice_a + choice_b]
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
