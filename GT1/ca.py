from typing import Callable
import numpy as np

from collections import deque, namedtuple
from itertools import islice
from pyics import Model


# Random seed constant, for verifiable results
RNG_SEED = 89234380


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


FitnessHistory = namedtuple('FitnessHistory', ['chrom', 'fitness', 'score'])


def play_tournament(rounds: int,
                    make_choice_a: Callable[[int], str],
                    make_choice_b: Callable[[int], str],
                    player_a_proces_choices: Callable[[str, str], None],
                    player_b_proces_choices: Callable[[str, str], None],
                    player_a_proces_outcome: Callable[[int, int], None],
                    player_b_proces_outcome: Callable[[int, int], None]):
    """Play a full tournament between two players, play a given amount of
    rounds. 'rounds' should be an integer.

    Output and input for every function:
    choice_of_x = make_choice_x(current_round)
    player_x_proces_choices(choice_of_player_self, choice_of_player_other)
    player_x_proces_outcome(points_to_player_self, points_to_player_other)"""
    for n in range(np.abs(rounds)):
        play_match(n, make_choice_a, make_choice_b, player_a_proces_choices,
                   player_b_proces_choices, player_a_proces_outcome,
                   player_b_proces_outcome)


def play_match(n: int,
               make_choice_a: Callable[[int], str],
               make_choice_b: Callable[[int], str],
               player_a_proces_choices: Callable[[str, str], None],
               player_b_proces_choices: Callable[[str, str], None],
               player_a_proces_outcome: Callable[[int, int], None],
               player_b_proces_outcome: Callable[[int, int], None]):
    """Play a single round of the prison game between two player.
    All input pararmeters are functions. Two function should be provided who
    should genereate a choice for either player. Two functions should be
    provided to do something with the given choices (like adding it to
    history). Lastly two functions should be provided to proces the distributed
    points between the two players. If the current round number is important,
    'n' should be set to useful positive integer.

    Output and input for every function:
    choice_of_x = make_choice_x(current_round)
    player_x_proces_choices(choice_of_player_self, choice_of_player_other)
    player_x_proces_outcome(points_to_player_self, points_to_player_other)"""
    # Check if function are not the same reference, if it is, it should only be
    # run once.
    if make_choice_a is not make_choice_b:
        choice_a = make_choice_a(n)
        choice_b = make_choice_b(n)
    else:
        choice_a = choice_b = make_choice_a(n)

    # Again, check that the functions are not the same object.
    if player_a_proces_choices is not player_b_proces_choices:
        player_a_proces_choices(choice_a, choice_b)
        player_b_proces_choices(choice_b, choice_a)
    else:
        player_a_proces_choices(choice_a, choice_b)

    outcome = rewards[choice_a + choice_b]

    # Again, check that the functions are not the same object.
    if player_a_proces_outcome is not player_b_proces_outcome:
        player_a_proces_outcome(outcome[0], outcome[1])
        player_b_proces_outcome(outcome[1], outcome[0])
    else:
        player_a_proces_outcome(outcome[0], outcome[1])


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        # self.make_param('r', 2)
        # self.make_param('k', 4)
        # self.make_param('width', 128)
        # self.make_param('height', 200)
        # self.make_param('rule', 30, setter=self.setter_rule)

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


class Population(CASim):
    def __init__(self,
                 rounds=50,
                 generations=100,
                 population_size=100,
                 crossover_prob=0.95,
                 mutation_prob=0.001):

        CASim.__init__(self)
        SEQ_LEN = 71
        HIST_LEN = 6
        self.__initialize_prob_params__(crossover_prob, mutation_prob)
        self.__initialize_match_params__(rounds, generations, population_size)
        self.__initialize_chrom_params__(SEQ_LEN, HIST_LEN)

    def __initialize_match_params__(self, rounds, generations,
                                    population_size):
        self.make_param('rounds', rounds)
        self.make_param('gens', generations)
        self.make_param('pop_size', population_size)

    def __initialize_prob_params__(self, crossover_prob, mutation_prob):
        self.make_param('crossover_prob', crossover_prob,
                        setter=self.__check_probability)
        self.make_param('mutation_prob', mutation_prob,
                        setter=self.__check_probability)
        self.rng = np.random.default_rng(RNG_SEED)

    def __initialize_chrom_params__(self, seq_len=None, hist_len=None):
        self.fitness_hist: list[FitnessHistory] = []
        if seq_len is not None:
            self.seq_len = seq_len
        if hist_len is not None:
            self.hist_len = hist_len
        self.pop_hist = [deque(maxlen=self.hist_len)
                         for _ in range(self.pop_size)]
        self.__initiaize_random_chroms__()

    def __initiaize_random_chroms__(self):
        self.chrom = self.rng.choice(
            ['C', 'D'], size=(self.pop_size, self.seq_len))

    def __check_probability(self, prop):
        """Probability should be between zero and one."""
        return max(0, min(prop, 1))

    def __true_by_chance(self, probability):
        return self.rng.uniform(0, 1) <= probability

    def get_best_five_chromosomes(self):
        """Return the best five scoring chromosomes in descending order."""
        if len(self.fitness_hist) == 0:
            return []
        return self.fitness_hist[-1].chrom

    def get_fitness(self, points, c=2):
        fitness = np.zeros(self.pop_size)

        points_average = np.mean(points)
        points_max = np.max(points)

        def calculate_a(c, average, maximum):
            max_av_diff = maximum - average
            if max_av_diff != 0:
                return (c - 1) * (average / max_av_diff)
            else:
                return 1

        def calculate_b(c, average, maximum):
            max_av_diff = maximum - average
            if max_av_diff != 0:
                return average * (average - (c * average)) / max_av_diff
            else:
                return 0

        # calculate parameters a and b described by Goldberg.
        a = calculate_a(c, points_average, points_max)
        b = calculate_b(c, points_average, points_max)

        # scale raw fitness scores.
        assert len(fitness) == len(points)
        fitness = a * points + b
        return fitness

    def extend_population(self, fitness):
        # helper function to mutate a given chromosome.
        def mutate_chromosome(chrom_str: str) -> np.ndarray:
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

            return np.array(mutated_copy)

        # Replace negative fitness results with zero.
        fitness[fitness < 0] = 0
        fitness_prob = fitness / np.sum(fitness)
        new_pop = []

        for _ in range(int(np.ceil(self.pop_size / 2))):
            try:
                parents_idx = np.random.choice(
                    self.pop_size, size=2, p=fitness_prob, replace=False)
            except ValueError:
                # Sometimes, if the chromosomes are too much alike, the
                # algorithm will stall no matter what. Any attempt to create a
                # new generation should be disallowed.
                self.gens = self.t
                print('No real scores could be calculated, this means that '
                      'all the chromosomes look to\n'
                      'much like each other.\n'
                      'The number of generations has been reset to the '
                      'current time, since no more\n'
                      'new children can be created.')
                return self.chrom
            parent_1, parent_2 = self.chrom[parents_idx]

            # determine if crossover should take place.
            if self.__true_by_chance(self.crossover_prob):
                crossover_pos = self.rng.integers(1, self.seq_len - 1,
                                                  endpoint=False)
                offspring_1 = np.concatenate(
                    (parent_1[:crossover_pos], parent_2[crossover_pos:]))
                offspring_2 = np.concatenate(
                    (parent_2[:crossover_pos], parent_1[crossover_pos:]))
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
                    # Make sure for every new player match to empty all the
                    # history queues.
                    self.pop_hist[i].clear()
                    self.pop_hist[j].clear()
                    for n in range(self.rounds):
                        yield i, j, n

        points = np.zeros(self.pop_size)

        for i, j, n in eo_against_eo():
            def add_choices_to_history_a(choice_self, choice_other):
                """add the current choices to the end of the queue of player
                a on index i."""
                # add our own first choice to queue.
                self.pop_hist[i].append(choice_self)
                # place opponent choice after our own.
                self.pop_hist[i].append(choice_other)

            def add_choices_to_history_b(choice_self, choice_other):
                """add the current choices to the end of the queue of player
                b on index j."""
                # do the same for the other player, but if i is j it should not
                # be added a second time.
                if i != j:
                    self.pop_hist[j].append(choice_self)
                    self.pop_hist[j].append(choice_other)

            def add_outcome_a(outcome_self, _):
                """Add the points of a to the correct index in the array, which
                for a is i."""
                points[i] += outcome_self

            def add_outcome_b(outcome_self, _):
                """Add the points of b to the correct index in the array, which
                for b is j."""
                # Make sure not to add the points a second time (if 'i' is 'j').
                if i != j:
                    points[j] += outcome_self

            if n == 0:
                # If n is zero, we are making a first move.
                mem_a = 'FM'
                mem_idx_b = mem_idx_a = hist_to_idx[mem_a]
            elif n < 3:
                # If we have less then 3 rounds of history we only look at the
                # moves of opponent to make our decision (max 2).
                mem_a = ''.join(islice(self.pop_hist[i], 1, None, 2))
                mem_b = ''.join(islice(self.pop_hist[j], 1, None, 2))
                assert 0 < len(mem_a) <= 2 and 0 < len(mem_b) <= 2, \
                    f"{n = }\n" \
                    f"{i = }\n{j = }\n" \
                    f"{mem_a = }\n{mem_b = }\n{self.pop_hist[i] = }\n" \
                    f"{self.pop_hist[j] = }"
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

            # Although the current round number is provided, it won't be used
            # inside any of the provided functions.
            play_match(n, lambda _: choice_a, lambda _: choice_b,
                       add_choices_to_history_a, add_choices_to_history_b,
                       add_outcome_a,
                       add_outcome_b)
        return points

    def reset(self):
        """Create a new set of chromosomes and reset history."""
        self.t = 0
        self.__initialize_chrom_params__()

    def draw(self):
        """Draw the current state of the history."""
        # Helper function to convert raw history to a list
        def convert_to_hist(idx):
            t_l, fitness = [], []
            for t, ordered_fitness in enumerate(self.fitness_hist):
                t_l.append(t)
                fitness.append(ordered_fitness.score[idx])

            return t_l, fitness

        import matplotlib.pyplot as plt

        plt.cla()

        # Plot top five scoring chromosomes.
        t_l, fitness = convert_to_hist(4)
        plt.plot(t_l, fitness, '-m', label='fifth')
        t_l, fitness = convert_to_hist(3)
        plt.plot(t_l, fitness, '-c', label='fourth')
        t_l, fitness = convert_to_hist(2)
        plt.plot(t_l, fitness, '-g', label='third')
        t_l, fitness = convert_to_hist(1)
        plt.plot(t_l, fitness, '-b', label='second')
        t_l, fitness = convert_to_hist(0)
        plt.plot(t_l, fitness, '-r', label='first')
        plt.legend(title='Best Scoring Chromosomes')
        plt.title('Fitness history of the best five scoring chromosomes or '
                  'rule tables')

    def step(self):
        """Performs a single step of the simulation by advancing time and
        applying selection to determine the new generation."""
        self.t += 1
        if self.t > self.gens:
            print("The best five strategies are (in chromosome form):")
            for chrom_list in self.get_best_five_chromosomes():
                print(''.join(chrom_list))
            return True

        points = self.run_tournament()
        fitness = self.get_fitness(points)
        new_pop = self.extend_population(fitness)
        # Find the indices of the best five scoring chromosomes.
        best_chroms_idx = np.argsort(fitness)[:-6:-1]
        self.fitness_hist.append(
            FitnessHistory(self.chrom[best_chroms_idx],
                           fitness[best_chroms_idx],
                           points[best_chroms_idx]))
        # Replace current generation with a new one.
        self.chrom = new_pop


if __name__ == '__main__':
    sim = Population()
    from pyics import GUI
    cx = GUI(sim)

    cx.start()
