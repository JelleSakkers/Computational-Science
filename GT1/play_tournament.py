from ca import RNG_SEED, play_tournament, hist_to_idx
from collections import deque
from itertools import islice
import numpy as np

BEST_FIVE_CHROMS = \
    ['DDDDCDDDDDDDDCCCDCDCDDDCCDCDDCDDDDCCDCDCCDDDDDCDDDCCDDCDDDDCDDDDCDDDCDD',
     'DDDDCDDDDDDDDCCCDCDCDDDCCDCDDCDDDDDCDCDCCDDDDDCDDDCCDDDDDDDCDDDDCDDDCDD',
     'DDDDCDDDDDDDDCCCDCDCDDDCCDCDDCDDDDDCDCDCCDDDDDCDDDCCDDDDCDDCDDDDCDDDCDD',
     'DDCDCDDDDDDDDCCCDCDCDDDCCDCDDCDDDDCCDCDCCDDDDDCDDDCCDDDDDDDCDDDDCDDDCDD',
     'DDDDCDDDDDDDDCCCDCDCDDDCCDCDDCDDDDDCDCDCCDDDDDCDDDCCDDDDDDDCDDDDCDDDCDD']

HIST_LEN = 6
SEQ_LEN = 71


def main():
    rng = np.random.default_rng(RNG_SEED)
    random_strategy_b = rng.choice(['C', 'D'], size=(SEQ_LEN))

    for chrom in BEST_FIVE_CHROMS:
        hist_a = deque(maxlen=HIST_LEN)
        hist_b = deque(maxlen=HIST_LEN)

        points_a = points_b = 0

        genetic_strategy_a = np.array(list(chrom))

        def make_choice_a(n):
            """Mostly copied from the 'run_tournament' method inside the
            'Population' class in the file 'ca.py'"""
            if n == 0:
                # If n is zero, we are making a first move.
                mem_a = 'FM'
                mem_idx_a = hist_to_idx[mem_a]
            elif n < 3:
                # If we have less then 3 rounds of history we only look at the
                # moves of opponent to make our decision (max 2).
                mem_a = ''.join(islice(hist_a, 1, None, 2))
                assert 0 < len(mem_a) <= 2
                mem_idx_a = hist_to_idx[mem_a]
            else:
                # We have at least three rounds, we only have to convert the
                # queue to a string.
                mem_a = ''.join(hist_a)
                assert len(mem_a) == 6
                mem_idx_a = hist_to_idx[mem_a]

            return genetic_strategy_a[mem_idx_a]

        def make_choice_b(n):
            """Mostly copied from the 'run_tournament' method inside the
            'Population' class in the file 'ca.py'"""
            if n == 0:
                # If n is zero, we are making a first move.
                mem_b = 'FM'
                mem_idx_b = hist_to_idx[mem_b]
            elif n < 3:
                # If we have less then 3 rounds of history we only look at the
                # moves of opponent to make our decision (max 2).
                mem_b = ''.join(islice(hist_b, 1, None, 2))
                assert 0 < len(mem_b) <= 2
                mem_idx_b = hist_to_idx[mem_b]
            else:
                # We have at least three rounds, we only have to convert the
                # queue to a string.
                mem_b = ''.join(hist_b)
                assert len(mem_b) == 6
                mem_idx_b = hist_to_idx[mem_b]

            return random_strategy_b[mem_idx_b]

        # The next couple of functions are mostly copied from 'ca.py' and
        # modified so it works with the local variables here.
        def add_choices_to_history_a(choice_a, choice_b):
            """add the current choices to the end of the queue of player a."""
            # add our own first choice to queue.
            hist_a.append(choice_a)
            # place opponent choice after our own.
            hist_a.append(choice_b)

        def add_choices_to_history_b(choice_a, choice_b):
            """add the current choices to the end of the queue of player b."""
            # add our own first choice to queue.
            hist_b.append(choice_b)
            # place opponent choice after our own.
            hist_b.append(choice_a)

        def add_outcome_a(outcome_a, _):
            """Add the points of a to the total for player a."""
            nonlocal points_a
            points_a += outcome_a

        def add_outcome_b(_, outcome_b):
            """Add the points of b to the total for player b."""
            nonlocal points_b
            points_b += outcome_b

        play_tournament(1000, make_choice_a, make_choice_b,
                        add_choices_to_history_a, add_choices_to_history_b,
                        add_outcome_a, add_outcome_b)

        print(  # f"Genetic chromosome:\t{chrom}\n"
              f"Points for genetic strategy: {points_a}\n"
              # f"Random chromosome:\t{''.join(random_strategy_b)}\n"
              f"Points for random strategy: {points_b}")


if __name__ == '__main__':
    main()
