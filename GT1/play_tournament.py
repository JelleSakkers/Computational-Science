from typing import Callable, Deque, Generator, List, Tuple
from ca import RNG_SEED, play_tournament, hist_to_idx, rewards
from collections import deque
from itertools import islice
import numpy as np
from numpy.typing import NDArray

BEST_FIVE_CHROMS = \
    ['DDDDDDDCCCCDDCCDCDDCDCDDCCDCDDCDDCCDDDDDDCCDDDCDDCCCDDDCDCDCCDDCCDDCDDC',
     'DDDDDDDCCCCDDCCDCDDCDCDDCCDCCDCCDCCDDDDDDCCDDDCDDCCCDDDCDCDCCDDCCDDCDDC',
     'DDDDDDDCCCCDDCCDCDDCDCDDCCDCCDCCDCCDDDDDDCCDDDCDDCCCDDDCDCDCCDDCCDDCDDC',
     'DDDDCDDCCCCDDCCDCDDCDCCDCCDCCDCCDCCDDDDDDCCDDDCDDCCCDDDCDCDCCDDCCDDCDDC',
     'DDDDCDDCCCCDDCCDCDDCDCDDCCDCCDCCDCCDDDDDDCCDDDCDDCDCDDDCDCDCCDDCCDDCDDC']

HIST_LEN = 6
SEQ_LEN = 71


class Score():
    """Class made purely so it is possible to get a reference to the score of
    a specific player."""
    def __init__(self):
        self.score: int = 0


Player_data = Tuple[Score, Callable[[int], str], Callable[[str, str], None],
                    Callable[[int, int], None], Callable[[], None]]


def create_tit_for_tat_rule_set() -> Player_data:
    """Just repeat whatever the opponent did the previous time."""
    opp_last_move: str = ''
    self_score: Score = Score()

    def make_choice(_: int) -> str:
        """Repeat the opponent strategy. Initial move is positive."""
        nonlocal opp_last_move
        return 'C' if opp_last_move == '' else opp_last_move

    def save_opp_history(_: str, move_other: str):
        """Set the opponent's last move to 'move_other'."""
        nonlocal opp_last_move
        opp_last_move = move_other

    def save_outcome(outcome_self: int, _: int):
        """Save the total score."""
        nonlocal self_score
        self_score.score += outcome_self

    def reset() -> None:
        """Reset the score and history."""
        nonlocal opp_last_move, self_score
        opp_last_move = ''
        self_score.score = 0

    return self_score, make_choice, save_opp_history, save_outcome, reset


def create_random_rule_set(seed: int) -> Player_data:
    """Will create a proper random strategy based upon the provided seed."""
    rng = np.random.default_rng(seed)
    self_score: Score = Score()

    def make_choice(_: int) -> str:
        """Make a random choice."""
        return rng.choice(['C', 'D'])

    def ignore_hist(_: str, __: str):
        """We don't need any history information."""
        pass

    def save_outcome(outcome_self: int, _: int):
        """Saving the total score is still very useful."""
        self_score.score += outcome_self

    def reset() -> None:
        """Reset the score and history."""
        nonlocal self_score
        self_score.score = 0

    return self_score, make_choice, ignore_hist, save_outcome, reset


def convert_chrom_to_rule_set(chrom: str) -> Player_data:
    """Can convert a given chromosome string to a rule table and functions."""
    assert len(chrom) == SEQ_LEN
    all_hist: Deque[str] = deque(maxlen=HIST_LEN)
    self_score: Score = Score()
    genetic_strategy = np.array(list(chrom))

    def make_choice(n: int) -> str:
        """Mostly copied from the 'run_tournament' method inside the
        'Population' class in the file 'ca.py'"""
        nonlocal all_hist
        if n == 0:
            # If n is zero, we are making a first move.
            mem_self = 'FM'
            mem_idx_self = hist_to_idx[mem_self]
        elif n < 3:
            # If we have less then 3 rounds of history we only look at the
            # moves of opponent to make our decision (max 2).
            mem_self = ''.join(islice(all_hist, 1, None, 2))
            assert 0 < len(mem_self) <= 2, f"Current round: {n}, there are " \
                "less than three rounds of history there should be a max of " \
                f"2 opponent moves! It currently is: {len(mem_self)}"
            mem_idx_self = hist_to_idx[mem_self]
        else:
            # We have at least three rounds, we only have to convert the queue
            # to a string.
            mem_self = ''.join(all_hist)
            assert len(mem_self) == 6
            mem_idx_self = hist_to_idx[mem_self]

        return genetic_strategy[mem_idx_self]

    # The next couple of functions are mostly copied from 'ca.py' and modified
    # so it works with the local variables here.
    def add_choices_to_history(choice_self, choice_other):
        """add the current choices to the end of the queue of player a."""
        nonlocal all_hist
        # add our own first choice to queue.
        all_hist.append(choice_self)
        # place opponent choice after our own.
        all_hist.append(choice_other)

    def add_outcome(outcome_self, _):
        """Add the points of a to the total for player a."""
        nonlocal self_score
        self_score.score += outcome_self

    def reset() -> None:
        """Reset the score and history."""
        nonlocal all_hist, self_score
        all_hist.clear()
        self_score.score = 0

    return self_score, make_choice, add_choices_to_history, add_outcome, reset


def convert_best_chroms_to_rule_set() -> Generator[Player_data, None, None]:
    """Will convert the list of chromosomes to rule tables and functions."""
    for chrom in BEST_FIVE_CHROMS:
        yield convert_chrom_to_rule_set(chrom)


def create_defect_rule_set(defect_tolarance: float, hist_len: int) -> \
        Player_data:
    opp_hist: Deque[str] = deque(maxlen=hist_len)
    self_score: Score = Score()

    def make_choice(n: int) -> str:
        nonlocal opp_hist
        if n == 0:
            # Depending on how high the defect tolarance has been set, we will
            # initially cooperate.
            return 'C' if defect_tolarance >= 0.5 else 'D'
        else:
            # Check what the defect total ratio is.
            defect_sum = np.sum(np.array(opp_hist) == 'D')
            total_sum = len(opp_hist)
            def_ratio: float = defect_sum / total_sum
            # if the calculated ratio is more then the maximum allowed, we will
            # choose to defect.
            return 'D' if def_ratio > defect_tolarance else 'C'

    def save_history(_: str, choice_other: str) -> None:
        nonlocal opp_hist
        opp_hist.append(choice_other)

    def save_score(score_self: int, _: int) -> None:
        nonlocal self_score
        self_score.score += score_self

    def reset() -> None:
        """Reset the score and history."""
        nonlocal opp_hist, self_score
        opp_hist.clear()
        self_score.score = 0

    return self_score, make_choice, save_history, save_score, reset


def create_all_players() -> Tuple[List[Player_data], List[str]]:
    """Returns a list with all the player data, second list contains player
    names."""
    hist_len = 6
    player_data: List[Player_data] = []
    player_names: List[str] = []

    for idx, game_data in \
            enumerate(convert_best_chroms_to_rule_set()):
        player_names.append(f"best_gen_chrom_{idx}")
        player_data.append(game_data)

    game_data = create_defect_rule_set(1/3, hist_len)
    player_names.append("low_defect_tolerance")
    player_data.append(game_data)

    game_data = create_defect_rule_set(1/2, hist_len)
    player_names.append("mid_defect_tolerance")
    player_data.append(game_data)

    game_data = create_defect_rule_set(3/4, hist_len)
    player_names.append("high_defect_tolerance")
    player_data.append(game_data)

    game_data = create_defect_rule_set(-1, hist_len)
    player_names.append("always_defect")
    player_data.append(game_data)

    game_data = create_defect_rule_set(2, hist_len)
    player_names.append("always_cooperate")
    player_data.append(game_data)

    game_data = create_random_rule_set(8914534349193759)
    player_names.append("pure_random_choice")
    player_data.append(game_data)

    game_data = create_tit_for_tat_rule_set()
    player_names.append("tit_for_tat")
    player_data.append(game_data)

    game_data = convert_chrom_to_rule_set(
        'DCDCDCDDDCDCDDDCCCDCCDCCCCCCCCDCCDCCDDCDDCDCCDCCDCCDDDDCCDCCDCDDCDCDD'
        'CC')
    player_names.append("random_strategy_1")
    player_data.append(game_data)

    game_data = convert_chrom_to_rule_set(
        'CDDDCDCDDDDDCCCDCDDDDDCDDCDDCCDCCDCDCCDDCCDDDDCDDDCDCDCCDDDCCCDDDDDCD'
        'CC')
    player_names.append("random_strategy_2")
    player_data.append(game_data)

    game_data = convert_chrom_to_rule_set(
        'DDCCDDDCCCDCCCDDDCCCDCDCCDCCCCDCCCDDCDDCDCCDCDCDDCCDDDCDDCDDCDDDDDCDD'
        'CC')
    player_names.append("random_strategy_3")
    player_data.append(game_data)

    return player_data, player_names


def all_against_all(rounds: int) -> Tuple[List[str], NDArray, NDArray]:
    # Match everyone against everyone.
    game_data, player_names = create_all_players()
    players_amount: int = len(player_names)
    matrix_type = np.dtype([('row_player', int), ('column_player', int)])
    matrix_results = np.zeros(
        (players_amount, players_amount), dtype=matrix_type)
    total_results = np.zeros((players_amount), dtype=int)

    for i, (score_r, choice_r, history_r, outcome_r, reset_r) in \
            enumerate(game_data):
        for j, (score_c, choice_c, history_c, outcome_c, reset_c) in \
                enumerate(game_data):
            # Play a game against each other.
            play_tournament(rounds, choice_r, choice_c, history_r,
                            history_c, outcome_r, outcome_c)
            # Save outcomes both the total and the matrix version.
            total_results[i] += score_r.score
            total_results[j] += score_c.score
            matrix_results[i, j] = (score_r.score, score_c.score)
            # After a match reset both players.
            reset_r()
            reset_c()

    # Return the results.
    return player_names, matrix_results, total_results


def convert_struct_to_str(matrix: NDArray) -> NDArray:
    """Helper function to convert a complex matrix to a simple string
    elements."""
    # Create an empty matrix of strings with the same shape as the original
    # data.
    string_matrix = np.empty(matrix.shape, dtype=np.dtype('<U20'))

    # Convert each element to a string and fill the string matrix.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row, column = matrix[i, j]
            string_matrix[i, j] = f"{row}, {column}"

    return string_matrix


def print_best_players(best_players, player_names, total_results):
    for i, player in enumerate(best_players):
        print(f"#{i+1:>2}: {player_names[player]:^21} - "
              f"{total_results[player]:^4}")

def save_result_matrix(fname, player_names, matrix_results):
    rows = np.array(player_names, dtype='|S20')
    columns = np.concatenate((['(row player, column player)'], rows))
    # Convert data matrix to string matrix.
    matrix_results = convert_struct_to_str(matrix_results)
    np.savetxt(fname, np.vstack((columns[np.newaxis, :],
                                 np.hstack((rows[:, np.newaxis],
                                            matrix_results)))),
               delimiter=";", fmt="%s")


def main():
    rounds: int = 50
    player_names, matrix_results, total_results = all_against_all(rounds)
    best_players = np.argsort(total_results)[::-1]
    print(f"Results for default rule table and {rounds} rounds.\n\n"
          "Best scoring players and strategies in descending order.\n"
          "Format is as follows: #<rank>: <player name> - <total points>\n")
    print_best_players(best_players, player_names, total_results)

    print("\nThe full raw match matrix will now be saved as a text file, a "
          "index to name conversion table will be provided below.")
    fname: str = "matrix_results_1.csv"
    save_result_matrix(fname, player_names, matrix_results)

    # Change rule table and play less rounds.
    rewards['CC'] = (1, 1)
    rewards['DD'] = (5, 5)
    rewards['CD'] = (0, 3)
    rewards['DC'] = (3, 0)
    rounds = 25
    player_names, matrix_results, total_results = all_against_all(rounds)
    best_players = np.argsort(total_results)[::-1]
    print(f"Results saved as: '{fname}'.\n\n"
          "A modified rule table will now be used, rounds changed to: "
          f"{rounds}.\n"
          "Results will now be printed, format is the same as before.\n")
    print_best_players(best_players, player_names, total_results)

    print("\nThe score matrix of this tournament will also be saved.")
    fname: str = "matrix_results_2.csv"
    save_result_matrix(fname, player_names, matrix_results)
    print(f"Results saved as: '{fname}'.\n")

    print("Index to name mapping.\n")
    for i, name in enumerate(player_names):
        print(f"'{i}' is '{name}'")


if __name__ == '__main__':
    main()
