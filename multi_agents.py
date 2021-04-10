import numpy as np
import abc
import util
from game import Agent
from scipy.spatial import distance
import time
import heatmaps_constants
import random

RAND_MOVE_PROB = 0.25
TERMINAL_DEPTH = 0
NO_ACTION = []


class MCTSHelperAgent(Agent):
    def __init__(self, eval_func=0):
        super().__init__()

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.
        get_action chooses among the best options according to the evaluation function.
        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions(game_state.get_current_player())

        forward_moves = [move for move in legal_moves if
                         not is_going_backwards(8, move, game_state.get_current_player())]

        if len(forward_moves) != 0:
            legal_moves = forward_moves
        if len(legal_moves) == 0:
            return NO_ACTION
        if RAND_MOVE_PROB > random.random():
            return furthest_move(legal_moves)

        index = np.random.choice(range(len(legal_moves)))
        return legal_moves[index]


def better_mcts(state, action, maximazing_player):
    max_dist = max_dist_reflex_eval_func(state, action)
    tail_over_head = tail_over_head_reflex_eval_func(state, action)
    value = max_dist + 1.28 * tail_over_head
    if maximazing_player == 1:
        return value
    else:
        return value * (-1)


def MCTS_play_outs_eval(state, maximizing_player):
    return euclidean_minimize_eval_func(state, maximizing_player) + \
           0.14 * avoid_lone_wolf_eval_func(state, maximizing_player)


def state_in_end_game(state):
    pieces_in_goal_1 = [piece for piece in state._goal_camps[1] if state._board[piece[0], piece[1]] == 1]
    pieces_in_goal_1 = len(pieces_in_goal_1) / len(state._goal_camps[1])
    pieces_in_goal_2 = [piece for piece in state._goal_camps[2] if state._board[piece[0], piece[1]] == 2]
    pieces_in_goal_2 = len(pieces_in_goal_2) / len(state._goal_camps[2])

    return pieces_in_goal_1 >= 0.8 or pieces_in_goal_2 >= 0.8


def player_with_more_pieces(state):
    pieces_in_goal_1 = [piece for piece in state._goal_camps[1] if state._board[piece[0], piece[1]] == 1]
    pieces_in_goal_2 = [piece for piece in state._goal_camps[2] if state._board[piece[0], piece[1]] == 2]

    if len(pieces_in_goal_1) > len(pieces_in_goal_2):
        return 1
    elif len(pieces_in_goal_1) == len(pieces_in_goal_2):
        return np.random.choice([1, 2])
    else:
        return 2


def drop_bad_actions(state, actions):
    player = state.get_current_player()

    if player == 1:

        return [action for action in actions if action[1][0] - action[0][0] >= 0 and
                action[1][1] - action[0][1] >= 0]
    else:
        return [action for action in actions if action[0][0] - action[1][0] >= 0 and
                action[0][1] - action[1][1] >= 0]


def furthest_move(moves):
    index = np.argmax([distance.euclidean(action[0], action[1]) for action in moves])
    return moves[index]


def dist_to_camp_difference(current_game_state):
    result = 0

    for piece in current_game_state._agents_pieces[1]:
        if piece not in current_game_state._goal_camps[1]:
            distances = [distance.euclidean(piece, goal) for goal in current_game_state._goal_camps[1] if
                         current_game_state._board[goal[0], goal[1]] != 1]

            result -= np.average(distances)

    for piece in current_game_state._agents_pieces[2]:
        if piece not in current_game_state._goal_camps[2]:
            distances = [distance.euclidean(piece, goal) for goal in current_game_state._goal_camps[2] if
                         current_game_state._board[goal[0], goal[1]] != 2]
            result += np.average(distances)

    return result


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self, eval_func):
        self.evaluation_function = eval_func

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.
        get_action chooses among the best options according to the evaluation function.
        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions(game_state.get_current_player())

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]


def max_dist_reflex_eval_func(current_game_state, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (GameState.py) and returns a number, where higher numbers are better.

    yoav: this ev. function returns 0 if the player is going backwards, and the euclidean distance from src to dst
    otherwise
    """
    cur_player = current_game_state.get_current_player()

    if is_going_backwards(current_game_state.size, action, cur_player):
        return 0

    return distance.euclidean(action[0], action[1])


def tail_over_head_reflex_eval_func(current_game_state, action):
    """
    Checks if the last marble (tail) advanced in a way that it became the first one (head).
    :param current_game_state: A game_state object represents the current game state.
    :param action: A tuple of two coordinates. The first represents the old place and the second represents the place
    the piece has reached.
    :return:
    """
    bases = np.array([[0, 0], [current_game_state.size, current_game_state.size]])
    # Sort the pieces by tailiness.
    pieces = [x for x in current_game_state._agents_pieces[current_game_state._current_player]]
    pieces_norm = [np.linalg.norm(x - bases[current_game_state._current_player - 1]) for x in pieces]
    head_idx = np.argmax(pieces_norm)
    head = pieces[head_idx]

    # Calculate the distance of the premove position from the current head minus the distance of the postmove
    # position from the current head.
    eval = np.linalg.norm(head - np.asarray(action[0])) - np.linalg.norm(head - np.asarray(action[1]))
    return eval


def better_reflex(current_game_state, action):
    max_dist = max_dist_reflex_eval_func(current_game_state, action)
    tail_over_head = tail_over_head_reflex_eval_func(current_game_state, action)
    return 1 * max_dist + 1.28 * tail_over_head  # Best was 1.28 * tail_over_head


def is_going_backwards(board_size, action, cur_player):
    """
    Helper function form max_dist_reflex_eval_func
    returns true if the action is a solider going backwards (inclusive)
    """
    src_row = action[0][0]
    src_col = action[0][1]
    dest_row = action[1][0]
    dest_col = action[1][1]

    if cur_player == 1:
        if src_row + src_col >= board_size - 1:
            # src is after half
            if src_row > dest_row or src_col > dest_col:
                return True
        else:
            # src is before the half
            if src_row >= dest_row or src_col >= dest_col:
                return True

    if cur_player == 2:
        if src_row + src_col <= board_size - 1:
            # src is after half
            if src_row < dest_row or src_col < dest_col:
                return True
        else:
            # src is before the half
            if src_row <= dest_row or src_col <= dest_col:
                return True

    return False


def piece_before_half(state):
    """
    :return: true if there exists a player on current's player team before the half-line
    """
    cur_player = state.get_current_player()
    pieces = state._agents_pieces[cur_player]
    board_size = state.size

    if cur_player == 1:
        for piece in pieces:
            if piece[0] + piece[1] < board_size - 1:
                return True

    elif cur_player == 2:
        for piece in pieces:
            if piece[0] + piece[1] > board_size - 1:
                return True

    return False


def ignore_action_in_minimax(state, action):
    """
    Ignore the actions that meets this criteria:
    1. action source is not inside a camp AND action is moving backwards OR
    2. action source is inside a camp AND moving backwards AND there is a player before the half

    (NOT A and B) or (A and B and C) ---> B and (NOT A or C)
    """
    src_row = action[0][0]
    src_col = action[0][1]
    board_size = state.size
    cur_player = state.get_current_player()

    if is_going_backwards(board_size, action, cur_player):
        if [src_row, src_col] not in state._goal_camps[cur_player] or piece_before_half(state):
            return True

    return False


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, maximizing_player, evaluation_function='scoreEvaluationFunction', depth=2, time_limit=10000):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth
        self.maximizing_player = maximizing_player

        self.time_limit = time_limit  # time limit per move
        self._max_time_for_next_move = 0  # gets updated in get_action

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def max_value(self, state, depth):
        actions = state.get_legal_actions(state.get_current_player())

        # If state is a leaf.
        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        # max_state = 0
        max_value = np.NINF
        # max_action = 0

        candidates = [[0, np.NINF, 0]]
        rnd_index = 0

        for curr_action in actions:

            successor_state = state.generate_successor(curr_action)
            successor_value = self.min_value(successor_state, depth - 1)[2]

            if successor_value > max_value:
                candidates = [[successor_state, curr_action, successor_value]]
                max_value = successor_value
            elif successor_value == max_value:
                candidates.append([successor_state, curr_action, successor_value])

            rnd_index = np.random.randint(0, len(candidates))

        return candidates[rnd_index]

    def min_value(self, state, depth):
        actions = state.get_legal_actions(state.get_current_player())

        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        min_value = np.inf

        candidates = [[0, np.inf, 0]]
        rnd_index = 0

        for curr_action in actions:
            # state_copy = deepcopy(state)

            successor_state = state.generate_successor(curr_action)
            successor_value = self.max_value(successor_state, depth - 1)[2]

            if successor_value < min_value:
                candidates = [[successor_state, curr_action, successor_value]]
                min_value = successor_value
            elif successor_value == min_value:
                candidates.append([successor_state, curr_action, successor_value])

            rnd_index = np.random.randint(0, len(candidates))

        return candidates[rnd_index]

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        self._max_time_for_next_move = time.time() + self.time_limit
        best_action = self.max_value(game_state, self.depth)
        return best_action[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_value(self, state, depth, alpha, beta):
        actions = state.get_legal_actions(state.get_current_player())
        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        max_value = np.NINF

        candidates = [[0, np.NINF, 0]]
        rnd_index = 0

        # Move Ordering - Descending
        actions.sort(key=lambda x: max_dist_reflex_eval_func(state, x), reverse=True)

        for curr_action in actions:
            # IgnoreAction feature. last condition is to ensure some action will be made and we won't skip all
            if self.depth >= 3 and candidates[0] != [0, np.NINF, 0] and ignore_action_in_minimax(state, curr_action):
                continue
            successor_state = state.generate_successor(curr_action)
            successor_value = self.min_value(successor_state, depth - 1, alpha,
                                             beta)[2]

            if successor_value > max_value:
                candidates = [[successor_state, curr_action, successor_value]]
                max_value = successor_value
            elif successor_value == max_value:
                candidates.append([successor_state, curr_action, successor_value])

            rnd_index = np.random.randint(0, len(candidates))

            if max_value >= beta:
                return candidates[rnd_index]
            alpha = np.maximum(alpha, max_value)

        return candidates[rnd_index]

    def min_value(self, state, depth, alpha, beta):
        actions = state.get_legal_actions(state.get_current_player())

        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        # min_state = 0
        min_value = np.inf
        # min_action = 0

        candidates = [[0, np.inf, 0]]
        rnd_index = 0

        # Move Ordering - Ascending - https://stackoverflow.com/a/9964572
        actions.sort(key=lambda x: max_dist_reflex_eval_func(state, x), reverse=False)

        for curr_action in actions:
            # IgnoreAction feature. last condition is to ensure some action will be made and we won't skip all
            if self.depth >= 3 and candidates[0] != [0, np.inf, 0] and ignore_action_in_minimax(state, curr_action):
                continue
            successor_state = state.generate_successor(curr_action)
            successor_value = self.max_value(successor_state, depth - 1, alpha,
                                             beta)[2]

            if successor_value < min_value:
                candidates = [[successor_state, curr_action, successor_value]]
                min_value = successor_value
            elif successor_value == min_value:
                candidates.append([successor_state, curr_action, successor_value])

            rnd_index = np.random.randint(0, len(candidates))

            if min_value <= alpha:
                return candidates[rnd_index]

            beta = np.minimum(beta, min_value)

        return candidates[rnd_index]

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        self._max_time_for_next_move = time.time() + self.time_limit

        best_action = self.max_value(game_state, self.depth, np.NINF, np.inf)

        if best_action[1] == NO_ACTION:
            return NO_ACTION

        return best_action[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def max_value(self, state, depth):

        actions = state.get_legal_actions(state.get_current_player())

        # If state is a leaf.
        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        # max_state = 0
        max_value = np.NINF
        # max_action = 0

        candidates = [[0, np.NINF, 0]]
        rnd_index = 0

        for curr_action in actions:

            successor_state = state.generate_successor(curr_action)
            successor_value = self.min_value(successor_state, depth - 1)[2]

            if successor_value > max_value:
                candidates = [[successor_state, curr_action, successor_value]]
                max_value = successor_value
            elif successor_value == max_value:
                candidates.append([successor_state, curr_action, successor_value])

            rnd_index = np.random.randint(0, len(candidates))

        return candidates[rnd_index]

    def min_value(self, state, depth):
        actions = state.get_legal_actions(state.get_current_player())

        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        min_state = 0
        sum_values = 0
        min_action = 0

        for curr_action in actions:
            successor_state = state.generate_successor(curr_action)
            successor_value = self.max_value(successor_state, depth - 1)[2]

            sum_values += successor_value
            min_state = successor_state
            min_action = curr_action

        sum_values /= len(actions)
        return [min_state, min_action, sum_values]

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        self._max_time_for_next_move = time.time() + self.time_limit

        best_action = self.max_value(game_state, self.depth)

        return best_action[1]


class IterativeDeepeningAlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_value(self, state, depth, alpha, beta):
        actions = state.get_legal_actions(state.get_current_player())
        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        # max_state = 0
        max_value = np.NINF
        # max_action = 0

        candidates = [[0, np.NINF, 0]]
        rnd_index = 0

        # Move Ordering - Descending
        actions.sort(key=lambda x: max_dist_reflex_eval_func(state, x), reverse=True)

        for curr_action in actions:
            # IgnoreAction feature. last condition is to ensure some action will be made and we won't skip all
            if self.depth >= 3 and candidates[0] != [0, np.NINF, 0] and ignore_action_in_minimax(state, curr_action):
                continue
            successor_state = state.generate_successor(curr_action)
            successor_value = self.min_value(successor_state, depth - 1, alpha,
                                             beta)[2]

            if successor_value > max_value:
                candidates = [[successor_state, curr_action, successor_value]]
                max_value = successor_value
            elif successor_value == max_value:
                candidates.append([successor_state, curr_action, successor_value])

            rnd_index = np.random.randint(0, len(candidates))

            if max_value >= beta:
                return candidates[rnd_index]
            alpha = np.maximum(alpha, max_value)

        return candidates[rnd_index]

    def min_value(self, state, depth, alpha, beta):
        actions = state.get_legal_actions(state.get_current_player())

        if depth == TERMINAL_DEPTH or len(actions) == 0 or time.time() > self._max_time_for_next_move:
            return [state, 0, self.evaluation_function(state, self.maximizing_player)]

        # min_state = 0
        min_value = np.inf
        # min_action = 0

        candidates = [[0, np.inf, 0]]
        rnd_index = 0

        # Move Ordering - Ascending - https://stackoverflow.com/a/9964572
        actions.sort(key=lambda x: max_dist_reflex_eval_func(state, x), reverse=False)

        for curr_action in actions:
            # IgnoreAction feature. last condition is to ensure some action will be made and we won't skip all
            if self.depth >= 3 and candidates[0] != [0, np.inf, 0] and ignore_action_in_minimax(state, curr_action):
                continue
            successor_state = state.generate_successor(curr_action)
            successor_value = self.max_value(successor_state, depth - 1, alpha,
                                             beta)[2]

            if successor_value < min_value:
                candidates = [[successor_state, curr_action, successor_value]]
                min_value = successor_value
            elif successor_value == min_value:
                candidates.append([successor_state, curr_action, successor_value])

            rnd_index = np.random.randint(0, len(candidates))

            if min_value <= alpha:
                return candidates[rnd_index]

            beta = np.minimum(beta, min_value)

        return candidates[rnd_index]

    def get_action(self, game_state, w_1=1, w_2=1):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        self._max_time_for_next_move = time.time() + self.time_limit

        for i in range(1, self.depth + 1):  # for example depth = 3 implies 1,2,3
            best_action = self.max_value(game_state, i, np.NINF, np.inf)
            if best_action[1] == NO_ACTION:
                print("NO ACTION")
                continue
            successor_state = game_state.generate_successor(best_action[1])
            if successor_state.get_winner_from_state() == self.maximizing_player:
                break

        if best_action[1] == NO_ACTION:
            return NO_ACTION

        return best_action[1]


def diagonal_convergence(current_game_state, maximizing_player):
    pieces_difference_1 = [abs(x[0] - x[1]) for x in current_game_state._agents_pieces[1]]
    pieces_difference_2 = [abs(x[0] - x[1]) for x in current_game_state._agents_pieces[2]]

    sum_differences_1 = np.sum(pieces_difference_1)
    sum_differences_2 = np.sum(pieces_difference_2)

    result = sum_differences_2 - sum_differences_1

    if maximizing_player == 2:
        result *= -1

    return result


def heatmap_forward_eval_func(current_game_state, maximizing_player):
    result = 0

    board = current_game_state._board

    if maximizing_player == 1:
        for i in range(0, len(board)):
            for j in range(0, len(board[0])):
                if len(board) == 8:
                    if board[i][j] == 1:
                        result += heatmaps_constants.HEATMAP_RED_8[i][j]
                    if board[i][j] == 2:
                        result -= heatmaps_constants.HEATMAP_RED_AD_8[i][j]
                if len(board) == 10:
                    if board[i][j] == 1:
                        result += heatmaps_constants.HEATMAP_RED_10[i][j]
                    if board[i][j] == 2:
                        result -= heatmaps_constants.HEATMAP_RED_AD_10[i][j]
    if maximizing_player == 2:
        for i in range(0, len(board)):
            for j in range(0, len(board[0])):
                if len(board) == 8:
                    if board[i][j] == 2:
                        result += heatmaps_constants.HEATMAP_GREEN_8[i][j]
                    if board[i][j] == 1:
                        result -= heatmaps_constants.HEATMAP_GREEN_AD_8[i][j]
                if len(board) == 10:
                    if board[i][j] == 2:
                        result += heatmaps_constants.HEATMAP_GREEN_10[i][j]
                    if board[i][j] == 1:
                        result -= heatmaps_constants.HEATMAP_GREEN_AD_10[i][j]

    return result


def avoid_lone_wolf_eval_func(current_game_state, maximizing_player):
    result1 = 0
    result2 = 0

    board = current_game_state._board
    for i in range(0, len(board)):
        for j in range(0, len(board[0])):
            if board[i][j] == 1:
                if j < len(board) - 1 and board[i][j + 1] != 0:
                    result1 += 1
                if i < len(board) - 1 and board[i + 1][j] != 0:
                    result1 += 1
                if (i < len(board[0]) - 1) and j < (len(board[0]) - 1) and board[i + 1][j + 1] != 0:
                    result1 += 1
            if board[i][j] == 2:
                if j > 0 and board[i][j - 1] != 0:
                    result2 += 1
                if i > 0 and board[i - 1][j] != 0:
                    result2 += 1
                if i > 0 and j > 0 and board[i - 1][j - 1] != 0:
                    result2 += 1

    if maximizing_player == 1:
        result = result1 - result2
    else:
        result = result2 - result1

    return result


def euclidean_minimize_eval_func(current_game_state, maximizing_player):
    """
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
	"""
    result = 0

    for piece in current_game_state._agents_pieces[1]:
        if piece not in current_game_state._goal_camps[1]:
            distances = [distance.euclidean(piece, goal) for goal in current_game_state._goal_camps[1] if
                         current_game_state._board[goal[0], goal[1]] != 1]

            result -= max(distances)

    for piece in current_game_state._agents_pieces[2]:
        if piece not in current_game_state._goal_camps[2]:
            distances = [distance.euclidean(piece, goal) for goal in current_game_state._goal_camps[2] if
                         current_game_state._board[goal[0], goal[1]] != 2]
            result += max(distances)

    if maximizing_player == 2:
        result *= -1

    return result


def f1(current_game_state, maximizing_player):
    """
    Your extreme Halma evaluation function.
    :param current_game_state: A game_state object represents the current game state.
    :param maximizing_player: The number represents the player who is the root of the tree (through whom eyes are we looking), 1 or 2.
    :return: A better evaluation for a given state.
    """
    return 1 * euclidean_minimize_eval_func(current_game_state, maximizing_player) + 0.14 * avoid_lone_wolf_eval_func(
        current_game_state, maximizing_player)


def f2(current_game_state, maximizing_player):
    """
    Your extreme Halma evaluation function.
    :param current_game_state: A game_state object represents the current game state.
    :param maximizing_player: The number represents the player who is the root of the tree (through whom eyes are we looking), 1 or 2.
    :return: A better evaluation for a given state.
    """
    return 1 * euclidean_minimize_eval_func(current_game_state,
                                            maximizing_player) + 0.25 * euclidean_minimize_eval_func(current_game_state,
                                                                                                     maximizing_player)


def f3(current_game_state, maximizing_player):
    """
    Your extreme Halma evaluation function.
    :param current_game_state: A game_state object represents the current game state.
    :param maximizing_player: The number represents the player who is the root of the tree (through whom eyes are we looking), 1 or 2.
    :return: A better evaluation for a given state.
    """
    return 1 * heatmap_forward_eval_func(current_game_state, maximizing_player) + 2.2 * euclidean_minimize_eval_func(
        current_game_state, maximizing_player)


def f4(current_game_state, maximizing_player):
    """
    Your extreme Halma evaluation function.
    :param current_game_state: A game_state object represents the current game state.
    :param maximizing_player: The number represents the player who is the root of the tree (through whom eyes are we looking), 1 or 2.
    :return: A better evaluation for a given state.
    """
    return 1 * heatmap_forward_eval_func(current_game_state, maximizing_player) + 1.9 * avoid_lone_wolf_eval_func(
        current_game_state, maximizing_player)
