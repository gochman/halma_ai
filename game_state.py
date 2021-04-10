import numpy as np
import copy

TABLE_SIZE = 8

FIRST_PLAYER = 1
SECOND_PLAYER = 2
EMPTY_TILE = 0
NO_ACTION = []


class Vector:
    """
    The directions in respect to a matrix which where the cell (0,0) is in the top left corner!
    """
    DOWN = np.array([1, 0])
    UP = np.array([-1, 0])
    RIGHT = np.array([0, 1])
    LEFT = np.array([0, -1])
    MAIN_DIAG_UP = UP + LEFT
    MAIN_DIAG_DOWN = DOWN + RIGHT
    SEC_DIAG_UP = UP + RIGHT
    SEC_DIAG_DOWN = DOWN + LEFT

    @staticmethod
    def get_vectors():
        return [vector for direction, vector in vars(Vector).items() if not direction.startswith('__')
                and not direction.startswith('g')]


"""
Important note: We're gonna assume that an action is a tuple with the first element being a coordinate of the cell 
to be moved from and the second element is the coordinate to be moved at. 
It's not the best idea since we'll probably would like to convert them into cartesian coords later, but 
"""

"""
The logic of finding possible actions can be debugged in this file by changing the _init_game function with desired
pieces locations.
"""


class GameState(object):
    # TODO: Translate dict of actions to be single coordinate of cell (row * size + col) - might make things faster
    #  instead of storing array of arrays...

    def __init__(self, done=False, score=0, size=TABLE_SIZE, init=True, current_player=None,
                 board=None, pieces_locations=None, goal_camps=None, start_from_state=False):
        super(GameState, self).__init__()

        self.done = done
        self.score = score
        self.size = size
        self._init = init
        self._current_player = current_player

        # This happened when we use generate successor:
        if self._init and not start_from_state:
            self._board = board
            self._agents_pieces = pieces_locations
            self._goal_camps = goal_camps


        # This happen in case we try to init the game for the first time or we trying to init from certain state:
        else:
            # Dict consists of Key:  Number of player; Value: array of indices (row * size + col)
            self._agents_pieces = dict()
            self._goal_camps = dict()

            self._goal_camps[FIRST_PLAYER] = []
            self._agents_pieces[FIRST_PLAYER] = []
            self._goal_camps[SECOND_PLAYER] = []
            self._agents_pieces[SECOND_PLAYER] = []

            self._current_player = 1


        if not self._init:
            self._board = np.zeros((TABLE_SIZE, TABLE_SIZE))
            self._init_game()
            self._init = True

        elif start_from_state:
            self._board = board
            self._init_from_state()


    def _init_goal_camps(self):
        """
        Helper function for init from state in order to initialize only the camps.
        :return:
        """
        if self.size == 10 or self.size == 8:
            for j in range(4):
                for i in range(4 - j):
                    # self._goal_camps[SECOND_PLAYER].add(i * self.size + j)
                    # self._goal_camps[FIRST_PLAYER].add((self._board.shape[1] - i - 1) *
                    #                                    self.size + self._board.shape[1] - j - 1)
                    self._goal_camps[SECOND_PLAYER].append([i, j])
                    self._goal_camps[FIRST_PLAYER].append([self._board.shape[1] - i - 1, self._board.shape[1] - j - 1])

                    if j == 0 and i == 4:  # First 2 columns have same number of pieces.
                        break

        elif self.size == 16:
            for j in range(5):
                for i in range(6 - j):
                    # self._goal_camps[SECOND_PLAYER].add(i * self.size + j)
                    #
                    # self._goal_camps[FIRST_PLAYER].add((self._board.shape[1] - i - 1) *
                    #                                    self.size + self._board.shape[1] - j - 1)

                    self._goal_camps[SECOND_PLAYER].append([i, j])
                    self._goal_camps[FIRST_PLAYER].append([self._board.shape[1] - i - 1, self._board.shape[1] - j - 1])

                    if j == 0 and i == 4:  # First 2 columns have same number of pieces.
                        break

    def _init_game(self):
        """
        Method which initialize pieces of each agent, each agent's goal camp cells and the board itself.
        """
        if self.size == 10 or self.size == 8:
            for j in range(4):
                for i in range(4 - j):
                    """
                    Uncomment these for normal game (if not testing).
                    """
                    self._board[i, j] = FIRST_PLAYER
                    self._board[self._board.shape[1] - i - 1, self._board.shape[1] - j - 1] = SECOND_PLAYER

                    # self._goal_camps[SECOND_PLAYER].add(i * self.size + j)
                    # self._agents_pieces[FIRST_PLAYER].add(i * self.size + j)

                    # self._goal_camps[FIRST_PLAYER].add((self._board.shape[1] - i - 1) *
                    #                                    self.size + self._board.shape[1] - j - 1)
                    # self._agents_pieces[SECOND_PLAYER].add((self._board.shape[1] - i - 1) *
                    #                                        self.size + self._board.shape[1] - j - 1)

                    self._goal_camps[SECOND_PLAYER].append([i, j])
                    self._agents_pieces[FIRST_PLAYER].append([i, j])

                    self._goal_camps[FIRST_PLAYER].append([self._board.shape[1] - i - 1, self._board.shape[1] - j - 1])
                    self._agents_pieces[SECOND_PLAYER].append(
                        [self._board.shape[1] - i - 1, self._board.shape[1] - j - 1])

                    if j == 0 and i == 4:  # First 2 columns have same number of pieces.
                        break

        elif self.size == 16:
            for j in range(5):
                for i in range(6 - j):
                    """
                    Uncomment these for normal game (if not testing).
                    """
                    self._board[i, j] = FIRST_PLAYER
                    self._board[self._board.shape[1] - i - 1, self._board.shape[1] - j - 1] = SECOND_PLAYER


                    self._goal_camps[SECOND_PLAYER].append([i, j])
                    self._agents_pieces[FIRST_PLAYER].append([i, j])

                    self._goal_camps[FIRST_PLAYER].append([self._board.shape[1] - i - 1, self._board.shape[1] - j - 1])
                    self._agents_pieces[SECOND_PLAYER].append(
                        [self._board.shape[1] - i - 1, self._board.shape[1] - j - 1])

                    if j == 0 and i == 4:  # First 2 columns have same number of pieces.
                        break


    def _init_from_state(self):
        self._init_goal_camps()


        for coord, x in np.ndenumerate(self._board):
            if x == FIRST_PLAYER:
                self._agents_pieces[FIRST_PLAYER].append(coord)
            elif x == SECOND_PLAYER:
                self._agents_pieces[SECOND_PLAYER].append(coord)

    def get_legal_actions(self, agent_index):
        """
        First we get the indices of agent's pieces and then we'll get all possible actions from each piece.
        :param agent_index: Player's number.
        :return: A dict with all legal actions that can be taken from curr state. Key - coordinate to be moved from.
        Value - coordinates of cells array that can be achieved from key cell.
        """
        legal_actions = []
        if agent_index != FIRST_PLAYER and agent_index != SECOND_PLAYER:
            raise Exception("illegal agent index.")
        else:
            for coord in self._agents_pieces[agent_index]:
                output = self._get_agent_legal_actions(coord[0], coord[1])
                legal_actions += [[coord, action] for action in output]

        return legal_actions

    def _visited_coordinate(self, new_coord, visited):
        return new_coord.tolist() in visited

    # def _visited_coordinate(self, new_coord, visited):
    #     return new_coord in visited

    def _illegal_camp_move(self, old_coord, vec):
        """
        This method checks that a piece does not move outside of goal camp (if it is in the camp)..
        :return: True if piece inside goal camp and move outside of it, false otherwise.
        """
        # Goal camp of player 1 is in the lower right corner and player 2 in the upper left corner.
        # old_val = self.size * old_coord[0] + old_coord[1]
        # new_val = self.size * (old_coord[0] + vec[0]) + (old_coord[1] + vec[1])
        #
        # return old_val in self._goal_camps[self._current_player] and \
        #        new_val not in self._goal_camps[self._current_player]

        old_in_camp = False
        new_in_camp = False
        new_coord = [old_coord[0] + vec[0], old_coord[1] + vec[1]]

        for goal in self._goal_camps[self._current_player]:
            if new_coord == goal:
                new_in_camp = True
            if old_coord[0] == goal[0] and old_coord[1] == goal[1]:
                old_in_camp = True

        return old_in_camp and not new_in_camp

    def _coordinate_in_board(self, row, col):
        """
        Check if coordinate in bounds of board.
        :param row: row index
        :param col: col index
        :return: True if in bounds, otherwise false.
        """
        return not (row >= self.size or col >= self.size or row < 0 or col < 0)

    def _valid_coordinate(self, old_coor, vec, visited):
        new_coord = old_coor + vec
        ret_value = self._coordinate_in_board(new_coord[0], new_coord[1]) and not \
            self._illegal_camp_move(old_coor, vec) and not \
                        self._visited_coordinate(new_coord, visited)
        return ret_value

    def _can_jump(self, cell, vector):
        """
        This method checks whether the neighbor of cell (in direction of vector) is not empty and the
        desired cell to hooped into is empty.
        :param cell: Cell which be hooped from.
        :param vector: Direction of the hoop.
        :return: True if the condition holds, false otherwise.
        """
        neighbor_row = int(vector[0] / 2)
        neighbor_col = int(vector[1] / 2)
        return self._board[cell[0] + neighbor_row, cell[1] + neighbor_col] != EMPTY_TILE and \
               self._board[cell[0] + vector[0], cell[1] + vector[1]] == EMPTY_TILE

    def _actions_by_hoops(self, cell, visited):
        new_actions = []
        for vector in Vector.get_vectors():
            longer_vec = vector * 2  # vector is only for neighbor cells.
            if self._valid_coordinate(cell, longer_vec, visited):
                new_coord = cell + longer_vec
                if self._can_jump(cell, longer_vec):
                    new_actions.append(new_coord)
                    visited.append(new_coord.tolist())
                    # visited.append(new_coord)
                    new_actions += self._actions_by_hoops(new_coord, visited)
        return new_actions

    def _actions_from_coordinate(self, cell):
        # TODO: The above function looks almost the same but is specific for hooping moves. I couldn't combine them
        #  into one method that doesn't look ridiculous.
        """
        Recursive function that checks conditions for each neighbor of the current cell and recurses to cells in the
        direction of occupied neighbors.
        :param row: Row index of curr cell.
        :param col: Col index of curr cell
        :param visited: List of coordinates of visited cells (in order to not recurse forever between two states..)
        :return: List of possible actions from current state as coordinates.
        """
        visited = []
        new_actions = []
        for vector in Vector.get_vectors():
            if self._valid_coordinate(cell, vector, visited):
                new_coord = cell + vector

                if self._board[new_coord[0], new_coord[1]] == EMPTY_TILE:
                    new_actions.append(new_coord)
                    visited.append(new_coord.tolist())

        new_actions += self._actions_by_hoops(cell, visited)
        return new_actions

    def _get_agent_legal_actions(self, row, col):
        if self._board[row][col] == EMPTY_TILE:
            print("There isn't a piece there to move.")
            return
        legal_moves = self._actions_from_coordinate([row, col])
        return legal_moves

    def print_board_colored(self, last_action_done=None):
        """

        :param last_action_done: last action that was made in the board
        :return: nothing

        https://godoc.org/github.com/whitedevops/colors
        """
        if last_action_done is None:
            src = [0, 0]
            dest = [0, 0]
        else:
            src = last_action_done[0]
            dest = last_action_done[1]

        for i in range(0, len(self._board)):
            for j in range(0, len(self._board[0])):
                if i == src[0] and j == src[1]:  # Yellow src square
                    print('\033[93m', '▫', '\033[0m', end="")
                elif i == dest[0] and j == dest[1]:
                    print('\033[93m', '●', '\033[0m', end="")  # Yellow dst sircle
                elif int(self._board[i][j] == 1):
                    print('\033[31m', '●', '\033[0m', end="")  # red player
                elif int(self._board[i][j] == 2):
                    print('\033[32m', '●', '\033[0m', end="")  # green player

                else:
                    print(' .', end=" ")  # white brick

            print("")

        print("\n------------------------------------")

    def print_board(self):
        print(self._board, "\n")

    def _is_legal_action(self, old_row, old_col, new_row, new_col):
        """
        Just checks if the coordinate in the limits of the board and it's empty. Does not check whether it can be
        achieved.
        :return: True if 'legal', otherwise false.
        """
        return self._coordinate_in_board(old_row, old_col) and self._coordinate_in_board(new_row, new_col) and \
               self._board[new_row, new_col] == EMPTY_TILE and self._board[old_row, old_col] == self._current_player

    def apply_action(self, action):
        if action == NO_ACTION:
            self.done = True
            return
        # Check some conditions to legality of an action

        old_row, old_col = action[0][0], action[0][1]
        new_row, new_col = action[1][0], action[1][1]

        if not self._is_legal_action(old_row, old_col, new_row, new_col):
            raise Exception("illegal action.")

        # Change the board.
        self._board[old_row, old_col] = EMPTY_TILE
        self._board[new_row, new_col] = self._current_player

        self._agents_pieces[self._current_player].remove([old_row, old_col])
        self._agents_pieces[self._current_player].append([new_row, new_col])

        camp_pieces = [self._board[coord[0], coord[1]] == self._current_player
                       for coord in self._goal_camps[self._current_player]]

        if np.all(camp_pieces):
            self.done = True

    def switch_player(self):
        self._current_player = self._current_player % 2 + 1

    def generate_successor(self, action):
        successor = GameState(done=self.done, score=self.score, init=self._init,
                              current_player=self._current_player, board=copy.deepcopy(self._board),
                              pieces_locations=copy.deepcopy(self._agents_pieces), goal_camps=self._goal_camps)
        successor.apply_action(action)
        successor.switch_player()
        return successor

    def get_current_player(self):
        return self._current_player

    def get_winner_from_state(self):
        """

        :return: 1 if player 1 reached goal in cur state
                 2 if player 2 reached goal in cur state
                 0 if game still occurs

        """
        p1_won = True
        p2_won = True

        for goal in self._goal_camps[FIRST_PLAYER]:
            if self._board[goal[0]][goal[1]] != FIRST_PLAYER:
                p1_won = False

        for goal in self._goal_camps[SECOND_PLAYER]:
            if self._board[goal[0]][goal[1]] != SECOND_PLAYER:
                p2_won = False

        if p1_won:
            return FIRST_PLAYER
        elif p2_won:
            return SECOND_PLAYER
        return 0



if __name__ == '__main__':
    game = GameState()
    print(game.print_board())
    # cell_to_find_actions_from = [3, 0]
    # actions = game._get_agent_legal_actions(cell_to_find_actions_from[0], cell_to_find_actions_from[1])
    # actions = [[key, values] for key, values in game.get_legal_actions(SECOND_PLAYER).items()]
    # action = [[actions[0][0], value] for value in actions[0][1]]
    # game.apply_action(action)
    # for action in actions:
    actions = game.get_legal_actions(SECOND_PLAYER)

    print(actions)
    # print(actions)
