"""
Code for Final Project

Finding The Optimal Halma Agent with Artificial Intelligence


Yoav Gochman
yoav.gochman@mail.huji.ac.il

Assa Kariv
assa.kariv@mail.huji.ac.il

Eyal Schaffer
eyal.schaffer@mail.huji.ac.il

Noam Delbari
noam.delbari@mail.huji.ac.il

Introduction to Artificial Intelligence 67842, Spring 2020
The Rachel and Selim Benin School of Computer Science and Engineering,
The Hebrew University of Jerusalem,
Jerusalem, Israel

"""



import abc

import game_state
import numpy as np
import util

LONG_GAME_THRESHOLD = 150


class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def get_action(self, game_state):
        return

    def stop_running(self):
        pass


class RandomOpponentAgent(Agent):

    def get_action(self, state, w_1=1, w_2=1):
        actions = state.get_legal_actions(state.get_current_player())
        if actions == game_state.NO_ACTION:
            print("NO ACTION TO DO FOR RANDOM PLAYER NUM: \n", state.get_current_player())
            return actions
        return actions[np.random.choice(range(len(actions)))]


class Game(object):
    def __init__(self, agent, opponent_agent, first_name, second_name):
        super(Game, self).__init__()
        self.agent = agent
        self.opponent_agent = opponent_agent
        self._state = None
        self._should_quit = False
        self._winner = None
        self._move_counter = 0
        self._long_game = False
        self.first_name = first_name
        self.second_name = second_name

    def run(self, initial_state):
        self._should_quit = False
        self._state = initial_state
        return self._game_loop()

    def quit(self):
        self._should_quit = True
        self.agent.stop_running()
        self.opponent_agent.stop_running()

    def tie_breaker(self):
        """
        This function is called after LONG_GAME_THRESHOLD is reached. takes the current state and determines who is the
        winner according to who has more soldiers after the middle line. (middle coordinates are col + row = board size -1)
        in case of tie again - draw a coin (almost never happens - just in case of random vs random)


        :return:1 or 2, according to the winning player
        """
        p1_counter = 0
        p2_counter = 0

        pieces_1 = [x for x in self._state._agents_pieces[1]]
        pieces_2 = [x for x in self._state._agents_pieces[2]]

        for piece in pieces_1:
            if piece[0] + piece[1] > self._state.size - 1:
                p1_counter += 1

        for piece in pieces_2:
            if piece[0] + piece[1] < self._state.size - 1:
                p2_counter += 1

        result = 0
        if p1_counter == p2_counter:
            result = np.random.randint(1, 3)  # 1 or 2
        elif p1_counter > p2_counter:
            result = 1
        else:
            result = 2

        return result

    def _game_loop(self):

        while not self._state.done and not self._should_quit:
            self._move_counter += 1
            if self._move_counter == LONG_GAME_THRESHOLD:
                self._long_game = True
                self._winner = self.tie_breaker()
                print("Winner is player:", self._winner, "by tie-breaker function")
                return self._winner

            if self._state.get_current_player() == game_state.FIRST_PLAYER:
                print("Move", self._move_counter, "-", self.first_name)
                action = self.agent.get_action(self._state)
            else:
                print("Move", self._move_counter, "-", self.second_name)
                action = self.opponent_agent.get_action(self._state)

            self._state.apply_action(action)
            self._state.switch_player()
            self._state.print_board_colored(action)

        return 3 - self._state.get_current_player()


def create_agent(agent, depth, time_limit, func, player_playing):
    agent_name = 'No Player'
    if agent == 0:
        agent_name = 'Random Player'
        agent = RandomOpponentAgent()

    elif agent == 1:
        func = 'better_reflex'
        agent_name = 'Reflexive player'
        from multi_agents import ReflexAgent
        agent = ReflexAgent(util.lookup('multi_agents.' + func, globals()))

    elif agent == 2:
        agent = 'MinmaxAgent'
        agent_name = 'Min-Max player'
        agent = util.lookup('multi_agents.' + agent, globals())(depth=depth, evaluation_function=func,
                                                                maximizing_player=player_playing, time_limit=time_limit)
    elif agent == 3:
        agent = 'AlphaBetaAgent'
        agent_name = 'Alpha-Beta player'
        agent = util.lookup('multi_agents.' + agent, globals())(depth=depth, evaluation_function=func,
                                                                maximizing_player=player_playing, time_limit=time_limit)
    elif agent == 4:
        agent = 'ExpectimaxAgent'
        agent_name = 'ExpectiMax player'
        agent = util.lookup('multi_agents.' + agent, globals())(depth=depth, evaluation_function=func,
                                                                maximizing_player=player_playing, time_limit=time_limit)
    elif agent == 5:
        agent = 'IterativeDeepeningAlphaBetaAgent'
        agent_name = 'Iterative Deepening Alpha-Beta player'
        agent = util.lookup('multi_agents.' + agent, globals())(depth=depth, evaluation_function=func,
                                                                maximizing_player=player_playing, time_limit=time_limit)
    elif agent == 6:
        from multi_agents import MCTSHelperAgent
        from MCTS import MCTSAgent
        model = MCTSHelperAgent()
        path = 'MCTS_DB.npy'
        agent = MCTSAgent(model, iteration_num=depth, history_table_path=path)
        agent_name = "Monte Carlo Tree Search player"

    return agent, agent_name


def new_game(args):
    first_player_args = args[0]
    second_player_args = args[1]

    first_agent, first_name = create_agent(first_player_args[0], first_player_args[1], first_player_args[2],
                                           first_player_args[3], 1)
    second_agent, second_name = create_agent(second_player_args[0], second_player_args[1], second_player_args[2],
                                             second_player_args[3], 2)

    initial_state = game_state.GameState(init=False)
    game = Game(first_agent, second_agent, first_name, second_name)

    return game.run(initial_state)


if __name__ == '__main__':

    ########################################################################################################################
    inputs = []
    print("Hello, welcome to the game of Halma\n")

    for i in range(1, 3):
        agent_inputs = []
        agent_number_selected = False

        while not agent_number_selected:
            print("Choose agent ", i, " number:")
            print("Random Agent(0) ; Reflex Agent(1) ; Min-Max Agent(2) ; Alpha-Beta Agent(3) ; ExpectiMax Agent(4) ; "
                  " Iterative-Deepening Alpha-Beta(5) ; MCTS Agent(6)")

            agent_num = input()
            agent_num = int(agent_num)

            if agent_num < 0 or agent_num > 6:
                print("No such agent, check your answer and try again")
                continue
            else:
                agent_inputs.append(agent_num)
                agent_number_selected = True

        agent_depth_selected = False

        if agent_inputs[0] in [0, 1]:
            agent_inputs.append(0)  # just a place holder for depth
            agent_inputs.append(0)  # just a place holder for time limit
            agent_inputs.append(0)  # just a place holder for eval func

            inputs.append(agent_inputs)
            print("-------")
            continue

        while not agent_depth_selected:
            print("Choose depth of search agent or number of play outs to MCTS: ")
            print("Depths: 1, 2, 3")
            print("Iterations: 250, 450, 750, 1200, 2000")

            depth = input()
            depth = int(depth)

            if depth not in [1, 2, 3, 250, 450, 750, 1200, 2000]:
                print("No such depth or iteration number, check your answer and try again..:")
                continue
            else:
                agent_inputs.append(depth)
                agent_depth_selected = True


        if agent_inputs[0] == 6:  # If agent is MCTS
            agent_inputs.append(10000) # just a placeholder, time_limit is irrelevant
            agent_inputs.append("f1")  # just placeholder, func is irrelevant
            inputs.append(agent_inputs)
            print("-------")
            continue

        time_limit_selected = False


        while not time_limit_selected:
            print("Choose time limit for each move: ")
            print("Time limits in seconds: 7, 25, 40, 120, 10000")

            time_limit = input()
            time_limit = int(time_limit)

            if time_limit != 7 and time_limit != 25 and time_limit != 40 and time_limit != 120 and time_limit != 10000:
                print("No such time limit, check your answer and try again..:")
                continue
            else:
                agent_inputs.append(time_limit)
                time_limit_selected = True

        eval_func_selected = False



        while not eval_func_selected:
            print("Choose evaluation function for the search agent: ")
            print("functions are : f1, f2, f3, f4")
            print("\n- f1 is Euclidean w/ Lone-Wolf")
            print("- f2 is Euclidean w/ Diagonal")
            print("- f3 is Euclidean w/ Heatmap")
            print("- f4 is Heatmap w/ Lone-Wolf")

            func = input()

            if func != "f1" and func != "f2" and func != "f3" and func != "f4":
                print("No such evaluation function, check your answer and try again...:")
                continue
            else:
                agent_inputs.append(func)
                eval_func_selected = True

        inputs.append(agent_inputs)
        print("-------")

    first_player = inputs[0]
    second_player = inputs[1]
    print("Starts the game...")

    winner = new_game(inputs)
    print("Winner is player", winner )
