import numpy as np
from game import Agent
import abc
import game_state
import multi_agents

EXPLORATION_CONST = np.sqrt(2)
PROGRESSIVE_HISTORY_CONST = 10

WIN_VAL = 1
LOSE_VAL = 0

DEPTH = 17

class Node:
    def __init__(self, state, player, successors, valid_actions, fully_expanded=False):
        self.state = state
        self.edges = successors
        self.player = player
        self.fully_expanded = fully_expanded
        self.valid_actions = valid_actions
        self.wins = 0
        self.time_visited = 0

        self.score = 0

    def leaf_node(self):
        return len(self.valid_actions) == 0 and len(self.edges) == 0


class Edge:
    def __init__(self, action, old_node, new_node, player, score=0):

        self.score = score

        self.from_node = old_node
        self.to_node = new_node

        self.from_pos = action[0]
        self.single_pos_from = self.from_pos[0]*game_state.TABLE_SIZE + self.from_pos[1]
        self.to_pos = action[1]
        self.single_pos_to = self.to_pos[0]*game_state.TABLE_SIZE + self.to_pos[1]
        self.player = player


class TreeSearchAgent(Agent):

    def __init__(self, model, iteration_num=0, evaluation_function='dist_to_camp_difference', maximizing_player=0,
                 history_table_path=None, time_limit=10000):
        super().__init__()
        self.maximizing_player = maximizing_player
        self.model = model
        self.iteration_num = iteration_num
        self.root = None
        self.time_limit = time_limit

        if history_table_path is not None:
            self.moves_history_table = np.load(history_table_path)
        else:
            df_size = game_state.TABLE_SIZE * game_state.TABLE_SIZE
            self.moves_history_table = np.zeros((df_size, df_size))

    @abc.abstractmethod
    def get_action(self, game_state, w_1=1, w_2=1):
        return


class MCTSAgent(TreeSearchAgent):

    def better_uct(self, edge):
        if edge.to_node.time_visited == 0:
            return np.inf

        val_1 = edge.to_node.score / edge.to_node.time_visited + \
                EXPLORATION_CONST * np.sqrt(np.log(edge.from_node.time_visited) / edge.to_node.time_visited)

        val_2 = edge.score / self.moves_history_table[edge.single_pos_from, edge.single_pos_to]

        if edge.to_node.time_visited - edge.to_node.score + 1 == 0:

            val_3 = PROGRESSIVE_HISTORY_CONST / (edge.to_node.time_visited - edge.to_node.score)
        else:
            val_3 = PROGRESSIVE_HISTORY_CONST / (edge.to_node.time_visited - edge.to_node.score + 1)

        return val_1 + val_2 * val_3

    def get_action(self, state, w_1=1, w_2=1):
        valid_actions = state.get_legal_actions(state.get_current_player())

        forward_actions = [action for action in valid_actions if not
        multi_agents.is_going_backwards(game_state.TABLE_SIZE, action, state.get_current_player())]

        if len(forward_actions) != 0:
            valid_actions = forward_actions

        self.root = Node(state, state.get_current_player(), [], valid_actions)

        for i in range(self.iteration_num):

            path_to_leaf = self.selection(self.root, [])

            if len(path_to_leaf) == 0:  # Happens when there is only 1 node in the tree.
                path_to_leaf = [self.root]
                result = self.expand(path_to_leaf)
            else:
                result = self.expand(path_to_leaf)

            self.back_up(path_to_leaf, result)

        chosen_edge_index = np.argmax([edge.to_node.time_visited for edge in self.root.edges])
        chosen_edge = self.root.edges[chosen_edge_index]

        return [chosen_edge.from_pos, chosen_edge.to_pos]

    def selection(self, curr_node, path):
        """
        Recurse through the tree by looking for successor with maximal UCT (or similar calculation)
        pointed by Edges array until a leaf is reached.
        :param curr_node: Sub root of current iteration.
        :param path: Path from the root to the leaf.
        :return: New node for expansion.
        """

        if curr_node.leaf_node() or not curr_node.fully_expanded:
            return path

        maximal_UCT = np.NINF

        best_nodes = []
        for edge in curr_node.edges:
            val = self.better_uct(edge)
            if val == maximal_UCT:
                best_nodes.append(edge.to_node)

            elif val > maximal_UCT:
                best_nodes.clear()
                best_nodes.append(edge)
                maximal_UCT = val

        best_node_index = np.random.choice(range(len(best_nodes)))
        best_node = curr_node.edges[best_node_index].to_node
        return self.selection(best_node, path + [best_node])

    def play_game_from_state(self, state):
        moves_counter = 0
        play_outs_depth = 0

        curr_state = game_state.copy.deepcopy(state)
        curr_action = multi_agents.NO_ACTION
        while not curr_state.done and play_outs_depth < DEPTH:

            curr_action = self.model.get_action(curr_state)
            curr_state.apply_action(curr_action)
            curr_state.switch_player()
            moves_counter += 1
            play_outs_depth += 1

        if self.root.player == state.get_current_player():
            multi = -1
        else:
            multi = 1
        score = multi_agents.MCTS_play_outs_eval(curr_state, self.root.player)

        if curr_action != multi_agents.NO_ACTION:
            self.add_to_history(curr_action)

        return score * multi

    def add_to_history(self, action):
        from_action = action[0][0] * game_state.TABLE_SIZE + action[0][1]
        to_action = action[1][0] * game_state.TABLE_SIZE + action[1][1]
        if self.moves_history_table[from_action, to_action] == 0:
            self.moves_history_table[from_action, to_action] = 1
        else:
            self.moves_history_table[from_action, to_action] += 1

    def add_node_to_tree(self, parent, action):

        new_state = parent.state.generate_successor(action)
        curr_player = new_state.get_current_player()

        valid_actions = new_state.get_legal_actions(curr_player)

        forward_actions = [action for action in valid_actions if
                           not multi_agents.is_going_backwards(game_state.TABLE_SIZE, action, curr_player)]

        if len(forward_actions) != 0:
            valid_actions = forward_actions

        new_node = Node(new_state, curr_player, [], valid_actions)
        if len(valid_actions) == 0:
            new_node.state.done = True

        score = multi_agents.better_reflex(parent.state, action)
        new_edge = Edge(action, parent, new_node, curr_player, score=score)

        parent.edges.append(new_edge)

        return new_node

    def choose_action(self, last_node):
        """
        Chooses a random action from the current state.
        :param last_node: Leaf node from which we choose to expand.
        :param valid_actions: Valid actions from node's state.
        :return: action.
        """
        index = np.random.choice(range(len(last_node.valid_actions)))
        rnd_action = last_node.valid_actions[index]

        return rnd_action

    def compare_actions(self, action_1, action_2):
        return action_1[0] == action_2[0] and action_1[1][0] == action_2[1][0] and action_1[1][1] == action_2[1][1]

    def get_successor_node(self, last_node, rnd_action):
        """
        Generates a successor from the given action.
        :param last_node: Leaf node from which we expand.
        :param rnd_action: Chosen action.
        :return: Successor state to the leaf's state.
        """
        node = self.add_node_to_tree(last_node, rnd_action)
        last_node.valid_actions = [move for move in last_node.valid_actions if
                                   not self.compare_actions(move, rnd_action)]

        if len(last_node.valid_actions) == 0:
            last_node.fully_expanded = True

        return node

    def expand(self, path):
        """
        This function receives a leaf node of the current tree and has two options:
        If there is some action that yet has been explored, or in another words, if the tree doesn't include some
        successor of the current's state, then it chooses one randomly.
        If all state's successor already been expanded, then we select the best one.
        :param last_node:
        :return:
        """
        last_node = path[-1]
        if last_node.state.done or len(last_node.valid_actions) == 0:
            # Send to back propagating func with path.
            return LOSE_VAL

        else:
            rnd_action = self.choose_action(last_node)

            self.add_to_history(rnd_action)

            node = self.get_successor_node(last_node, rnd_action)

            if node.state.done or len(node.valid_actions) == 0:
                return WIN_VAL
            path.append(node)

            result = self.play_game_from_state(node.state)

            return result

    def back_up(self, path, result_value):
        """
        Updated the values of edges along the path of the last leaf from where the game has been simulated.
        :param path: Path in the actions-state graph from the perspective of root player.
        :param result_value: Value is given by the model which MCST uses to evaluate leaves in the graph:
        -1 <= value < =1
        :return: No return value, just update.
        """

        for node in path:
            if node.player == self.root.player:
                node.score += result_value
            else:
                node.score -= result_value

            node.time_visited += 1

if __name__ == '__main__':
    model = multi_agents.MCTSHelperAgent()
    path = 'MCTS_DB.npy'
    initial_state = game_state.GameState(init=False)

    agent = MCTSAgent(model, iteration_num=400, history_table_path=path)
    # action = agent.get_action()