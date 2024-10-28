# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.num_walls = 0
    
    def update_number_of_walls(self, board_size):
        if self.num_walls == 0:
            # I noticed that when the game starts, there are (SIZE_OF_BOARD - 2) walls. 
            # So I just set it to that value
            self.num_walls = (board_size - 2)
            # we might be off by 1 because we don't know who starts first
        else:
            # each time we put a wall, we need to add 1 to the number of walls
            self.num_walls += 1
        return self.num_walls

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        board_size = chess_board.shape[0]
        num_walls = self.update_number_of_walls(board_size)

        # first step, copy the chess board and create a state
        current_state = State(chess_board, my_pos, adv_pos, max_step, True, num_walls)

        root = StateTreeNode(current_state, None, 3) # for now, set max search depth to 3
        root.alpha_beta()
            
        if root.best_move is None:
            print("No move found", file=sys.stderr)
            return my_pos, 0
        new_pos, dir = root.best_move
        time_taken = time.time() - start_time
        if time_taken > 2:
            print(f"Time taken: {time_taken:.4f} seconds", file=sys.stderr)
        
        # we've place a wall, so we need to update the number of walls
        self.num_walls = self.update_number_of_walls(board_size)
        return new_pos, dir


class State():
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # up, right, down, left

    def __init__(self, chess_board, my_pos, adv_pos, max_step, my_turn, num_walls):
        self.chess_board = deepcopy(chess_board) # copy the state of the board
        self.board_size = chess_board.shape[0]
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.my_turn = my_turn
        self.num_walls = num_walls
        self.maximum_num_walls = 2 * (self.board_size) * (self.board_size - 1)

    # Get all possible moves with BFS
    def get_possible_moves(self):
        my_pos, adv_pos = self.my_pos, self.adv_pos
        # Swap if it's opponent's turn
        if not self.my_turn:
            my_pos, adv_pos = adv_pos, my_pos
        
        possible_moves = []
        state_queue = [(my_pos, 0)]
        visited = {tuple(my_pos)}

        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            row, col = cur_pos
            if cur_step > self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[row, col, dir]:
                    continue # wall exists, cannot move and position is not valid

                next_pos = (row + move[0], col + move[1])
                if np.array_equal(next_pos, adv_pos):
                    continue # cannot move to opponent's position

                # we will only accep this wall placement if in the current position of the agent, there is not 3 walls
                if np.sum(self.chess_board[row, col])+1 < 3:
                    possible_moves.append(((row, col), dir)) # add the move to the list of possible moves
                
                # We can move to the next position if it is not visited
                if tuple(next_pos) not in visited:
                    state_queue.append((next_pos, cur_step + 1))
                    visited.add(tuple(next_pos))

        return possible_moves

    # Helper functions ---------------------------------------------
    #  get_next_state : get the next state of the game given an action
    #  assume that the action is valid
    def get_child_state(self, action):
        # first step, copy the chess board and create a state
        (row, col), dir = action
        new_pos = (row, col) # new position of the agent either our agent or the opponent (for now)

        chess_board_copy = deepcopy(self.chess_board)
        if self.my_turn:
            # Create the child state with our agent's new position, setting it to be the opponent's turn
            child_state = State(chess_board_copy, new_pos, self.adv_pos, self.max_step, False, self.num_walls+1)
            # put wall in new chess board
            child_state.chess_board[row, col, dir] = True
            # put a wall in the opposite direction
            child_state.chess_board[row + self.moves[dir][0], col + self.moves[dir][1], (dir + 2) % 4] = True
            # return the new state
            return child_state
        else:
            # Create the child state with our opponent's new position, setting it to be our turn
            child_state = State(chess_board_copy, self.my_pos, new_pos, self.max_step, True, self.num_walls+1)
            # put wall in new chess board
            child_state.chess_board[row, col, dir] = True
            # return the new state
            return child_state


    # TODO A* search to know if there is a path to the oppenent
    #  check_endgame : check if the game ends and compute the current score of the agents
    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score

    #  TODO get_heuristic_score : get the heuristic score of the current state
    def get_heuristic_score(self):
        # Endgame Heuristic 1 : if number of walls is 60% of the total number of walls, we are in the endgame
        if self.num_walls >= 0.6 * self.maximum_num_walls:
            is_endgame, p0_score, p1_score = self.check_endgame()
            if is_endgame:
                if p0_score > p1_score:
                    return 100
                elif p0_score < p1_score:
                    return -100
                else:
                    return 0
            # TODO : there is a path between our agent and the opponent, so we need to find the shortest path
        # Endgame Heuristic 2 : early game, we want to be as far away from the opponent as possible
        #early_heuristic = 0
        #if self.num_walls <= 0.2 * self.maximum_num_walls:
            # maximize distance between the two agents
        #    distance_weight = 1
        #    distance = abs(self.my_pos[0] - self.adv_pos[0]) + abs(self.my_pos[1] - self.adv_pos[1])
        #    early_heuristic += distance_weight * (distance)

            # minimize distance between our agent and the center of the board
            # middle_line_weight = 1
            # early_heuristic += middle_line_weight * 1 / abs(self.my_pos[0] - self.board_size//2) + abs(self.my_pos[1] - self.board_size//2)

            # student agent is trapped into cell with 3 walls
            # maximize distance between the two agents
            # distance_weight = 5
            # early_heuristic += distance_weight * abs(self.my_pos[0] - self.adv_pos[0]) + abs(self.my_pos[1] - self.adv_pos[1])
            # minimize distance between our agent and the center of the board
            # middle_line_weight = 5
            # early_heuristic += middle_line_weight * abs(self.my_pos[0] - self.board_size//2) + abs(self.my_pos[1] - self.board_size//2)
            # Don't go to cells with two walls
            # corridor_weight = 5
            # early_heuristic -= corridor_weight * np.sum(self.chess_board[self.my_pos] == 2)
            # Don't go to cells with three walls
            # deadend_weight = 10
            # early_heuristic -= deadend_weight * np.sum(self.chess_board[self.my_pos] == 3)

        #    return early_heuristic
        
        # if nothing else, just return a random number
        score = np.random.randint(0, 100) * (-1)**(not self.my_turn)
        return score

class StateTreeNode():
    # TODO parent might not be needed, but we will see if we do backtracking...
    def __init__(self, state, parent=None, max_search_depth=1):
        self.state = state
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.max_search_depth = max_search_depth
        self.alpha = -np.inf
        self.beta = np.inf
        self.best_move = None
 
    def alpha_beta(self):
        if self.depth == self.max_search_depth - 1:
            # if there is 0.75 * size of board in walls, we are in near the end of game.
            return self.state.get_heuristic_score()

        possible_moves = self.state.get_possible_moves()

        if self.state.my_turn:  # Maximizing player
            value = -np.inf
            for action in possible_moves:
                new_state = self.state.get_child_state(action)
                child = StateTreeNode(new_state, self, self.max_search_depth)
                child_value = child.alpha_beta()
                if child_value > value:
                    value = child_value
                    self.best_move = action  # Update the best move
                self.alpha = max(self.alpha, value)
                if self.alpha >= self.beta:
                    break

        else:  # Minimizing player
            value = np.inf
            for action in possible_moves:
                new_state = self.state.get_child_state(action)
                child = StateTreeNode(new_state, self, self.max_search_depth)
                child_value = child.alpha_beta()
                if child_value < value:
                    value = child_value
                    self.best_move = action  # Update the best move
                self.beta = min(self.beta, value)
                if self.alpha >= self.beta:
                    break
        return value