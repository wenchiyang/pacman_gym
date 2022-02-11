import gym
from gym.spaces import Box, Discrete

from .pacman.pacman import readCommand, ClassicGameRules
import numpy as np
from .pacman.layout import Layout
import random as rd
import networkx as nx

def sample_layout(layout):
    # generate a text given width, height and num_ghost, num_food, wall_positions
    width = layout.width
    height = layout.height
    num_food = np.count_nonzero(np.array(layout.food.data) == True)
    num_agents = len(layout.agentPositions)
    walls = str(layout.walls).split("\n")
    non_wall_positions = []
    wall_positions = []
    for r in range(height):
        for c in range(width):
            if walls[r][c]=='F':
                non_wall_positions.append((r,c))
            else:
                wall_positions.append((r, c))
    layout_text = sample_l(width, height, num_agents, num_food, non_wall_positions, wall_positions)
    new_layout = Layout(layout_text.split('\n'))
    return new_layout

def sample_l(width, height, num_agents, num_food, non_wall_positions, wall_positions):
    layout = None
    pacman_position = None
    ghost_positions = None
    food_positions = None
    while not valid(layout, width, height, pacman_position, food_positions, ghost_positions, non_wall_positions, wall_positions):
        positions = rd.sample(non_wall_positions, num_agents + num_food)
        pacman_position = positions[0]
        ghost_positions = positions[1:num_agents]
        food_positions = positions[num_agents:]
        layout = generate_layout_text(width, height, pacman_position, ghost_positions, food_positions, wall_positions)

    return layout

def safe_add(s, p1, p2s, width, height):
    p1r, p1c = p1
    for p2 in p2s:
        p2r, p2c = p2
        if 0 <= p1r < height and 0 <= p2r < height and 0 <= p1c < width and 0 <= p2c < width:
            s.add((p1, p2))

def all_edges(width, height):
    edges = set()
    for r in range(height):
        for c in range(width):
            safe_add(edges, (r, c), [(r+1,c), (r-1,c), (r,c+1), (r,c-1)], width, height)
    return edges

def valid(layout, width, height, pacman_position, food_positions, ghost_positions, non_wall_positions, wall_positions):
    if layout is None:
        return False
    valid_postions = [l for l in non_wall_positions if l not in ghost_positions]
    g = nx.Graph()
    g.add_nodes_from(valid_postions)

    for n1, n2 in all_edges(width, height):
        if n1 in valid_postions and n2 in valid_postions:
            g.add_edge(n1, n2)

    for food_position in food_positions:
        if not nx.has_path(g, pacman_position, food_position):
            print("not valid")
            return False
    return True

def generate_layout_text(width, height, pacman_position, ghost_positions, food_positions, wall_positions):
    layout = []
    for r in range(height):
        row = ''
        for c in range(width):
            if (r,c) in wall_positions:
                row += '%'
            elif (r,c) == pacman_position:
                row += 'P'
            elif (r, c) in ghost_positions:
                row += 'G'
            elif (r, c) in food_positions:
                row += '.'
            else:
                row += ' '
        layout.append(row)
    return '\n'.join(layout)

# def sample_positions(layout):
#     all_valid_positions = []
#     for h in range(layout.height):
#         for w in range(layout.width):
#             if not layout.walls[w][h]:
#                 all_valid_positions.append((w, h))
#
#     pos_num = len(layout.agentPositions)
#     food_num = np.count_nonzero(np.array(layout.food.data) == True)
#     positions = rd.sample(all_valid_positions, pos_num + food_num)
#
#     return positions[:pos_num], positions[pos_num:]


class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, layout, seed, reward_goal, reward_crash, reward_food, reward_time, render
    ):
        """"""
        input_args = [
                "--layout",
                layout,
                "--reward-goal",
                str(reward_goal),
                "--reward-crash",
                str(reward_crash),
                "--reward-food",
                str(reward_food),
                "--reward-time",
                str(reward_time)
            ]
        self.render_or_not = render
        if not render:
            input_args.append("--quietTextGraphics")

        args = readCommand(input_args)

        # set OpenAI gym variables
        self._seed = seed
        self.A = ["Stop", "North", "South", "West", "East"]
        self.steps = 0
        self.history = []

        # port input values to fields
        self.layout = args["layout"]
        self.pacman = args["pacman"]
        self.ghosts = args["ghosts"]
        self.display = args["display"]
        self.numGames = args["numGames"]
        self.record = args["record"]
        self.numTraining = args["numTraining"]
        self.numGhostTraining = args["numGhostTraining"]
        self.withoutShield = args["withoutShield"]
        self.catchExceptions = args["catchExceptions"]
        self.timeout = args["timeout"]
        self.symX = args["symX"]
        self.symY = args["symY"]

        self.reward_goal = args["reward_goal"]
        self.reward_crash = args["reward_crash"]
        self.reward_food = args["reward_food"]
        self.reward_time = args["reward_time"]

        self.rules = ClassicGameRules(
            args["timeout"],
            self.reward_goal,
            self.reward_crash,
            self.reward_food,
            self.reward_time,
        )

        ######

        self.grid_size = 1
        self.grid_height = self.layout.height
        self.grid_weight = self.layout.width
        self.color_channels = 1

        import __main__

        __main__.__dict__["_display"] = self.display

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(
                self.grid_height * self.grid_size,
                self.grid_weight * self.grid_size
            )
        )
        self.action_space = Discrete(5) # default datatype is np.int64
        self.action_size = 5
        self.np_random = rd.seed(self._seed)
        self.reward_range = (0, 10)
        self.beQuiet = not render

        self.reset()

    def step(self, action, observation_mode="human"):
        """
        Parameters
        ----------
        action :
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        agentIndex = 0

        if isinstance(action, np.int64) or isinstance(action, int):
            action = self.A[action]

        action = "Stop" if action not in self.get_legal_actions(0) else action

        # perform "doAction" for the pacman
        self.game.agents[agentIndex].doAction(self.game.state, action)
        self.game.take_action(agentIndex, action)
        # self.render()


        reward = self.game.state.data.scoreChange

        # if self.game.gameOver:
        #     eps_info = {"last_r": reward}
        # else:
        #     eps_info = dict()
        # move the ghosts
        # if not self.game.gameOver:
        #     for agentIndex in range(1, len(self.game.agents)):
        #         state = self.game.get_observation(agentIndex)
        #         action = self.game.calculate_action(agentIndex, state)
        #         self.game.take_action(agentIndex, action)
        #         self.render()
        #         reward += self.game.state.data.scoreChange

        # return self.game.state, reward, self.game.gameOver, dict()
        return self.my_render(), reward, self.game.gameOver, dict()

    def reset(self, observation_mode="human"):
        # self.beQuiet = self.game_index < self.numTraining + self.numGhostTraining

        if self.beQuiet:
            # Suppress output and graphics
            from .pacman import textDisplay

            self.gameDisplay = textDisplay.NullGraphics()
            self.rules.quiet = True
        else:
            self.gameDisplay = self.display
            self.rules.quiet = False

        sampled_layout = sample_layout(self.layout)

        self.game = self.rules.newGame(
            sampled_layout,
            self.pacman,
            self.ghosts,
            self.gameDisplay,
            self.beQuiet,
            self.catchExceptions,
            self.symX,
            self.symY,
        )
        self.game.start_game()

        return self.my_render()

    def render(self, mode="human", close=False):
        self.game.render()

    def my_render(self):
        return self.game.my_render(grid_size=self.grid_size)

    def get_legal_actions(self, agentIndex):
        return self.game.state.getLegalActions(agentIndex)

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return self.A

    @staticmethod
    def constraint_func(self):
        return

ACTION_LOOKUP = {
    0: 'stay',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right',
}