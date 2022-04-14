
import gym
from gym.spaces import Box, Discrete

from .pacman.pacman import readCommand, ClassicGameRules
import numpy as np
import random as rd

class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, layout, seed, reward_goal, reward_crash, reward_food, reward_time, render, max_steps
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
        self.max_steps = max_steps

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


        # self.reset()

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
        done = self.game.gameOver or self._check_if_maxsteps()


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
        info = dict()
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["is_success"] = self.game.state.isWin()

        # return self.game.state, reward, self.game.gameOver, dict()
        return self.my_render(), reward, done, info

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

        self.game = self.rules.newGame(
            self.layout,
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

    def _check_if_maxsteps(self):
        return (self.max_steps == len(self.game.moveHistory))

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