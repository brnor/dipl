import os
import numpy as np

# Environment structure example taken from: https://github.com/mpSchrader/gym-sokoban

class IceGame():
    def __init__(self,
                filename,
                max_steps=100,
                multiple=False):
        self.levels = self.loadLevels(filename)
        # current_level_num holds the index of the level
        self.current_level_num = 0
        self.multiple = multiple # play multiple levels one after another?

        self.max_steps = max_steps

        #Rewards
        self.reward_finished = 20
        self.penalty_step = -0.1
        self.reward_prev = 0

        self.reset()

    def reset(self, resetRew=True):
        if len(self.levels) < 0:
            raise ValueError("No levels found.")
        # current_level holds the original level throughout the play
        self.current_level = self.levels[self.current_level_num].copy()
        # level_state holds the current level state as it changes through playing
        self.level_state = self.levels[self.current_level_num].copy()

        self.player_pos = np.argwhere(self.level_state == 2)[0] # returns array but there should only be 1 result
        self.env_steps = 0
        # Used to keep the reward when changing levels
        if resetRew:
            self.reward_prev = 0
        observation = self.level_state.copy()
        
        return observation

    def step(self, action):
        assert action in ACTIONS
        self.env_steps += 1

        moved = self._move_player(action)
        self._calculate_reward()
        done = self._check_done()
        win = self._check_goal()
        if win and self.multiple:
            if self.current_level_num < len(self.levels) - 1:
                done = False
                self.current_level_num += 1 
                observation = self.reset(resetRew=False)
        observation = self.level_state.copy()

        #debug info
        info = {
            "action.name": ACTIONS[action],
            "action.moved": moved
        }
        if done:
            info["maxsteps_reached"] = self._check_maxsteps()
            info["goal_reached"] = self._check_goal()
        
        return observation, self.reward_prev, done, info

    def _move_player(self, action):
        movement = MOVE_CHANGE[action]
        current_pos = self.player_pos.copy()
        new_pos = self.player_pos.copy()
        temp_pos = self.player_pos + movement
        bounds = self.level_state.shape

        while temp_pos[0] >= 0 and temp_pos[0] < bounds[0] \
                and temp_pos[1] >= 0 and temp_pos[1] < bounds[1] :
            # if the next position is not a wall
            if self.level_state[temp_pos[0], temp_pos[1]] != 1:
                new_pos = temp_pos.copy()

            # if the next position is a wall or if current position is finish
            if self.level_state[temp_pos[0], temp_pos[1]] == 1 \
                    or self.level_state[new_pos[0], new_pos[1]] == 3:
                break

            temp_pos += movement

        if new_pos[0] != current_pos[0] \
                or new_pos[1] != current_pos[1]:
            self.player_pos = new_pos
            # set new position to player
            self.level_state[(new_pos[0], new_pos[1])] = 2 
            # set old position to empty space
            self.level_state[current_pos[0], current_pos[1]] = 0
            return True
        
        return False

    # Reward is calculated:
    # Making a step: - 0.1
    # Reaching the goal: + 20
    def _calculate_reward(self):
        self.reward_prev = self.penalty_step

        win = self._check_goal()
        if win :
            self.reward_prev += self.reward_finished

    def _check_done(self):
        return self._check_maxsteps() or self._check_goal()

    def _check_maxsteps(self):
        return (self.max_steps == self.env_steps)

    def _check_goal(self):
        end_pos = np.argwhere(self.current_level == 3)[0]
        win = end_pos[0] == self.player_pos[0] \
                and end_pos[1] == self.player_pos[1]
        return win

    def loadLevels(self, filename):
        if os.path.exists(filename):
            levels = []
            with open(filename, "r") as f:
                level = []
                for line in f:
                    if line.strip() != "":
                        row = list(line.strip())
                        level.append(row)
                    else:
                        if len(level) > 0:
                            levels.append(level)
                        level = []
            if len(level) > 0:
                levels.append(level)
            # Replace all characters to int
            levels = np.array(levels)
            for level in levels:
                for key, val in CHAR_TO_INT.items():
                    level[level == key] = val

            levels  = levels.astype(int)
            return levels   

    def set_maxsteps(self, num):
        self.max_steps = num

    def get_actions(self):
        return ACTIONS

    def get_levels(self):
        return self.levels

    def get_current_level(self):
        return self.current_level

    def get_current_level_num(self):
        return self.current_level_num

    def set_level(self, levelNum):
        if levelNum >= 0 and levelNum < len(self.levels):
            self.current_level_num = levelNum
            self.reset()
        return self.current_level_num

# Character definitions:
# '_' : empty space that player can move on
# '#' : wall 
# 'P' : player
# 'X' : goal

CHAR_TO_INT = {
    '_': 0,
    '#': 1,
    'P': 2,
    'X': 3
}

INT_TO_CHAR = {
    0: '_',
    1: '#',
    2: 'P',
    3: 'X'
}

ACTIONS = {
    0: 'move up',
    1: 'move right',
    2: 'move down',
    3: 'move left'
}

MOVE_CHANGE = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}