# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An agent for starcraft build marines minigame."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math
import datetime
import csv
import os.path

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
from statistics import mean



DATA_FILE = 'beacon_agent_data'
# Functions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_MOVE_MARINE = actions.FUNCTIONS.Move_screen.id
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_SELECT_IDLE = actions.FUNCTIONS.select_idle_worker.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_PLAYER_SELF = 1

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_MARINE = 48

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

# Shortcuts
ACTION_DO_NOTHING = 'donothing'
ACTION_UL = 'moveul'
ACTION_U = 'moveu'
ACTION_UR = 'moveur'
ACTION_L = 'movel'
ACTION_R = 'mover'
ACTION_DL = 'movedl'
ACTION_D = 'moved'
ACTION_DR = 'movedr'

'''ACTION_SELECT_COMMANDCENTER = 'selectcommand'
ACTION_BUILD_SCV = 'buildscv'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
#ACTION_SELECT_ARMY = 'selectarmy'
#ACTION_ATTACK = 'attack' '''

SELECTED_IDLE = 'selectidle'

# List of possible actions to select at any time
smart_actions = [
    ACTION_UL,
    ACTION_U,
    ACTION_UR,
    ACTION_L,
    ACTION_R,
    ACTION_DL,
    ACTION_D,
    ACTION_DR
]

# Magic numbers

SUPPLY_DEPOT_MIN_X = 40
BARRACKS_MIN_X = 24
BARRACKS_MAX_Y = 58
SUPPLY_DEPOT_Y = 5
SUPPLY_DEPOT_SIZE = 7
BARRACKS_SIZE = 11



class BeaconAgent(base_agent.BaseAgent):
    """An agent to play the train marine minigame."""
    def __init__(self):
        super(BeaconAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.choice = ''

        # Tracks previous state
        self.previous_action = None
        self.previous_state = [0, 0]
        self.previous_beacon = None
        self.beacon_count = 0
        self.cumulative_reward = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    '''Hard coded values to reset to after each game
    ends so everything is reset'''
    def self_reset(self):
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        self.choice = ''

        # Tracks previous state
        self.previous_action = None
        self.previous_state = [0, 0]
        self.previous_beacon = None
        self.beacon_count = 0
        self.cumulative_reward = 0


    # Performs distance to score a state
    def score_state(self, state):
        return math.sqrt(state[0] * state[0] + state[1] * state[1] )

    def step(self, obs):
        super(BeaconAgent, self).step(obs)

        player_relative = obs.observation["feature_screen"].player_relative
        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]

        if _MOVE_MARINE in obs.observation["available_actions"]:

            beacon_y, beacon_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not beacon_y.any():
                return actions.FunctionCall(_NO_OP, [])
            marine_y, marine_x = (unit_type == _TERRAN_MARINE).nonzero()
            if not marine_y.any():
                return actions.FunctionCall(_NO_OP, [])

            beacon_mean_x, beacon_mean_y = mean(beacon_x), mean(beacon_y)
            marine_mean_x, marine_mean_y = mean(marine_x), mean(marine_y)

            current_state_x = 0
            if beacon_mean_x - marine_mean_x > 0 :
                current_state_x = 1
            else if beacon_mean_x - marine_mean_x < 0:
                current_state_x = -1

            current_state_y = 0
            if beacon_mean_y - marine_mean_y > 0 :
                current_state_y = 1
            else if beacon_mean_y - marine_mean_y < 0:
                current_state_y = -1

            current_state = [current_state_x, current_state_y]

            if self.previous_action is not None:
                reward = 0
                current_score = self.score_state(current_state)
                previous_score = self.score_state(self.previous_state)

                if self.previous_beacon is not None:
                    if self.score_state([self.previous_beacon[0] - beacon_mean_x,
                                        self.previous_beacon[1] - beacon_mean_y]) > 1:
                        reward += 100
                        self.beacon_count += 1

                self.previous_beacon = [beacon_mean_x, beacon_mean_y]
                reward +=  previous_score - current_score
                # No gamma here, just summing them
                self.cumulative_reward += reward
                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))


            rl_action = self.qlearn.choose_action(str(current_state))

            self.choice = smart_actions[rl_action]
            self.previous_state = current_state
            self.previous_action = rl_action

            target = [beacon_mean_x, beacon_mean_y]

            if self.choice == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP,[])

            elif self.choice == ACTION_UL:
                target = [marine_mean_x - 2, marine_mean_y - 2]

            elif self.choice == ACTION_U:
                target = [marine_mean_x, marine_mean_y - 2]

            elif self.choice == ACTION_UR:
                target = [marine_mean_x + 2, marine_mean_y - 2]

            elif self.choice == ACTION_L:
                target = [marine_mean_x - 2, marine_mean_y]

            elif self.choice == ACTION_R:
                target = [marine_mean_x + 2, marine_mean_y]

            elif self.choice == ACTION_DL:
                target = [marine_mean_x - 2, marine_mean_y + 2]

            elif self.choice == ACTION_D:
                target = [marine_mean_x, marine_mean_y + 2]

            elif self.choice == ACTION_DR:
                target = [marine_mean_x + 2, marine_mean_y + 2]

            return actions.FunctionCall(_MOVE_MARINE, [_NOT_QUEUED, target])

        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])


    #Copy pasted code from guide, not sure exactly whats up
    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False



# Directly from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

''' python -m pysc2.bin.agent --map BuildMarines --agent move_agent.BeaconAgent  '''

def main(unused_argv):
    agent = BeaconAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="MoveToBeacon",# sets map
                    players=[sc2_env.Agent(sc2_env.Race.terran)],# sets player(s)
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=8,#how many gamesteps between actions(limits APM)
                    game_steps_per_episode=0,#able to limit the length of a game
                    visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()

                '''Prints the state at the end of the game, will be useful
                    for tracking how much the training helps'''
                data = [agent.beacon_count, agent.cumulative_reward, str(datetime.datetime.now())]
                with open('training_beacon.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(data)
                # print(agent.previous_state)
                agent.self_reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
