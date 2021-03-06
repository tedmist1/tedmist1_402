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
import datetime
import csv
import os.path

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app



DATA_FILE = 'agent_data'
# Functions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_IDLE = actions.FUNCTIONS.select_idle_worker.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

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
ACTION_SELECT_COMMANDCENTER = 'selectcommand'
ACTION_BUILD_SCV = 'buildscv'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
SELECTED_IDLE = 'selectidle'

# List of possible actions to select at any time
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE
]

# Magic numbers

SUPPLY_DEPOT_MIN_X = 40
BARRACKS_MIN_X = 24
BARRACKS_MAX_Y = 58
SUPPLY_DEPOT_Y = 5
SUPPLY_DEPOT_SIZE = 7
BARRACKS_SIZE = 11

'''Previously known as the third iteration of MoveAgent'''

class BuildMarinesAgent(base_agent.BaseAgent):
    """An agent to play the train marine minigame."""
    def __init__(self):
        super(BuildMarinesAgent, self).__init__()

        # e_decay does not work outside of a single sitting, since it is only stored locally
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))),
            learning_rate=0.05, e_decay = 0.9995)
        self.choice = ''
        # Used for scoring
        self.previous_army_supply = 0

        # Tracks previous state
        self.previous_action = None
        self.previous_state = [0, 0, 0, 15]
        self.supply_depot_count = 0
        self.barracks_count = 0
        # Used because you can only act every other step, those other steps are
        # predetermined by what you did in the previous step
        self.step_num = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    '''Hard coded values to reset to after each game
    ends so everything is reset'''
    def self_reset(self):
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        self.choice = ''
        # Used for scoring
        self.previous_army_supply = 0

        # Tracks previous state
        self.previous_action = None
        self.previous_state = [0, 0, 0, 15]
        self.supply_depot_count = 0
        self.barracks_count = 0
        self.step_num = 0


    def step(self, obs):
        super(BuildMarinesAgent, self).step(obs)
        self.step_num += 1
        if obs.first():
            self.self_reset()

        # Setting up all of the information for the current state

        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        SCV_y, SCV_x = (unit_type == _TERRAN_SCV).nonzero()
        SCV_count = obs.observation['player'][6]
        supply_limit = obs.observation['player'][4]
        supply_count = obs.observation['player'][3]

        #Add steps remaining? Def would make it too complicated...
        current_state = [
            supply_count,
            supply_limit,
            self.barracks_count
        ]

        # First step of each action. Learning is done here, and then first
        # of the two steps occurs
        if self.step_num % 2 == 0:

            # Deals with learning, rewards, and q learning
            if self.previous_action is not None:
                reward = 0
                army_supply = obs.observation['player'][5]
                if army_supply > self.previous_army_supply:
                    reward = army_supply - self.previous_army_supply
                    self.previous_army_supply = army_supply

                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))


            # Setting the previous action and state to the current
            rl_action = self.qlearn.choose_action(str(current_state))

            self.choice = smart_actions[rl_action]
            self.previous_state = current_state
            self.previous_action = rl_action

            ''' Updates barrack and supply depot count, and tell the agent
                to select an scv as its first action'''
            if _BUILD_BARRACKS in obs.observation['last_actions']:
                self.barracks_count += 1
                self.choice = ACTION_SELECT_SCV

            if _BUILD_SUPPLY_DEPOT in obs.observation['last_actions']:
                self.supply_depot_count += 1
                self.choice  = ACTION_SELECT_SCV


            if self.choice == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP,[])

            # First action is to select command center
            elif self.choice == ACTION_BUILD_SCV:
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                if unit_x.any():
                    CC = random.randint(0, len(unit_x) - 1) # selects a random CC if needed
                    target = [unit_x[CC], unit_y[CC]]
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # Both build supplydepot and buildbarracks start with selecting an scv
            elif self.choice == ACTION_BUILD_SUPPLY_DEPOT or self.choice == ACTION_BUILD_BARRACKS:
                # Optimizes SCV usage manually, to select idle workers first
                # Can also backfire and cause stuck, idle scvs to be called though
                if _SELECT_IDLE in obs.observation["available_actions"]:
                    return actions.FunctionCall(_SELECT_IDLE, [_NOT_QUEUED])

                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                if unit_x.any():
                    scv = random.randint(0, len(unit_x) - 1)
                    target = [unit_x[scv], unit_y[scv]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # Build marine starts with selecting all barrack
            elif self.choice == ACTION_BUILD_MARINE:
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                if unit_y.any():
                    barracks = random.randint(0, len(unit_x) - 1)
                    target = [unit_x[barracks], unit_y[barracks]]
                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])



        else:
            if self.choice == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP,[])


            elif self.choice == ACTION_BUILD_SCV:
                if _TRAIN_SCV in obs.observation["available_actions"]:
                    return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])

            elif self.choice == ACTION_BUILD_SUPPLY_DEPOT:
                if _BUILD_SUPPLY_DEPOT in obs.observation["available_actions"]:
                    target = [(self.supply_depot_count * SUPPLY_DEPOT_SIZE) % (80 - SUPPLY_DEPOT_MIN_X) + SUPPLY_DEPOT_MIN_X,
                              ((self.supply_depot_count * SUPPLY_DEPOT_SIZE) // (80 - SUPPLY_DEPOT_MIN_X) ) * 7 + SUPPLY_DEPOT_Y]

                    # Hard coded limit of supply depots, otherwise it eventually tries
                    # to build outside the map, crashing the program
                    if self.supply_depot_count >= 30:
                        return actions.FunctionCall(_NO_OP, [])

                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])


            elif self.choice == ACTION_BUILD_BARRACKS:
                if _BUILD_BARRACKS in obs.observation["available_actions"]:

                    target = [(self.barracks_count * BARRACKS_SIZE) % (80 - BARRACKS_MIN_X) + BARRACKS_MIN_X, BARRACKS_MAX_Y - ((self.barracks_count * BARRACKS_SIZE) // (80 - BARRACKS_MIN_X) ) * BARRACKS_SIZE ]

                    # Hard coded limit of barracks, same reason as depots
                    if self.barracks_count >= 11:
                        return actions.FunctionCall(_NO_OP, [])

                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

            elif self.choice == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation["available_actions"]:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])


        return actions.FunctionCall(_NO_OP, [])



# Directly from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow with minimum modifications
class QLearningTable:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9, e_decay = 1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay = e_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        self.epsilon *= self.epsilon_decay
        # At random chance, explore instead of exploit
        if np.random.uniform() < self.epsilon:
            # Get resulting Q values from State information
            state_action = self.q_table.ix[observation, :]
            # Gets max of the actions in the given state
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # Current Q value for state action pair
        q_predict = self.q_table.ix[s, a]
        # Reward for performing action in the state, plus predicted reward for next state
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # Update Q value with the difference times the learning rate
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

# Method to run without main method, but fails after a few iterations
''' python -m pysc2.bin.agent --map BuildMarines --agent move_agent.BuildMarinesAgent  '''

def main(unused_argv):
    agent = BuildMarinesAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="BuildMarines",# sets map
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
                data = [agent.previous_army_supply, *(agent.previous_state), str(datetime.datetime.now())]
                with open('training.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(data)
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
