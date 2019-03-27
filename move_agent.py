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
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

# Functions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

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

_NOT_QUEUED = [0]
_QUEUED = [1]

# Shortcuts
ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_COMMANDCENTER = 'selectcommand'
ACTION_BUILD_SCV = 'buildscv'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
#ACTION_SELECT_ARMY = 'selectarmy'
#ACTION_ATTACK = 'attack'

SELECTED_IDLE = 'selectidle'

# List of possible actions to select at any time
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_COMMANDCENTER,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE
]



class MoveAgent(base_agent.BaseAgent):
    """An agent to play the train marine minigame."""
    def __init__(self):
        super(MoveAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.choice = ''
        # Used for scoring
        self.previous_army_supply = 0

        # Tracks previous state
        self.previous_action = None
        self.previous_state = None

    def step(self, obs):
        super(MoveAgent, self).step(obs)


        # Setting up all of the information for the current state

        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        # Possibly change this to actually count depots
        supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        # Same as depot
        barracks_count = 1 if barracks_y.any() else 0

        SCV_y, SCV_x = (unit_type == _TERRAN_SCV).nonzero()
        SCV_count = len(SCV_y)

        supply_limit = obs.observation['player'][4]

        # Consider adding in (queued actions?)
        # Something about barracks currently building
        # is supply limit necessary?
        current_state = [
            SCV_count,
            supply_depot_count,
            barracks_count,
            supply_limit
        ]


        # Update using the last action
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



        # HARD CODED WAY TO HANDLE EVERY ACTION IN THE ACTION SPACE
        if self.choice == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP,[])

        elif self.choice == ACTION_SELECT_COMMANDCENTER:
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
            if unit_x.any():
                CC = random.randint(0, len(unit_x) - 1) # selects a random CC if needed
                target = [unit_x[CC], unit_y[CC]]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif self.choice == ACTION_BUILD_SCV:
            if _TRAIN_SCV in obs.observation["available_actions"]:
                # Not sure if a target is needed for build scv
                return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])

        elif self.choice == ACTION_SELECT_SCV:
            # Optimizes SCV usage manually, to select idle workers first
            if _SELECT_IDLE in obs.observation["available_actions"]:
                return actions.FunctionCall(_SELECT_IDLE, [_NOT_QUEUED])

            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
            if unit_x.any():
                scv = random.randint(0, len(unit_x) - 1)
                target = [unit_x[scv], unit_y[scv]]

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif self.choice == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation["available_actions"]:
                target = [random.randint(0, 83), random.randint(0, 83)]
                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])


        elif self.choice == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                target = [random.randint(0, 83), random.randint(0, 83)]
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        elif self.choice == ACTION_SELECT_BARRACKS: # Need to update to select(all) barracks?
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
            if unit_y.any():
                barracks = random.randint(0, len(unit_x) - 1)
                target = [unit_x[barracks], unit_y[barracks]]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif self.choice == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation["available_actions"]:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])


        return actions.FunctionCall(_NO_OP, [])

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


def main(unused_argv):
    agent = MoveAgent()
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

                ''' Prints the state at the end of the game, will be useful
                for tracking how much the training helps'''
                print(agent.previous_state)

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
