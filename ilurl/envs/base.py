'''
            Traffic Light Environments

    Extends the flow's green wave environment
'''
__author__ = "Guilherme Varela"
__date__ = "2019-12-10"
import pdb
import os
import json
import numpy as np

from flow.core import rewards
from flow.envs.ring.accel import AccelEnv

from ilurl.core.ql.reward import RewardCalculator
from ilurl.utils.serialize import Serializer
from ilurl.utils.properties import delegate_property, lazy_property

import ilurl.loaders.parsers as parsers

from ilurl.core.agents_wrapper import AgentsWrapper

ILURL_HOME = os.environ['ILURL_HOME']

NETWORKS_PATH = \
    f'{ILURL_HOME}/data/networks/'


class TrafficLightEnv(AccelEnv, Serializer):
    """
    Environment used to train traffic light systems.

    This is a single TFLQLAgent controlling a variable number of
    traffic lights (TFL) with discrete features defined as such:

    1. One TFLQLAgent controlling k = 1, 2, ..., K TFL

    2. The actions for the agent is for each of the
       K-TFL is to:
        2.1 0 - short green (direction 0), Ex:
            (10s Green, 5s Yellow, 25s Red)
        2.2 1 - long green (direction 0), Ex:
            (25s Green, 5s Yellow, 10s Red)
        2.3 short green (direction 0) implies long
            green (direction 1) and vice-versa.

    3. Each k-TFL can only observe it's subjacent edges -
        meaning the state is described by the cars locally
        available on the neighborhood of K-TFL.
        

    4. The state Sk for each of the K-TFL can be represented by
       a tuple such that Sk = (vk, nk) where:
        4.1 vk is the mean speed over all adjacent edges.
        4.2 nk is the total number of all adjacent edges.

    5. The state S is a set describes all the possible configurations
        such that S = (S1, S2 ..., SK) for ease of implementation
        the S representation is flattened such that:

            S = (v1, n1, v2, n2  ..vK, nK)

    PARAMETERS:
    -----------

    * switch_time: minimum time a light must be constant before it
                    switches (in seconds). Earlier RL commands are
                    ignored.
    * tls_type: whether the traffic lights should be actuated by sumo or
                RL, options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be
                discrete or continuous.

    """
    def __init__(self,
                 env_params,
                 sim_params,
                 network,
                 simulator='traci'):

        super(TrafficLightEnv, self).__init__(env_params,
                                              sim_params,
                                              network,
                                              simulator=simulator)

        # Load MDP parameters from file (train.config[mdg_args]).
        mdp_params = parsers.parse_mdp_params()

        # TODO: Allow for mixed networks with actuated,
        # controlled and static traffic light configurations.
        self.tls_type = env_params.additional_params.get('tls_type')

        # Whether TLS timings are static or controlled by agent.
        self.static = (self.tls_type == 'static')

        # Cycle time.
        self.cycle_time = network.cycle_time

        # TLS programs.
        self.programs = network.programs

        # Keeps the internal value of sim step.
        self.sim_step = sim_params.sim_step

        # Problem formulation params.
        self.mdp_params = mdp_params

        # Object that handles RL agents logic.
        mdp_params.phases_per_traffic_light = network.phases_per_tls
        mdp_params.num_actions = network.num_signal_plans_per_tls
        self.agents = AgentsWrapper(mdp_params)

        # Reward function.
        self.reward_calculator = RewardCalculator(self.mdp_params)

        self.actions_log = {}
        self.states_log = {}

        # if self.tls_type != "actuated":
        self._reset()

    @property
    def stop(self):
        pass

    @stop.setter
    def stop(self, stop):
        self.agents.stop = stop

    # TODO: generalize delegation
    @delegate_property
    def tls_ids(self):
        pass

    @delegate_property
    def tls_max_capacity(self):
        pass

    @delegate_property
    def tls_phases(self):
        pass

    @delegate_property
    def tls_states(self):
        pass

    @lazy_property
    def tls_durations(self):
        return {
            tid: np.cumsum(durations).tolist()
            for tid, durations in self.network.tls_durations.items()
        }

    def update_observation_space(self):
        """
        Updates the observation space.

        Assumes that each traffic light carries a speed sensor.
        (counts and speeds)

        Updates the following data structures:

        * incoming: nested dict
            1st order (outer) keys: int
                    traffic_light_id
            2nd order keys: int
                    TLS phase
            3rd order (inner) keys: float
                    frame_id of observations ranging from 0 to duration 
            values: list
                    vehicle ids and speeds for the given TLS, phase and edges

            """
        def observe(components):
            veh_ids = []
            for component in components:
                edge_id, lanes = component
                veh_ids += \
                    [veh_id
                     for veh_id in self.k.vehicle.get_ids_by_edge(edge_id) 
                     if self.k.vehicle.get_lane(veh_id) in lanes]

            speeds = [
               self.k.vehicle.get_speed(veh_id)
                for veh_id in veh_ids
            ]
            return veh_ids, speeds

        for node_id in self.tls_ids:
            for phase, data in self.tls_phases[node_id].items():
                self.incoming[node_id][phase][self.duration] = \
                                    observe(data['components'])

    def get_observation_space(self):
        """
        Consolidates the observation space.
        Aggregates all data belonging to a complete cycle.

        Update:
        ------
        observation space is now a 3 level hierarchial dict:

            *   intersection: dict
                the top most represents the traffic lights
                (traffic_light_id)

            *   phases: dict
                the second layer represents the phases components
                for each intersection/traffic light

            *   values: list
                the third and final layer represents the variables
                being observed by the agent

        WARNING:
            when all cars are dispatched the
            state will be encoded with speed zero --
            change for when there aren't any cars
            the state is equivalent to maximum speed
        """

        def delay(t):
            return round(
                t - self.sim_step
                if t >= self.sim_step else
                self.cycle_time - self.sim_step
                if self.step_counter > 1 else 0.0, 2)
        prev = delay(self.duration)

        if (prev not in self.memo_observation_space) or self.step_counter <= 2:
            observations = {}
            normalize = self.mdp_params.normalize_state_space
            for nid in self.tls_ids:
                tls_data = []
                for phase in self.tls_phases[nid]:
                    max_speed, max_count = self.tls_max_capacity[nid][phase]
                    incoming = self.incoming[nid][phase]
                    values = []
                    for label in self.mdp_params.states_labels:

                        if label in ('count',):
                            counts = self.memo_counts[nid][phase]
                            count = 0
                            count += len(incoming[prev][1]) if prev in incoming else 0.0
                            counts[prev] = count
                            value = np.mean(list(counts.values()))

                        elif label in ('speed',):
                            counts = self.memo_counts[nid][phase].copy()
                            count = 0
                            count += len(incoming[prev][1]) if prev in incoming else 0.0

                            mem = self.memo_speeds[nid][phase]
                            speeds = []
                            speeds += incoming[prev][1] \
                                     if prev in incoming else []
                            mem[prev] = \
                                0.0 if not any(speeds) else round(np.mean(speeds), 2)
                            value = np.mean(list(mem.values()))

                            if normalize:
                                value = value / max_speed
                        else:
                            raise ValueError(f'`{label}` not implemented')

                        values.append(round(value, 2))
                    tls_data.append(values)
                observations[nid] = tls_data

            self.memo_observation_space[prev] = observations
        return self.memo_observation_space[prev]

    def get_state(self):
        """
        Return the state of the simulation as perceived by the RL agent.
        
        Returns
        -------
        state : array_like
            information on the state of the vehicles, which is provided to the
            agent
        """
        obs = self.get_observation_space()

        # Categorize.
        if self.mdp_params.discretize_state_space:
            obs = self.mdp_params.categorize_space(obs)

        # Flatten.
        flattened = \
            self.mdp_params.flatten_space(obs)

        return flattened

    def rl_actions(self, state):
        """
        Return the selected action given the state of the environment.

        Params:
        ------
            state : dict
            information on the state of the vehicles, which is provided to the
            agent
        
        Returns
        -------
            action : array_like
                information on the state of the vehicles, which is
                provided to the agent

        """
        if self.duration == 0:
            action = self.agents.act(state)
        else:
            action = None

        return action

    def cl_actions(self, static=False):
        """Executes the control action according to a program
            
        Params:
        ------
            * static: boolean
                If true execute the default program or change states at
                duration == tls_durations for each tls.
                Otherwise; (i) fetch the rl_action, (ii) fetch program,
                (iii) execute control action for program
        Returns:
        -------
            * cl_actions: tuple<bool>
                False;  duration<state_k> < duration < duration<state_k+1>
                True;  duration == duration<state_k+1>
        """
        ret = []
        dur = int(self.duration)

        def fn(tid):
            if (dur == 0 and self.step_counter > 1):
                return True

            if static:
                return dur in self.tls_durations[tid]
            else:
                progid = self._current_rl_action()[tid]
                return dur in self.programs[tid][progid]

        ret = [fn(tid) for tid in self.tls_ids]

        return tuple(ret)

    def apply_rl_actions(self, rl_actions):
        """
        Specify the actions to be performed by the RL agent(s).

        Parameters
        ----------
        rl_actions: list of actions or None
        """

        # Update observation space.
        self.update_observation_space()

        if self.tls_type != 'actuated':
            if self.duration == 0:
                # New cycle.

                # Get the number of the current cycle.
                cycle_number = \
                    int(self.step_counter / self.cycle_time)

                # Get current state.
                state = self.get_state()

                # Select new action.
                if rl_actions is None:
                    rl_action = self.rl_actions(state)
                else:
                    rl_action = rl_actions

                self.actions_log[cycle_number] = rl_action
                self.states_log[cycle_number] = state

                if self.step_counter > 1: # and not self.stop:
                    # RL-agent update.
                    reward = self.compute_reward(None)
                    prev_state = self.states_log[cycle_number - 1]
                    prev_action = self.actions_log[cycle_number - 1]
                    self.agents.update(prev_state, prev_action, reward, state)

            self.memo_rewards = {}
            self.memo_observation_space = {}

            # Update traffic lights' control signals.
            self._apply_cl_actions(self.cl_actions(static=self.static))
        else:
            if self.duration == 0:
                self.memo_rewards = {}
                self.memo_observation_space = {}

        # Update timer.
        self.duration = \
            round(self.duration + self.sim_step, 2) % self.cycle_time

    def _apply_cl_actions(self, cl_actions):
        """For each tls shift phase or keep phase

        Params:
        -------
            * cl_actions: list<bool>
                False; keep state
                True; switch to next state
        """
        for i, tid in enumerate(self.tls_ids):
            if cl_actions[i]:
                states = self.tls_states[tid]
                self.state_indicator[tid] = \
                        (self.state_indicator[tid] + 1) % len(states)
                next_state = states[self.state_indicator[tid]]
                self.k.traffic_light.set_state(
                    node_id=tid, state=next_state)
            
    def _current_rl_action(self):
        """Returns current rl action"""
        # adjust for duration
        N = (self.cycle_time / self.sim_step)
        actid = \
            int(max(0, self.step_counter - 1) / N)
        return self.actions_log[actid]

    def compute_reward(self, rl_actions, **kwargs):
        """
        Reward function for the RL agent(s).
        Defaults to 0 for non-implemented environments.
        
        Parameters
        ----------
        rl_actions : array_like
            actions performed by rl vehicles or None

        kwargs : dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward : float or list of float
        """
        if self.duration not in self.memo_rewards:
            reward = self.reward_calculator.calculate(
                self.get_observation_space()
            )
            self.memo_rewards[self.duration] = reward
        return self.memo_rewards[self.duration]

    def reset(self):
        super(TrafficLightEnv, self).reset()
        self._reset()

    def _reset(self):

        # duration measures the amount of time the current
        # configuration has been going on
        self.duration = 0.0

        self.incoming = {}

        self.memo_speeds = {}
        self.memo_counts = {}
        self.memo_flows = {}
        self.memo_queue = {}

        # stores the state index
        # used for opt iterations that did not us this variable
        self.state_indicator = {}
        for node_id in self.tls_ids:
            num_phases = len(self.tls_phases[node_id])
            if self.tls_type != 'actuated':
                self.state_indicator[node_id] = 0
                s0 = self.tls_states[node_id][0]
                self.k.traffic_light.set_state(node_id=node_id, state=s0)

            self.incoming[node_id] = {p: {} for p in range(num_phases)}

            self.memo_speeds[node_id] = {p: {} for p in range(num_phases)}
            self.memo_counts[node_id] = {p: {} for p in range(num_phases)}

        self.memo_rewards = {}
        self.memo_observation_space = {}
