"""
    Experiment class to run simulations.
    (See flow.core.experiment)
"""
import os
from pathlib import Path
import warnings
import datetime
import logging

from tqdm import tqdm

import numpy as np

from ilurl.utils.precision import action_to_double_precision

# TODO: Track those anoying warning
warnings.filterwarnings('ignore')

class Experiment:
    """
    Class to run an experiment in any supported simulator.

    This class acts as a runner for a scenario and environment:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> exp = Experiment(env, ...)  # for some env and scenario
        >>> exp.run(num_steps=1000)

    If you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> sim_params = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object.

    """
    def __init__(self,
                env,
                exp_path : str,
                train : bool = True,
                save_agent : bool = False,
                save_agent_interval : int = 100,
                tls_type : str = 'rl'):
        """

        Parameters:
        ----------
        env : flow.envs.Env
            the environment object the simulator will run.
        exp_path : str
            path to dump experiment-related objects.
        train : bool
            Whether to train RL agents.
        save_agent : bool
            Whether to save RL agents parameters throughout training.
        save_agent_interval : int
            RL agent save interval (in number of agent-update steps).
        tls_type : str
            Traffic ligh system type: ('rl', 'random', 'actuated',
            'static', 'webster' or 'max_pressure')

        """
        # Guarantees that the enviroment has stopped.
        if not train:
            env.stop = True

        self.env = env
        self.train = train
        self.exp_path = Path(exp_path) if exp_path is not None else None
        self.cycle = getattr(env, 'cycle_time', None)
        self.save_agent = save_agent
        self.save_agent_interval = save_agent_interval
        self.tls_type = tls_type

        logging.info(" Starting experiment {} at {}".format(
            env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(
            self,
            num_steps : int,
            rl_actions = None,
            stop_on_teleports : bool = False
    ):
        """
        Run the given scenario (env) for a given number of steps.

        Parameters:
        ----------
        num_steps : int
            number of steps to run.
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any).
        stop_on_teleport : boolean
            if true will break execution on teleport which occur on:
            * OSM scenarios with faulty connections.
            * collisions.
            * timeouts a vehicle is unable to move for 
            -time-to-teleport seconds (default 300) which is caused by
            wrong lane, yield or jam.

        Returns:
        -------
        info_dict : dict
            contains experiment-related data.

        References:
        ---------
        https://sourceforge.net/p/sumo/mailman/message/33244698/
        http://sumo.sourceforge.net/userdoc/Simulation/Output.html
        http://sumo.sourceforge.net/userdoc/Simulation/Why_Vehicles_are_teleporting.html

        """
        if rl_actions is None:

            def rl_actions(*_):
                return None

        info_dict = {}

        observation_spaces = []
        rewards = []
        vels = []
        vehs = []

        veh_ids = []
        veh_speeds = []

        agent_updates_counter = 0

        state = self.env.reset()

        for step in tqdm(range(num_steps)):

            # WARNING: Env reset is not synchronized with agents' cycle time.
            # if step % 86400 == 0 and agent_updates_counter != 0: # 24 hours
            #     self.env.reset()

            state, reward, done, _ = self.env.step(rl_actions(state))

            step_ids = self.env.k.vehicle.get_ids()
            step_speeds = [s for s in self.env.k.vehicle.get_speed(step_ids) if s > 0]

            veh_ids.append(len(step_speeds))
            veh_speeds.append(np.nanmean(step_speeds))

            if self._is_save_step():

                observation_spaces.append(
                    self.env.observation_space.feature_map())

                rewards.append(reward)

                vehs.append(np.nanmean(veh_ids).round(4))
                vels.append(np.nanmean(veh_speeds).round(4))
                veh_ids = []
                veh_speeds = []

                agent_updates_counter += 1

            if done and stop_on_teleports:
                break

            if self.save_agent and (self.tls_type == 'rl' or self.tls_type == 'centralized') and \
                self._is_save_agent_step(agent_updates_counter):
                self.env.tsc.save_checkpoint(self.exp_path)

        # Save train log (data is aggregated per traffic signal).
        info_dict["rewards"] = rewards
        info_dict["velocities"] = vels
        info_dict["vehicles"] = vehs
        info_dict["observation_spaces"] = observation_spaces
        info_dict["actions"] = action_to_double_precision([a for a in self.env.actions_log.values()])
        info_dict["states"] = [s for s in self.env.states_log.values()]

        if self.tls_type not in ('static', 'actuated'):
            self.env.tsc.terminate()
        self.env.terminate()

        return info_dict

    def _is_save_step(self):
        if self.cycle is not None:
            return self.env.duration == 0.0
        return False

    def _is_save_agent_step(self, counter):
        if self.env.duration == 0.0 and counter % self.save_agent_interval == 0:
            return self.train and self.exp_path
        return False
