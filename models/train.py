"""Provides baseline for networks

    References:
    ==========
    * seed:
    https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
    http://sumo.sourceforge.net/userdoc/Simulation/Randomness.html
"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'
import json
from os import environ
from pathlib import Path

import numpy as np
import random
import configargparse
from configargparse import ArgumentTypeError
import configparser

from flow.core.params import EnvParams, SumoParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

from ilurl.core.experiment import Experiment
from ilurl.core.params import QLParams
from ilurl.core.params import MDPParams
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network

# TODO: move this inside networks
from ilurl.loaders.nets import get_tls_custom

ILURL_PATH = Path(environ['ILURL_HOME'])
EMISSION_PATH = ILURL_PATH / 'data/emissions/'
CONFIG_PATH = ILURL_PATH / 'config'

def get_arguments(config_file):

    if config_file is None:
        config_file = []

    flags = configargparse.ArgParser(
        default_config_files=config_file,
        description="""
            This script runs a traffic light simulation based on
            custom environment with presets saved on data/networks
        """
    )

    flags.add('--network', '-n', type=str, nargs='?', dest='network',
              default='intersection',
              help='Network to be simulated')

    flags.add('--experiment-time', '-t', dest='time', type=int,
              default=90000, nargs='?',
              help='Simulation\'s real world time in seconds')

    flags.add('--experiment-log', '-l', dest='log_info',
              type=str2bool, default=False, nargs='?',
              help='''Whether to save experiment-related data in a JSON file
                      thoughout training (allowing to live track training)''')

    flags.add('--experiment-log-interval', dest='log_info_interval',
              type=int, default=200, nargs='?',
              help='''[ONLY APPLIES IF --experiment-log is TRUE]
              Log into json file interval (in agent update steps)''')

    flags.add('--experiment-save-agent', '-a', dest='save_agent',
              type=str2bool, default=False, nargs='?',
              help='Whether to save RL-agent parameters throughout training')

    flags.add('--experiment-save-agent-interval', dest='save_agent_interval',
              type=int, default=500, nargs='?',
              help='''[ONLY APPLIES IF --experiment-save-agent is TRUE]
              Save agent interval (in agent update steps)''')

    flags.add('--experiment-seed', '-d', dest='seed', type=int,
              default=None, nargs='?',
              help='''Sets seed value for both rl agent and Sumo.
                     `None` for rl agent defaults to RandomState()
                     `None` for Sumo defaults to a fixed but arbitrary seed''')

    flags.add('--sumo-render', '-r', dest='render', type=str2bool,
              default=False, nargs='?',
              help='Renders the simulation')

    flags.add('--sumo-emission', '-e',
              dest='emission', type=str2bool, default=False, nargs='?',
              help='Saves emission data from simulation on /data/emissions')

    flags.add('--tls-type', '-y',
              dest='tls_type', type=str, choices=('controlled', 'actuated',
              'static', 'random'), default='controlled', nargs='?',
              help='Saves emission data from simulation on /data/emissions')

    flags.add('--inflows-switch', '-W', dest='switch',
              type=str2bool, default=False, nargs='?',
              help='''Assign higher probability of spawning a vehicle
                   every other hour on opposite sides''')

    # TODO: Remove this.
    flags.add('--env-normalize', dest='normalize',
              type=str2bool, default=True, nargs='?',
              help='''If true will normalize grid and target''')

    return flags.parse_args()


def str2bool(v, exception=None):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('boolean value expected')

def print_arguments(args):

    print('\nArguments (models/train.py):')
    print('\tExperiment network: {0}'.format(args.network))
    print('\tExperiment time: {0}'.format(args.time))
    print('\tExperiment seed: {0}'.format(args.seed))
    print('\tExperiment log info: {0}'.format(args.log_info))
    print('\tExperiment log info interval: {0}'.format(args.log_info_interval))
    print('\tExperiment save RL agent: {0}'.format(args.save_agent))
    print('\tExperiment save RL agent interval: {0}'.format(args.save_agent_interval))

    print('\tSUMO render: {0}'.format(args.render))
    print('\tSUMO emission: {0}'.format(args.emission))
    print('\tSUMO tls_type: {0}'.format(args.tls_type))

    print('\tInflows switch: {0}\n'.format(args.switch))

    print('\tNormalize state-space (speeds): {0}\n'.format(args.normalize))


def main(train_config=None):

    flags = get_arguments(train_config)
    print_arguments(flags)

    inflows_type = 'switch' if flags.switch else 'lane'

    # Load cycle time and TLS programs.
    baseline = (flags.tls_type == 'actuated')
    cycle_time, programs = get_tls_custom(flags.network, baseline=baseline)

    network_args = {
        'network_id': flags.network,
        'horizon': flags.time,
        'demand_type': inflows_type,
        'insertion_probability': 0.1,
    }
    if flags.tls_type == 'actuated':
        network_args['tls'] = programs

    network = Network(**network_args)
    normalize = flags.normalize

    # Create directory to store data.
    experiment_path = EMISSION_PATH / network.name
    if not experiment_path.exists():
        experiment_path.mkdir()
    print(f'Experiment: {str(experiment_path)}')

    sumo_args = {
        'render': flags.render,
        'print_warnings': False,
        'sim_step': 1, # step = 1 by default.
        'restart_instance': True
    }

    # Setup seeds.
    if flags.seed is not None:
        random.seed(flags.seed)
        np.random.seed(flags.seed)
        sumo_args['seed'] = flags.seed

    if flags.emission:
        sumo_args['emission_path'] = experiment_path.as_posix()
    sim_params = SumoParams(**sumo_args)

    additional_params = {}
    additional_params.update(ADDITIONAL_ENV_PARAMS)
    additional_params['cycle_time'] = cycle_time
    additional_params['tls_type'] = flags.tls_type

    # TODO: make this an argument.
    # Maybe join with tls_type?
    additional_params['agent_type'] = 'QL'
    env_args = {
        'evaluate': True,
        'additional_params': additional_params
    }
    env_params = EnvParams(**env_args)

    # Number of phases per traffic light.
    phases_per_tls = {tid: len(network.tls_phases[tid])
                    for tid in network.tls_ids}

    # Number of actions per traffic light.
    num_actions = {tid: len(programs[tid])
                    for tid in network.tls_ids}

    category_counts = [5, 10, 15, 20, 25, 30]
    if normalize:
        category_speeds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    else:
        category_speeds = [2, 3, 4, 5, 6, 7]

    # TODO: save mdp params
    mdp_args = {
                'num_actions': num_actions,
                'phases_per_traffic_light': phases_per_tls,
                'states': ('speed', 'count'),
                'category_counts': category_counts,
                'category_speeds': category_speeds,
                'normalize_state_space': normalize,
                'discretize_state_space': True,
                'reward': {'type': 'target_velocity',
                    'additional_params': {
                        'target_velocity': 1.0
                    }
                }
    }
    mdp_params = MDPParams(**mdp_args)

    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        network=network,
        TLS_programs=programs,
        mdp_params=mdp_params,
    )

    # Override possible inconsistent params.
    if flags.tls_type not in ('controlled',):
        env.stop = True
        flags.save_agent = False
        flags.save_agent_interval = None

    exp = Experiment(
            env=env,
            dir_path=experiment_path.as_posix(),
            train=True,
            log_info=flags.log_info,
            log_info_interval=flags.log_info_interval,
            save_agent=flags.save_agent,
            save_agent_interval=flags.save_agent_interval
     )

    # Store parameters.
    parameters = {}
    parameters['network_args'] = network_args
    parameters['sumo_args'] = sumo_args
    parameters['env_args'] = env_args
    parameters['programs'] = programs

    filename = \
            f"{env.network.name}.params.json"

    params_path = experiment_path / filename 
    with params_path.open('w') as f:
        json.dump(parameters, f)

    info_dict = exp.run(flags.time)

    # Save train log.
    filename = \
            f"{env.network.name}.train.json"

    result_path = experiment_path / filename
    with result_path.open('w') as f:
        json.dump(info_dict, f)

    return str(experiment_path)

if __name__ == '__main__':
    train_path = CONFIG_PATH / 'train.config'
    train_config = configparser.ConfigParser()
    train_config.read(str(train_path))
    main(train_config)
