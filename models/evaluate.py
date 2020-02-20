'''Evaluation script for smart grid network'''

__author__ = 'Guilherme Varela'
__date__ = '2019-09-24'

import os
from glob import glob
import json
import argparse

import dill

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightQLEnv
from ilurl.networks.base import Network

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

NET_PATH = \
    f'{ILURL_HOME}/data/networks/'

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on
            custom environment with with presets saved on data/networks
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('dir_pickle', type=str, nargs='?',
                        default=f'{EMISSION_PATH}', help='Path to pickle')

    parser.add_argument('--experiment-time', '-t', dest='time', type=int,
                        default=900, nargs='?',
                        help='Simulation\'s real world time in seconds')

    parser.add_argument('--inflows-switch', '-W', dest='switch', type=str2bool,
                        default=False, nargs='?',
                        help='Simulation\'s real world time in seconds')


    parser.add_argument('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation')

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate(env, code, policies, horizon, path, render=False):
    # load all networks with config -- else build
    netid = env.network.network_id
    routes_path = \
        f"{NET_PATH}{netid}/{netid}"

    routes = sorted(glob(f"{routes_path}.[0-9].{horizon}.{code}.rou.xml"))
    for i, route in enumerate(routes):
        #TODO: save network objects rather then routes
        network = Network.load(env.network.network_id, route)
        network.name = env.network.name

        env_eval = TrafficLightQLEnv(env.env_params,
                                     env.sim_params,
                                     env.ql_params,
                                     network)
        env_eval.sim_params.render = render
        env_eval.stop = True
        num_iterations = len(policies) + 1
        exp_eval = Experiment(env_eval,
                              dir_path=path,
                              train=False,
                              policies=policies)
        print(f"Running evaluation {i + 1}")
        info = exp_eval.run(num_iterations, horizon)
        ipath = os.path.join(path, f'{env_eval.network.name}.{i}.eval.info.json')
        with open(ipath, 'w') as f:
            json.dump(info, f)


if __name__ == '__main__':
    args = get_arguments()

    render = args.render
    dir_pickle = args.dir_pickle
    time = args.time
    x = 'w' if args.switch else 'l'

    paths = glob(f"{dir_pickle}/*.pickle")
    if not any(paths):
        raise Exception("Environment pickles must be saved on root")

    # TODO: handle case with multiple environments
    env = None
    policies = []   # memories
    for path in paths:
        with open(path, mode='rb') as f:
            obj = dill.load(f)
        if isinstance(obj, TrafficLightQLEnv):
            env = obj
        else:
            policies.append(obj)

    evaluate(env, x, policies, time, dir_pickle, render=render)
