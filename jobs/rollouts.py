"""
    jobs/rollouts.py
"""

from pathlib import Path
import itertools
import time
import sys
from os import environ
import json
import tempfile
import argparse
import configparser
import multiprocessing
import multiprocessing.pool
from collections import defaultdict

from ilurl.utils.decorators import processable
from models.rollout import main as roll
from ilurl.utils import str2bool

ILURL_HOME = environ['ILURL_HOME']
CONFIG_PATH = \
    Path(f'{ILURL_HOME}/config/')

mp = multiprocessing.get_context('spawn')


class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass

class NoDaemonContext(type(multiprocessing.get_context('spawn'))):
    Process = NoDaemonProcess

class NonDaemonicPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NonDaemonicPool, self).__init__(*args, **kwargs)

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This scripts runs a set of rollouts for every checkpoint stored
            on the experiment path.
        """
    )
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='''A directory which it\'s subdirectories are train runs.''')

    parsed = parser.parse_args()
    sys.argv = [sys.argv[0]]

    return parsed


def delay_roll(args):
    """Delays execution.

        Parameters:
        -----------
        * args: tuple
            Position 0: execution delay of the process.
            Position 1: store the train config file.

        Returns:
        -------
        * fnc : function
            An anonymous function to be executed with a given delay
    """
    time.sleep(args[0])
    return roll(args[1])


def concat(evaluations):
    """Receives an experiments' json and merges it's contents

    Params:
    -------
        * evaluations: list
        list of rollout evaluations

    Returns:
    --------
        * result: dict
        where `id` key certifies that experiments are the same
              `list` params are united
              `numeric` params are appended

    """
    result = {}
    result['id'] = []
    for qtb in evaluations:
        exid = 99999
        # can either be a rollout from the prev
        # exid or a new experiment
        if exid not in result['id']:
            result['id'].append(exid)

        for k, v in qtb.items():
            is_iterable = isinstance(v, list) or isinstance(v, dict)
            # check if integer fields match
            # such as cycle, save_step, etc
            if not is_iterable:
                if k in result:
                    if result[k] != v:
                        raise ValueError(
                            f'key:\t{k}\t{result[k]} and {v} should match'
                        )
                else:
                    result[k] = v
            else:
                if k not in result:
                    result[k] = defaultdict(list)
                result[k][exid].append(v)
    return result


def rollout_batch(experiment_dir=None):

    print('\nRUNNING jobs/rollouts.py\n')

    if not experiment_dir:

        # Read script arguments.
        args = get_arguments()
        # Clear command line arguments after parsing.

        batch_path = Path(args.experiment_dir)

    else:
        batch_path = Path(experiment_dir)

    chkpt_pattern = 'checkpoints'

    # Get names of train runs.
    experiment_names = list({p.parents[0] for p in batch_path.rglob(chkpt_pattern)})

    # Get checkpoints numbers.
    chkpts_dirs = [p for p in batch_path.rglob(chkpt_pattern)]
    if len(chkpts_dirs) == 0:
        raise ValueError('No checkpoints found.')

    run_config = configparser.ConfigParser()
    run_config.read(str(CONFIG_PATH / 'run.config'))

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args", "train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        'must match the number of runs (`num_runs`) argument.')

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')

    # Read rollouts arguments from rollouts.config file.
    rollouts_config = configparser.ConfigParser()
    rollouts_config.read(str(CONFIG_PATH / 'rollouts.config'))
    num_rollouts = int(rollouts_config.get('rollouts_args', 'num-rollouts'))
    rollouts_config.remove_option('rollouts_args', 'num-rollouts')

    # Write .xml files for plots creation.
    rollouts_config.set('rollouts_args', 'sumo-emission', str(True))

    rollout_time = rollouts_config.get('rollouts_args', 'rollout-time')
    print(f'\nArguments (jobs/rollouts.py):')
    print('-------------------------')
    print(f'Experiment dir: {batch_path}')
    print(f'Number of processors: {num_processors}')
    print(f'Num. train runs found: {len(experiment_names)}')
    print(f'Num. rollouts per train run: {num_rollouts}')
    print(f'Num. rollout total: {len(experiment_names) * num_rollouts}')
    print(f'Rollout time: {rollout_time}\n')

    # Allocate seeds.
    custom_configs = []
    for rn, rp in enumerate(experiment_names):
        base_seed = max(train_seeds) + num_rollouts * rn
        for rr in range(num_rollouts):
            seed = base_seed + rr + 1
            custom_configs.append((rp, seed))

    with tempfile.TemporaryDirectory() as f:

        tmp_path = Path(f)
        # Create a config file for each rollout
        # with the respective seed. These config
        # files are stored in a temporary directory.
        rollouts_cfg_paths = []
        cfg_key = "rollouts_args"
        for cfg in custom_configs:
            run_path = cfg[0]
            seed = cfg[1]

            # Setup custom rollout settings.
            rollouts_config.set(cfg_key, "run-path", str(run_path))
            rollouts_config.set(cfg_key, "seed", str(seed))
            
            # Write temporary train config file.
            cfg_path = tmp_path / f'rollouts-{run_path.name}-{seed}.config'
            rollouts_cfg_paths.append(str(cfg_path))
            with cfg_path.open('w') as fw:
                rollouts_config.write(fw)

        # rvs: directories' names holding experiment data
        if num_processors > 1:
            pool = NonDaemonicPool(num_processors)
            rvs = pool.map(delay_roll, [(delay, [cfg])
                            for (delay, cfg) in zip(range(len(rollouts_cfg_paths)), rollouts_cfg_paths)])
            pool.close()
        else:
            rvs = []
            for cfg in rollouts_cfg_paths:
                rvs.append(delay_roll((0.0, [cfg])))

    res = concat(rvs)
    filename = f'rollouts_test.json'
    target_path = batch_path / filename
    with target_path.open('w') as fj:
        json.dump(res, fj)

    sys.stdout.write(str(batch_path))

    return str(batch_path)

@processable
def rollout_job():
    # Suppress textual output.
    return rollout_batch()

if __name__ == '__main__':
    #rollout_job()
    rollout_batch() # Use this line for textual output.
