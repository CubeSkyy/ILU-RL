import sys
import os


class PipeGuard(object):
    """PipeGuards supresses stdout in order to prevent text outputs to interfere
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args, **kwargs):
        sys.stdout.close()
        sys.stdout = self._stdout

