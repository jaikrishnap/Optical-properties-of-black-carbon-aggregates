"""
Collection of general utility functions.
"""
import importlib
import inspect
import os
import sys
import tempfile
import types
from typing import Tuple, Union, Sequence

from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds


def ceil_div(a: int, b: int) -> int:
    # assert a >= 0 and b > 0
    return (a + b - 1) // b


class Bunch:
    def __init__(self, adict):
        self._adict = adict

    def __getattr__(self, item):
        return self._adict[item]

    def __getitem__(self, item):
        return self._adict[item]


def make_experiment(exp_name: str = None, ingredients: Sequence[Ingredient] = ()) -> Experiment:
    # TODO: sacred's source detection seems to break when we instantiate the experiment
    #   outside the experiment main file

    # Get the main directory
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    #print(utils_dir) #D:\TUK\5. Winter 2021\Project 1\SacredTemplate\src\utils
    main_dir = os.path.abspath(os.path.join(utils_dir, '..', '..'))
    #print(main_dir) #D:\TUK\5. Winter 2021\Project 1\SacredTemplate
    log_dir = os.path.join(main_dir, 'log')
    #print(log_dir)#D:\TUK\5. Winter 2021\Project 1\SacredTemplate\log

    if exp_name is None:
        exp_name, _ = os.path.splitext(os.path.basename(sys.argv[0]))
        print(exp_name) #sample_experiment
    experiment = Experiment(exp_name, base_dir=main_dir, ingredients=ingredients)
    #print(ingredients) #()
    # For file storage
    #experiment_path = os.path.join(log_dir, exp_name)
    #experiment.observers.append(FileStorageObserver(experiment_path))
    # Alternatively use the MongoObserver
    experiment.observers.append(MongoObserver(url='mongodb://root:example@localhost', db_name=exp_name))
    # This avoids logging every tqdm progress bar update to the cout.txt file
    experiment.captured_out_filter = apply_backspaces_and_linefeeds
    return experiment


def make_experiment_tempfile(filename: str, run: Run, suffix=None,mode: str = 'w+b'):
    # Create temporary file and add it as an artifact to the run as soon as it is closed
    # Windows won't let us read a tempfile while it is open, so we have to set delete=False,
    # close it and then delete it manually
    tmpfile = tempfile.NamedTemporaryFile(delete=False, mode=mode, suffix=suffix)
    close_method = tmpfile.close

    def close(self):
        close_method()
        run.add_artifact(self.name, name=filename)
        # Cleanup
        try:
            os.remove(self.name)
        except FileNotFoundError:
            # We don't care if the file was already deleted
            pass

    tmpfile.close = types.MethodType(close, tmpfile)
    return tmpfile


def str2cls(fully_qualified_name: str):
    # Ignore if it is already a class
    if inspect.isclass(fully_qualified_name):
        return fully_qualified_name

    fully_qualified_name = fully_qualified_name.rsplit('.', 1)
    if len(fully_qualified_name) == 2:
        # load the module, will raise ImportError if module cannot be loaded
        m = importlib.import_module(fully_qualified_name[0])
    else:
        m = globals()

    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, fully_qualified_name[-1])
    return c
