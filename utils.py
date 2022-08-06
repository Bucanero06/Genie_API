import ast
import inspect
from os.path import exists
from subprocess import Popen, PIPE, CalledProcessError

import _typing as tp


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def multiline_eval(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate several lines of input, returning the result of the last line.

    Args:
        expr: The expression to evaluate.
        context: The context to evaluate the expression in.

    Returns:
        The result of the last line of the expression.

    Raises:
        SyntaxError: If the expression is not valid Python.
        ValueError: If the expression is not valid Python.
    """
    if context is None:
        context = {}
    tree = ast.parse(inspect.cleandoc(expr))
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
    exec(compile(exec_expr, "file", "exec"), context)
    return eval(compile(eval_expr, "file", "eval"), context)


def next_path(path_pattern):
    import os

    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


def this_elsethis(this, elsethis):
    x = this if this != None else elsethis
    if x == None:
        print('Error!! Required value missing')
        exit()
    return x


def fetch_run_time_settings(self):
    """
    Adds self.run_time_settings
    @return self.run_time_settings (to be used if needed)
    """

    run_time_settings = self.load_module_from_path(self.run_time_module_path,
                                                   object_name=self.run_time_dictionary_name)
    #
    optimization_sim_path = run_time_settings["Portfolio_Settings"]["Simulator"]["optimization"]
    optimization_sim_module_path, optimization_sim_dictionary_name = optimization_sim_path.rsplit('.', 1)
    optimization_sim = self.load_module_from_path(optimization_sim_module_path,
                                                  object_name=optimization_sim_dictionary_name)
    #
    strategy_sim_path = run_time_settings["Strategy_Settings"]["Strategy"]
    strategy_sim_module_path, strategy_sim_dictionary_name = strategy_sim_path.rsplit('.', 1)
    strategy_sim = self.load_module_from_path(strategy_sim_module_path,
                                              object_name=strategy_sim_dictionary_name)
    #
    run_time_settings["Strategy_Settings"]["Strategy"] = strategy_sim
    run_time_settings["Portfolio_Settings"]["Simulator"]["optimization"] = optimization_sim
    #
    self.run_time_settings = run_time_settings

    return self.run_time_settings


def return_unique_name_for_path(path_given):
    output_path = path_given
    if exists(output_path):
        split_path = output_path.rsplit('.', 1)
        output_path = next_path(f'{split_path[0]}_%s.{split_path[1]}')
    return output_path



def Execute(command):
    #>Executes to command line
    with Popen(command, stdout=PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here
    if p.returncode != 0:
        # raise CalledProcessError(p.returncode, p.args)
        exit()
