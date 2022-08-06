"""
Class accept a dictionary of known keys, parses each string according to a set of rules and links the parsed strings to a known corresponding function_eval_string found in ACTIONS_DICT, then return a dataframe with columns
["id", "Parsed_Command", "Template_Code", "Variable Value"].
The each element of parsed strings is split in a tuple ranked based on the the order that they are found in the ACTIONS_DICT for that element's Class keyname. You can make custom rules for each key if they do not all meet a certain pattern
EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        config_file_path="mmt_debug.py.debug_settings",
        refine=dict(
            grid_n=5,
        ),
    ),
    MIP=dict(
        agg=True,
    ),
    Neighbors=dict(
        n_neighbors=20,
    ),
    Data_Manager=dict(
        delete_first_month=True,
    ),
    Overfit=dict(
        cscv=dict(
            n_bins=10,
            objective='sharpe_ratio',
            PBO=True,
            PDes=True,
            SD=True,
            POvPNO=True,
        )
    ),
    Data_Manager_1=dict(  # todo add _{} parser
        split_into=2,
    ),
    Filters=dict(
        quick_filters=True,
        delete_loners=True,
    ),
)

Output e.g. id    parsed_string                      functions_to_evaluate
             0    ('Genie','config_file_path.config_dict_name','refine','grid_n','max_variance=5')      ('genie','genie_obj = Genie_obj('config_file_path.config_dict_name')','genie_obj.n_trials = 500','refine','genie_obj.refine.strategy_seeds = strategy_seeds','genie_obj.refine.grid_n_n(variance=5)','genie_obj.run()')
             2    ('MIP','agg')                                                                 ('MIP','mip_obj = MIP_obj(pf_params_and_metrics_df)','mip_obj.agg = previous_mip_values','mip_obj.run()')
             3    ('Neighbors','n_neighbors = 20')                                              ('Neighbors','neighbors_obj = Neighbors_obj(pf_params_and_metrics_df or pf_params_df)','neighbors_obj.run()')
             1    ('Data_Manager','delete_first_month')                                         ('Data_Manager','data_manager_obj = Data_Manager_obj(df)','data_manager_obj.delete_first_month = True','data_manager_obj.run()')
             1    ('Overfit','cscv','n_bins=10','sharpe_ratio','PBO','PDes','SD','POvPNO')      ('Overfit','cscv_obj = cscv_obj(pf_locations or masked_pf_locations)','cscv_obj.n_bins = int(n_bins)','cscv_obj.objective = objective','cscv_obj.PBO = True','cscv_obj.PDes = True','cscv_obj.SD = True','cscv_obj.POvPNO = True','cscv_obj.run()')
             4    ('Data_Manager','split_into',2)                                               ('Data_Manager','data_manager_obj = Data_Manager_obj(df)','data_manager_obj.split_into = int(n_split)','data_manager_obj.run()')
             5    ('Filters','quick_filters','delete_loners')                                   ('Filters','filters_obj = Filters_obj(pf_locations or masked_pf_,locations)','filters_obj.quick_filters()','filters_obj.run()')
"""
import datetime
import subprocess
from multiprocessing import cpu_count
from os import remove
from os.path import exists

import flatdict
import numpy as np
import pandas as pd

from genie_api_actions import ACTIONS_DICT
from utils import multiline_eval, return_unique_name_for_path

Spaces_Program_Info = dict(
    Genie=dict(
        working_dir="/home/ruben/Programs/mini_Genie_test",
        main_path="/home/ruben/Programs/mini_Genie_test/mini_genie_source/main_mini_genie.py",
        config=dict(
            template='from mini_genie_config_template import config_template\n config_template',
            output_config='temp_input_config',
            corresponding_input_flag='--config_file_path',
            command_line_flag=[]
        ),
    ),
    Filters=dict(
        working_dir="/home/ruben/PycharmProjects/Post_Processing_Genie",
        main_path="/home/ruben/PycharmProjects/Post_Processing_Genie/post_processing_genie_source/post_processing_genie_main.py",
        config=dict(
            template='from filters_config_template import config_template\n config_template',
            corresponding_input_flag='--actions',
            command_line_flag=['-f']
        ),
    )
)


class ApiHandler:
    def __init__(self, input_dict, actions_dict=ACTIONS_DICT):
        self.Results = None
        self.input_dict = input_dict
        self.actions_dict = actions_dict
        self.n_head_spaces = len(self.input_dict)
        self.head_spaces_names = self.input_dict.keys()

        self.parsed_strings = []
        self.functions_to_evaluate = []
        self.id = 0
        self.df = pd.DataFrame(columns=["ID", "Parsed_Command", "Template_Code", "Variable_Value"])

        # Sanity check to make sure all keys in runtime are in known actions
        self.prep_input()

        self.runtime_settings = flatdict.FlatDict(self.input_dict, delimiter='.')
        self.actions_settings = flatdict.FlatDict(self.actions_dict, delimiter='.')
        #
        self.check_input()
        #
        # ""Commons""
        self.pf_params_and_metrics_df = None
        self.previous_mip_values = None
        self.common = "dict(" \
                      "self=self," \
                      "datetime=datetime," \
                      ")"

        # self.common = """dict(
        #     self=self
        #     this_elsethis=multiline_eval(expr='from utils import this_elsethis\nthis_elsethis'),
        #     datetime=multiline_eval(expr='import datetime\ndatetime'),
        #     np=multiline_eval(expr='import numpy as np \nnp'),
        #     numpy=multiline_eval(expr='import numpy  \nnumpy'),
        # )"""

        # "dict(" \
        #           "self=self," \
        #           ")"
        #
        #   # "pf_params_and_metrics_df=self.pf_params_and_metrics_df," \
        #                       # "previous_mip_values=self.previous_mip_values" \
        #
        from utils import dotdict
        self.dotdict = dotdict

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    @staticmethod
    class Genie_obj:
        def __init__(self):
            self.n_trials = None
            self.data_files_names = None
            self.tick_size = None
            self.study_name = None
            self.start_date = None
            self.end_date = None
            self.timer_limit = None
            self.Continue = None
            self.batch_size = None
            self.max_initial_combinations = None
            self.stop_after_n_epoch = None
            self.Strategy = None
            self.max_spread_allowed = None
            self.trading_fees = None
            self.max_orders = None
            self.init_cash = None
            self.size = None
            self.ray_init_num_cpus = None
            self.simulate_signals_num_cpus = None
            #
            self.call_dict = Spaces_Program_Info["Genie"]

            self.common = "dict(" \
                          "self=self," \
                          ")"

        def run(self):
            # todo if any of the methods were called, if only one run type selected then use context to compile a call
            #   other wise pop an error indicating only one type per run can be chosen

            if "_Study_Name" in Spaces_Program_Info:
                if self.study_name:
                    assert Spaces_Program_Info["_Study_Name"] == self.study_name
                else:
                    self.study_name = Spaces_Program_Info["_Study_Name"]
            else:
                Spaces_Program_Info["_Study_Name"] = self.study_name

            working_dir = self.call_dict["working_dir"]
            main_path = self.call_dict["main_path"]
            config = self.call_dict["config"]
            # command_line_flags = self.call_dict["command_line_flags"]
            #
            template_settings_dict = multiline_eval(expr=config["template"])

            context = {}
            context.update(self.__dict__,
                           this_elsethis=multiline_eval(expr='from utils import this_elsethis\nthis_elsethis'),
                           datetime=datetime,
                           np=np,
                           cpu_count=cpu_count,
                           )

            temp_config_file_name = f'temp_run_time_settings.py'
            temp_config_file_path = f'{working_dir}/{temp_config_file_name}'
            run_time_settings = multiline_eval(expr=template_settings_dict, context=context)
            #
            # import pprint
            # pprint.pprint(run_time_settings)
            #
            assert exists(working_dir)
            with open(temp_config_file_path, 'w') as fout:
                fout.write(str('from pandas import Timestamp\n'))
                fout.write(str('import datetime\n'))
                fout.write(str('import numpy as np\n'))
                fout.write(str('from numpy import inf\n'))
                fout.write(str(f'{run_time_settings = }'))

            flags = ''
            for i in config["command_line_flag"]:
                flags = f'{flags} {i}'

            cmd_line_str = f'cd {working_dir} && pipenv run {main_path} {flags} {config["corresponding_input_flag"]} {temp_config_file_name}.run_time_settings -gp'
            # for flag, default_value in command_line_flags.items():
            #     usr_value = getattr(self, flag)
            #     usr_value = usr_value if usr_value else command_line_flags[flag]
            #     cmd_line_str = f'{cmd_line_str} -{flag} {usr_value}'

            print(cmd_line_str)

            self.cmd_line_call = cmd_line_str
            self.returned_output = subprocess.call(cmd_line_str, shell=True)
            remove(temp_config_file_path)

            return self

    @staticmethod
    class MIP_obj:
        def __init__(self):
            self.agg = None
            self.n_trials = None

        def run(self):
            return self

    @staticmethod
    class Neighbors_obj:
        def __init__(self):
            self.n_neighbors = None
            self.computations_type = None

        def run(self):
            return self

    @staticmethod
    class Data_Manager_obj:
        def __init__(self):
            self.delete_first_month = None
            self.delete_last_month = None
            self.delete_max_drawdown_month = None
            # ...
            self.n_split = None

        def run(self):
            return self

    @staticmethod
    class Overfit_obj:
        def __init__(self):
            ...

        def cscv(self):
            self.n_bins = 10,
            self.objective = 'sortino',
            self.PBO = None,
            self.PDes = None,
            self.SD = None,
            self.POvPNO = None,

        def run(self):
            return self

    @staticmethod
    class Overfit_obj:
        def __init__(self):
            ...

        def cscv(self):
            self.n_bins = None,
            self.objective = None,
            self.PBO = None,
            self.PDes = None,
            self.SD = None,
            self.POvPNO = None,

        def run(self):
            return self

    @staticmethod
    class Filters_obj:
        def __init__(self):
            self.study_name = None
            self.output_path = None
            self.quick_filters = None
            self.Min_total_trades = None
            self.Profit_factor = None
            self.Expectancy = None
            self.Daily_drawdown = None
            self.Total_drawdown = None
            self.Profit_for_month = None
            self.Total_Win_Rate = None
            self.quick_filters = None
            #
            # self.delete_drawdown = None
            # self.delete_profit = None
            # self.delete_expectency = None
            # self.delete_loners = None
            # ...
            #
            self.call_dict = Spaces_Program_Info["Filters"]

            self.common = "dict(" \
                          "self=self," \
                          ")"

        def run(self):
            # Do
            if "_Study_Name" in Spaces_Program_Info:
                if self.study_name:
                    assert Spaces_Program_Info["_Study_Name"] == self.study_name
                else:
                    self.study_name = Spaces_Program_Info["_Study_Name"]
            else:
                Spaces_Program_Info["_Study_Name"] = self.study_name
            # Finally Do
            self.study_path = f'{Spaces_Program_Info["Genie"]["working_dir"]}/Studies/{self.study_name}'
            assert exists(self.study_path)

            self.default_output_path = f'{Spaces_Program_Info["Genie"]["working_dir"]}/Studies/portfolio_stats.csv'

            working_dir = self.call_dict["working_dir"]
            assert exists(working_dir)
            main_path = self.call_dict["main_path"]
            config = self.call_dict["config"]
            #
            template_settings_dict = multiline_eval(expr=config["template"])
            #
            context = {}
            context.update(self.__dict__,
                           study_path=self.study_path,
                           this_elsethis=multiline_eval(expr='from utils import this_elsethis\nthis_elsethis'),
                           return_unique_name_for_path=return_unique_name_for_path,
                           # datetime=datetime,
                           np=np,
                           # cpu_count=cpu_count,
                           )
            #
            run_time_settings = multiline_eval(expr=template_settings_dict, context=context)
            #
            flags = ''
            for i in config["command_line_flag"]:
                flags = f'{flags} {i}'
            #
            cmd_line_str = f'cd {working_dir} && pipenv run {main_path} {flags} {config["corresponding_input_flag"]} \"{run_time_settings}\"'
            self.cmd_line_call = cmd_line_str
            self.returned_output = subprocess.call(self.cmd_line_call, shell=True)
            return self

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def prep_input(self):
        # > Adding non-first time usage of Head-Spaces "_{int}" handler
        #   Check for keys that start with {Head-Space}, that includes {Head-Space}_{int}
        #       Make note of number of each set of keys, if not 1, then add {Head-Space}_{int}'s to actions_settings
        #       with {Head-Space} values. Then fill as normal.
        from copy import deepcopy
        copy_of_actions_dict = deepcopy(self.actions_dict)
        for hs_key in copy_of_actions_dict.keys():
            hs_space_instance = self.fetch_keys_that_start_with_optimized(self.input_dict, hs_key)
            if len(hs_space_instance) > 1:
                for hs_instance in hs_space_instance[1:]:
                    self.actions_dict[hs_instance] = self.actions_dict[hs_key]

    def check_input(self):
        # > Assert all input keys are valid
        not_recognized_keys = []
        for input_key in self.runtime_settings.keys():
            if input_key not in self.actions_settings.keys():
                not_recognized_keys.append(input_key)
        if not_recognized_keys:  multiline_eval('print(f"{not_recognized_keys} keys is not recognized") \n exit()',
                                                context=dict(not_recognized_keys=not_recognized_keys))

    @staticmethod
    def fetch_keys_that_start_with_optimized(dictionary, start_str):
        """Finds the keys of the input dictionary that start with 'start_str', returns them in the order they are found
        in the original dictionary"""

        keys = []
        for key in dictionary:
            if key[:len(start_str)] == start_str:
                keys.append(key)
        return keys

    @staticmethod
    def fetch_keys_that_end_with_optimized(dictionary, end_str):
        """Finds the keys of the input dictionary that end with 'end_str', returns them in the order they are found in
        the original dictionary"""

        keys = []
        for key in dictionary:
            if key[-len(end_str):] == end_str:
                keys.append(key)
        return keys

    def fill_empty_settings(self, HS):
        temp_dict = {}
        for unheaded_space_key, unheaded_space_value in self.actions_settings[HS].items():
            #
            space_actions_key = f'{HS}.{unheaded_space_key}'
            temp_dict[unheaded_space_key] = self.runtime_settings.get(space_actions_key, None)
        return temp_dict

    def fill_output_df(self):
        # Fill Dataframe with assignments for each command
        index_ = 0
        for i, j in self.Master_Command_List.items():
            print(f'____{i}____')
            self.id += 1

            for parsed_cmd in j:
                if not parsed_cmd.startswith('*'):
                    variable_value = self.runtime_settings[parsed_cmd]
                    template_code = self.actions_settings[parsed_cmd]
                    # self.id += 1
                    index_ += 1
                    self.df.loc[index_] = [self.id, parsed_cmd, template_code, variable_value]
                    # self.df.loc[self.id] = [self.id, parsed_cmd, template_code, variable_value]
                    print('{} {} = variable_value --> "{}"'.format(self.id, parsed_cmd, template_code))
                else:
                    print(parsed_cmd)

            print(f'___________\n')

    def parse_input_dict(self):
        self.Master_Command_List = {}
        command_pipe = ["foo"]
        for item_index, runtime_key in enumerate(self.runtime_settings.keys()):
            Head_Space = runtime_key.split('.')[0]
            #
            if f'{Head_Space}.init' != command_pipe[0]:
                # Initialize All Values of Runtime Settings
                command_pipe = []
                #
                self.runtime_settings[Head_Space] = self.fill_empty_settings(Head_Space)
                assert self.runtime_settings[Head_Space].keys()[0] == f'init'
                #
                # Parse
                for item_index, (runtime_key, runtime_value) in enumerate(self.runtime_settings[Head_Space].items()):
                    runtime_key_split = runtime_key.split('.')
                    for split_index, split_key in enumerate(runtime_key_split):
                        if split_key == 'init':
                            if split_index == item_index == 0:  # initialization for Head_Space
                                parsed_cmd = f'{Head_Space}.init'
                                command_pipe.append(parsed_cmd)
                            else:
                                parsed_cmd = f'{Head_Space}.{runtime_key}'
                                command_pipe.append(f'*{runtime_key_split[split_index - 1]}*')
                                command_pipe.append(parsed_cmd)
                        elif split_key == 'run_command':
                            parsed_cmd = f'{Head_Space}.{runtime_key}'
                            command_pipe.append(parsed_cmd)
                            self.Master_Command_List[Head_Space] = command_pipe
                        elif runtime_value != None:
                            parsed_cmd = f'{Head_Space}.{runtime_key}'
                            if parsed_cmd not in command_pipe:
                                command_pipe.append(parsed_cmd)

                        # else:
                        #     parsed_cmd = f'{Head_Space}.{runtime_key}'
                        #     if parsed_cmd not in command_pipe:
                        #         command_pipe.append(parsed_cmd)
                        #     print(runtime_key)

    def parse(self):
        self.parse_input_dict()
        self.fill_output_df()
        return self.df

    @staticmethod
    def _return_expr_n_context(space_cmds, HS):
        context = {}
        expression = ''

        _space_cmds = space_cmds[["Parsed_Command", "Template_Code", "Variable_Value"]].transpose()
        for index, (index_key, (api_call, cmd, contx)) in enumerate(_space_cmds.items()):
            api_call_dot_split = api_call.split('.')
            assert HS == api_call_dot_split[0]

            if "=" in cmd:
                if contx:
                    x = cmd.split('.')[-1].split('=')[0].strip()
                    context[f'{x}'] = contx
                else:
                    # print(f'{api_call_dot_split[-1].strip() = }')
                    assert api_call_dot_split[-1].strip() == 'init'
                    split_cmd = cmd.split('=')
                    assert len(split_cmd) == 2

                    if index == 0:  # Initialization
                        hs_init_obj = split_cmd[0].strip()
                        icontext = split_cmd[1].strip()
                    else:
                        init_obj = split_cmd[0].strip()
                        assert init_obj.split('.')[0] == hs_init_obj
                        icontext = split_cmd[1]
                        # x = cmd.split('.')[-1].split('=')[0].strip()
                        x = init_obj.split('.')[-1]
                        context[f'{x}'] = icontext
                        print(f'!!Warning {split_cmd[1] = }!!')
            else:
                assert contx != False
                cmd_dot_split = cmd.split('.')
                assert hs_init_obj == cmd_dot_split[0]

                if cmd_dot_split[1] == 'run()':
                    assert api_call.split('.')[-1].strip() == 'run_command'
                    assert cmd.split('.')[-1].strip() == 'run()'
                # elif cmd_dot_split[1] == '':
                else:
                    assert api_call_dot_split[-1] == 'init'
                    assert api_call_dot_split[-2] in cmd
                    assert cmd_dot_split[0] in cmd

            expression = f'{expression}\n{cmd}'

        assert len(expression.splitlines()) - 1 == len(space_cmds)
        return expression, context

    def _run_space(self, space_id, HS):
        space_cmds = self.df[self.df["ID"] == space_id]
        expression, context = self._return_expr_n_context(space_cmds, HS)
        context.update(eval(self.common))
        return multiline_eval(expr=expression, context=context)

    def run(self):
        Results = dict()
        for space_id, HS in enumerate(self.head_spaces_names, start=1):
            Results[HS] = self._run_space(space_id, HS)
        #
        self.Results = Results
        for i, j in self.Results.items():
            print(f'{j.__dict__}')
        return self.Results


if '__main__' == __name__:
    EXAMPLE_INPUT_DICT = dict(
        Genie=dict(
            study_name='Test_Study',
            Strategy='mini_genie_source/Strategies/Money_Maker_Strategy.py',
            data_files_names=['AUDUSD'],
            tick_size=[0.001],
            init_cash=1_000_000,
            size=100_000,
            start_date=datetime.datetime(month=1, day=1, year=2022),
            end_date=datetime.datetime(month=3, day=1, year=2022),
            #
            # Continue=False,
            batch_size=10,
            timer_limit=None,
            stop_after_n_epoch=5,
            max_initial_combinations=2,
            trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
            max_orders=10,
        ),
        Filters=dict(
            study_name='Test_Study',
            Min_total_trades=1,
            Profit_factor=1.0,
            Expectancy=0.01,
            Daily_drawdown=0.05,
            Total_drawdown=0.1,
            Profit_for_month=0.1,
            Total_Win_Rate=0.03,
            quick_filters=True,
            # delete_loners=True,
        ),
        # MIP=dict(
        #     agg=True,
        # ),
        # Neighbors=dict(
        #     n_neighbors=20,
        # ),
        # Data_Manager=dict(
        #     delete_first_month=True,
        # ),
        # Overfit=dict(
        #     cscv=dict(
        #         n_bins=10,
        #         objective='sharpe_ratio',
        #         PBO=True,
        #         PDes=True,
        #         SD=True,
        #         POvPNO=True,
        #     )
        # ),
        # Data_Manager_1=dict(  # todo add _{} parser
        #     n_split=2,
        # ),
        # Filters=dict(
        #     quick_filters=True,
        #     delete_loners=True,
        # ),
    )

    api_handler = ApiHandler(EXAMPLE_INPUT_DICT)
    api_handler.parse()
    # print(api_handler.df[['Template_Code', 'Variable_Value']])
    api_handler.run()
