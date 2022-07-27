"""
Design an api handler class for a python 3 project. The class needs to accept a dictionary of known keys, parse each string according to a set of rules and link the parsed strings to a known corresponding function_eval_string found in ACTIONS_DICT, then return a dataframe with columns ["id","parsed_string","functions_to_evaluate"].
The each element of parsed strings is split in a tuple ranked based on the the order that they are found in the ACTIONS_DICT for that element's Class keyname. You can make custom rules for each key if they do not all meet a certain pattern
EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        n_trials=500,
        config_file_path="mmt_debug.py.debug_settings",
        refine=dict(
            max_variance=5,
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
    Data_Manager_1=dict(
        split_into=2,
    ),
    Filters=dict(
        quick_filters=True,
        delete_loners=True,
    ),
)

Output e.g. id    parsed_string                      functions_to_evaluate
             0    ('Genie','config_file_path.config_dict_name','n_trials=500','refine','grid','max_variance=5')      ('genie','genie_obj = Genie_obj('config_file_path.config_dict_name')','genie_obj.n_trials = 500','refine','genie_obj.refine.strategy_seeds = strategy_seeds','genie_obj.refine.grids(variance=5)','genie_obj.run()')
             2    ('MIP','agg')                                                                 ('MIP','mip_obj = MIP_obj(pf_params_and_metrics_df)','mip_obj.agg = previous_mip_values','mip_obj.run()')
             3    ('Neighbors','n_neighbors = 20')                                              ('Neighbors','neighbors_obj = Neighbors_obj(pf_params_and_metrics_df or pf_params_df)','neighbors_obj.run()')
             1    ('Data_Manager','delete_first_month')                                         ('Data_Manager','data_manager_obj = Data_Manager_obj(df)','data_manager_obj.delete_first_month = True','data_manager_obj.run()')
             1    ('Overfit','cscv','n_bins=10','sharpe_ratio','PBO','PDes','SD','POvPNO')      ('Overfit','cscv_obj = cscv_obj(pf_locations or masked_pf_locations)','cscv_obj.n_bins = int(n_bins)','cscv_obj.objective = objective','cscv_obj.PBO = True','cscv_obj.PDes = True','cscv_obj.SD = True','cscv_obj.POvPNO = True','cscv_obj.run()')
             4    ('Data_Manager','split_into',2)                                               ('Data_Manager','data_manager_obj = Data_Manager_obj(df)','data_manager_obj.split_into = int(n_split)','data_manager_obj.run()')
             5    ('Filters','quick_filters','delete_loners')                                   ('Filters','filters_obj = Filters_obj(pf_locations or masked_pf_,locations)','filters_obj.quick_filters()','filters_obj.run()')
"""

ACTIONS_DICT = dict(
    Genie=dict(
        init='genie_obj = Genie_obj()',
        config_file_path='config_file_path.config_dict_name',
        n_trials='genie_obj.n_trials = int(n_trials)',
        refine=dict(
            init='genie_obj.refine.strategy_seeds = strategy_seeds',
            grid='genie_obj.refine.grids = int(grid_n)',
            product='genie_obj.refine.product = True',
            search_algorythm=dict()
        ),
        run_command='genie_obj.run()',
    ),
    MIP=dict(
        init='mip_obj = MIP_obj(pf_params_and_metrics_df)',
        n_parameters='mip_obj.n_parameters = return_n_parameters',
        agg='mip_obj.agg = previous_mip_values',
        run_command='mip_obj.run()',
    ),
    Overfit=dict(
        init='overfit_obj = Overfit_obj(pf_locations or masked_pf_locations)',
        cscv=dict(
            init='overfit_obj.cscv = True',
            n_bins='overfit_obj.n_bins = int(n_bins)',
            objective='overfit_obj.objective = objective',
            PBO='overfit_obj.PBO = True',
            PDes='overfit_obj.PDes = True',
            SD='overfit_obj.SD = True',
            POvPNO='overfit_obj.POvPNO = True',
        ),
        run_command='overfit_obj.run()'
    ),
    Neighbors=dict(
        init='neighbors_obj = Neighbors_obj(pf_params_and_metrics_df or pf_params_df)',
        n_neighbors='neighbors_obj.n_neighbors = n_neighbors',
        computations_type='neighbors_obj.computations_type = computations_type',
        run_command='neighbors_obj.run()',
    ),
    Filters=dict(
        init='filters_obj = Filters_obj(pf_locations or masked_pf_locations)',
        quick_filters='filters_obj.quick_filters()',
        #
        delete_drawdown='filters_obj.delete_drawdown = drawdown_cmd',
        delete_profit='filters_obj.delete_profit = profit_cmd',
        delete_expectency='filters_obj.delete_expectency = expectency_cmd',
        delete_loners='filters_obj.delete_loners = True',
        # ...
        run_command='filters_obj.run()',
    ),
    Data_Manager=dict(
        init='data_manager_obj = Data_Manager_obj(df)',
        delete_first_month='data_manager_obj.delete_first_month = True',
        delete_last_month='data_manager_obj.delete_last_month = True',
        delete_max_drawdown_month='data_manager_obj.delete_max_drawdown_month = True',
        # ...
        split_into='data_manager_obj.split_into = int(n_split)',
        run_command='data_manager_obj.run()',
    ),
)

import pandas as pd


class ApiHandler:
    def __init__(self, input_dict, actions_dict=ACTIONS_DICT):
        self.input_dict = input_dict
        self.actions_dict = actions_dict
        self.parsed_strings = []
        self.functions_to_evaluate = []
        self.id = 0
        self.df = pd.DataFrame(columns=["id", "Parsed_Command", "Template_Code", "Variable Value"])

        # Sanity check to make sure all keys in runtime are in known actions
        self.prep_input()

        import flatdict
        self.runtime_settings = flatdict.FlatDict(self.input_dict, delimiter='.')
        self.actions_settings = flatdict.FlatDict(self.actions_dict, delimiter='.')
        #
        self.check_input()
        #
        self.prepare_runtime_settings(print_settings=True)

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
        if not_recognized_keys:  print(f"{not_recognized_keys} keys is not recognized")

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

    def prepare_runtime_settings(self, print_settings=False):
        self.Master_Command_List = {}
        command_pipe = ["foo"]
        for item_index, runtime_key in enumerate(self.runtime_settings.keys()):
            Head_Space = runtime_key.split('.')[0]
            #
            if f'{Head_Space}.init' != command_pipe[0]:
                print(Head_Space)
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
        #   #   #   #   #
        #
        #Fill Dataframe with assignments for each command
        for i, j in self.Master_Command_List.items():
            print(f'____{i}____')
            for parsed_cmd in j:
                if not parsed_cmd.startswith('*'):
                    variable_value = self.runtime_settings[parsed_cmd]
                    template_code = self.actions_settings[parsed_cmd]
                    self.id += 1
                    self.df.loc[self.id] = [self.id, parsed_cmd, template_code, variable_value]
                    print('{} {} = variable_value --> "{}"'.format(self.id, parsed_cmd, template_code))

                else:
                    print(parsed_cmd)
            print(f'___________\n')

        exit()

    def parse_input_dict_(self, sub_dict):
        for key, value in sub_dict.items():
            if isinstance(value, dict):
                self.parse_input_dict(value)
            else:
                self.parse_string(key, value)

    def parse_input_dict(self):
        print(self.input_dict)
        for key, value in self.input_dict.items():

            exit()
            if isinstance(value, dict):
                self.parse_input_dict_(value)
            else:
                self.parse_string(key, value)

    def parse_string(self, key, value):
        if key in ACTIONS_DICT:
            self.parsed_strings.append(key)
            self.functions_to_evaluate.append(ACTIONS_DICT[key]['init'])
            if isinstance(value, dict):
                for key, value in value.items():
                    if key in ACTIONS_DICT[key]:
                        self.parsed_strings.append(key)
                        self.functions_to_evaluate.append(ACTIONS_DICT[key][key])
                    else:
                        self.parsed_strings.append(key)
                        self.functions_to_evaluate.append(ACTIONS_DICT[key]['run_command'])
            else:
                self.parsed_strings.append(key)
                self.functions_to_evaluate.append(ACTIONS_DICT[key]['run_command'])
        else:
            self.parsed_strings.append(key)
            self.functions_to_evaluate.append(ACTIONS_DICT[key]['run_command'])

    def create_df(self):
        self.df.loc[self.id] = [self.id, self.parsed_strings, self.functions_to_evaluate]
        self.id += 1
        self.parsed_strings = []
        self.functions_to_evaluate = []

    def run(self):
        self.parse_input_dict()
        print(f'After self.parse_input_dict(): {self.parsed_strings}')
        exit()
        self.create_df()
        return self.df


EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        n_trials=500,
        config_file_path="mmt_debug.py.debug_settings",
        refine=dict(
            grid=5,
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

api_handler = ApiHandler(EXAMPLE_INPUT_DICT)
df = api_handler.run()
print(df)
print(api_handler.Master_Command_List)
