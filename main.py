"""
Class accept a dictionary of known keys, parses each string according to a set of rules and links the parsed strings to a known corresponding function_eval_string found in ACTIONS_DICT, then return a dataframe with columns
["id", "Parsed_Command", "Template_Code", "Variable Value"].
The each element of parsed strings is split in a tuple ranked based on the the order that they are found in the ACTIONS_DICT for that element's Class keyname. You can make custom rules for each key if they do not all meet a certain pattern
EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        n_trials=500,
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
             0    ('Genie','config_file_path.config_dict_name','n_trials=500','refine','grid_n','max_variance=5')      ('genie','genie_obj = Genie_obj('config_file_path.config_dict_name')','genie_obj.n_trials = 500','refine','genie_obj.refine.strategy_seeds = strategy_seeds','genie_obj.refine.grid_n_n(variance=5)','genie_obj.run()')
             2    ('MIP','agg')                                                                 ('MIP','mip_obj = MIP_obj(pf_params_and_metrics_df)','mip_obj.agg = previous_mip_values','mip_obj.run()')
             3    ('Neighbors','n_neighbors = 20')                                              ('Neighbors','neighbors_obj = Neighbors_obj(pf_params_and_metrics_df or pf_params_df)','neighbors_obj.run()')
             1    ('Data_Manager','delete_first_month')                                         ('Data_Manager','data_manager_obj = Data_Manager_obj(df)','data_manager_obj.delete_first_month = True','data_manager_obj.run()')
             1    ('Overfit','cscv','n_bins=10','sharpe_ratio','PBO','PDes','SD','POvPNO')      ('Overfit','cscv_obj = cscv_obj(pf_locations or masked_pf_locations)','cscv_obj.n_bins = int(n_bins)','cscv_obj.objective = objective','cscv_obj.PBO = True','cscv_obj.PDes = True','cscv_obj.SD = True','cscv_obj.POvPNO = True','cscv_obj.run()')
             4    ('Data_Manager','split_into',2)                                               ('Data_Manager','data_manager_obj = Data_Manager_obj(df)','data_manager_obj.split_into = int(n_split)','data_manager_obj.run()')
             5    ('Filters','quick_filters','delete_loners')                                   ('Filters','filters_obj = Filters_obj(pf_locations or masked_pf_,locations)','filters_obj.quick_filters()','filters_obj.run()')
"""

ACTIONS_DICT = dict(
    Genie=dict(
        init='genie_obj = self.Genie_obj()',
        config_file_path='genie_obj.config_file_path = str(config_file_path)',
        n_trials='genie_obj.n_trials = int(n_trials)',
        refine=dict(
            # init='genie_obj.refine = True',
            init='genie_obj.refine()',
            strategy_seeds='genie_obj.refine.strategy_seeds = strategy_seeds',
            grid_n='genie_obj.refine.grid_n = int(grid_n)',
            product='genie_obj.refine.product = True',
            search_algorythm=dict()
        ),
        run_command='genie_obj.run()',
    ),
    MIP=dict(
        init='mip_obj = self.MIP_obj()',
        n_parameters='mip_obj.n_parameters = return_n_parameters',
        agg='mip_obj.agg = previous_mip_values',
        run_command='mip_obj.run()',
    ),
    Overfit=dict(
        init='overfit_obj = self.Overfit_obj()',
        cscv=dict(
            init='overfit_obj.cscv()',
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
        init='neighbors_obj = self.Neighbors_obj()',
        n_neighbors='neighbors_obj.n_neighbors = n_neighbors',
        computations_type='neighbors_obj.computations_type = computations_type',
        run_command='neighbors_obj.run()',
    ),
    Filters=dict(
        init='filters_obj = self.Filters_obj()',
        quick_filters='filters_obj.quick_filters = True',
        #
        delete_drawdown='filters_obj.delete_drawdown = drawdown_cmd',
        delete_profit='filters_obj.delete_profit = profit_cmd',
        delete_expectency='filters_obj.delete_expectency = expectency_cmd',
        delete_loners='filters_obj.delete_loners = True',
        # ...
        run_command='filters_obj.run()',
    ),
    Data_Manager=dict(
        init='data_manager_obj = self.Data_Manager_obj()',
        delete_first_month='data_manager_obj.delete_first_month = True',
        delete_last_month='data_manager_obj.delete_last_month = True',
        delete_max_drawdown_month='data_manager_obj.delete_max_drawdown_month = True',
        # ...
        n_split='data_manager_obj.n_split = int(n_split)',
        run_command='data_manager_obj.run()',
    ),
)

import ast
import inspect

import pandas as pd

import _typing as tp

Spaces_Program_Info = dict(
    Genie=dict(
        main_path="/home/ruben/PycharmProjects/mini_Genie/mini_genie_source/main_mini_genie.py",
        command_line_flags=dict(
            gp=True,
            c=False,
        )
    )
)


# todo
#   1. I can load the configuration file
#   2. Parse Dictionary
#   3. Make the nessesary changes to the keys and values
#   4. Make a temporary copy of the file
#   5. Pass the temporary file's path to mini_Genie

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

        import flatdict
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
                      "pf_params_and_metrics_df=self.pf_params_and_metrics_df," \
                      "previous_mip_values=self.previous_mip_values" \
                      ")"

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    @staticmethod
    class Genie_obj:
        def __init__(self):
            self.config_file_path = None
            self.n_trials = None

        def run(self):
            return self

        @staticmethod
        class refine:
            def __init__(self):
                self.strategy_seeds = None
                self.grid_n = None
                self.product = None
                self.search_algorythm = None

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
            self.n_bins = 10,
            self.objective = 'sortino',
            self.PBO = None,
            self.PDes = None,
            self.SD = None,
            self.POvPNO = None,

        def run(self):
            return self

    @staticmethod
    class Filters_obj:
        def __init__(self):
            self.quick_filters = None
            #
            self.delete_drawdown = None
            self.delete_profit = None
            self.delete_expectency = None
            self.delete_loners = None
            # ...

        def run(self):
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

    def parse(self):
        self.parse_input_dict()
        self.fill_output_df()
        return self.df

    @staticmethod
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
                    assert api_call_dot_split[-1].strip() == 'init'
                    split_cmd = cmd.split('=')
                    assert len(split_cmd) == 2

                    if index == 0:  # Initialization
                        hs_init_obj = split_cmd[0].strip()
                        icontext = split_cmd[1].strip()
                    else:
                        init_obj = split_cmd[0].strip()
                        assert init_obj.split('.')[0] == hs_init_obj
                        print(f'!!ERROR {split_cmd[1] = }!!')
                        exit()
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
        # print(context)
        # print(expression)
        # print(f'{exec(compile(expression, "file", "exec"), context) = }')
        # print()
        return self.multiline_eval(expr=expression, context=context)

    def run(self):
        Results = dict()
        for space_id, HS in enumerate(self.head_spaces_names, start=1):
            Results[HS] = self._run_space(space_id, HS)

            #
            # exit()
        #
        self.Results = Results
        for i, j in self.Results.items():
            print(f'{j.__dict__}')
        # print(f'{self.Results = }')

        return self.Results


EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        n_trials=500,
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
        n_split=2,
    ),
    Filters=dict(
        quick_filters=True,
        delete_loners=True,
    ),
)

api_handler = ApiHandler(EXAMPLE_INPUT_DICT)
api_handler.parse()
# print(api_handler.df[['Template_Code', 'Variable_Value']])
api_handler.run()
