"""
Design an api handler class for a python 3 project. The class needs to accept a dictionary of known keys, parse each string according to a set of rules and link the parsed strings to a known corresponding function_eval_string found in ACTIONS_DICT, then return a dataframe with columns ["id","parsed_string","functions_to_evaluate"].
The each element of parsed strings is split in a tuple ranked based on the the order that they are found in the ACTIONS_DICT for that element's Class keyname. You can make custom rules for each key if they do not all meet a certain pattern
EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        n_trials=500,
        config_file_path='config_file_path.config_dict_name',
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
             1    ('Overfit','cscv','n_bins=10','sharpe_ratio','PBO','PDes','SD','POvPNO')      ('Overfit','cscv_obj = cscv_obj(pf_loactions or masked_pf_loactions)','cscv_obj.n_bins = int(n_bins)','cscv_obj.objective = objective','cscv_obj.PBO = True','cscv_obj.PDes = True','cscv_obj.SD = True','cscv_obj.POvPNO = True','cscv_obj.run()')
             4    ('Data_Manager','split_into',2)                                               ('Data_Manager','data_manager_obj = Data_Manager_obj(df)','data_manager_obj.split_into = int(n_split)','data_manager_obj.run()')
             5    ('Filters','quick_filters','delete_loners')                                   ('Filters','filters_obj = Filters_obj(pf_loactions or masked_pf_loactions)','filters_obj.quick_filters()','filters_obj.run()')
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
        init='overfit_obj = Overfit_obj(pf_loactions or masked_pf_loactions)',
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
        init='filters_obj = Filters_obj(pf_loactions or masked_pf_loactions)',
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
    def __init__(self, input_dict):
        self.input_dict = input_dict
        self.parsed_strings = []
        self.functions_to_evaluate = []
        self.id = 0
        self.df = pd.DataFrame(columns=["id", "parsed_string", "functions_to_evaluate"])

        exit()

    def parse_input_dict_(self, sub_dict):
        for key, value in sub_dict.items():
            if isinstance(value, dict):
                self.parse_input_dict(value)
            else:
                self.parse_string(key, value)

    def parse_input_dict(self):
        for key, value in self.input_dict.items():
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
        self.create_df()
        return self.df


EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        n_trials=500,
        config_file_path='config_file_path.config_dict_name',
        refine=dict(
            max_variance=5,
        ),
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
    # Data_Manager_1=dict(
    #     split_into=2,
    # ),
    # Filters=dict(
    #     quick_filters=True,
    #     delete_loners=True,
    # ),
)

api_handler = ApiHandler(EXAMPLE_INPUT_DICT)
df = api_handler.run()
print(df)
