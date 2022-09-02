ACTIONS_DICT = dict(
    Genie=dict(
        init='genie_obj = self.Genie_obj()',
        run_mode='genie_obj.run_mode = str(run_mode)',
        data_files_names='genie_obj.data_files_names = [str(i) for i in data_files_names]',
        tick_size='genie_obj.tick_size = [float(i) for i in tick_size]',
        study_name='genie_obj.study_name = str(study_name)',
        # start_date='genie_obj.start_date = datetime.datetime(**start_date)',
        start_date='genie_obj.start_date = start_date',
        # end_date='genie_obj.end_date = datetime.datetime(**end_date)',
        end_date='genie_obj.end_date = end_date',
        # timer_limit='genie_obj.timer_limit = datetime.datetime(timer_limit)',
        timer_limit='genie_obj.timer_limit = timer_limit',
        Continue='genie_obj.Continue = bool(Continue)',
        batch_size='genie_obj.batch_size = int(batch_size)',
        max_initial_combinations='genie_obj.max_initial_combinations = int(max_initial_combinations)',
        stop_after_n_epoch='genie_obj.stop_after_n_epoch = int(stop_after_n_epoch)',
        Strategy='genie_obj.Strategy = str(Strategy)',
        max_spread_allowed='genie_obj.max_spread_allowed = int(max_spread_allowed)',
        trading_fees='genie_obj.trading_fees = float(trading_fees)',  # 0.00005 or 0.005%, $5 per $100_000
        max_orders='genie_obj.max_orders = int(max_orders)',
        init_cash='genie_obj.init_cash = float(init_cash)',
        size='genie_obj.size = float(size)',
        #
        refine=dict(
            # init='genie_obj.refine()',
            init='genie_obj.refine = self.dotdict()',
            strategy_seeds='genie_obj.refine.strategy_seeds = strategy_seeds',
            grid_n='genie_obj.refine.grid_n = int(grid_n)',
            product='genie_obj.refine.product = bool(product)',
            search_algorythm=dict()
        ),
        run_command='genie_obj.run()',
    ),
    # MIP=dict(
    #     init='mip_obj = self.MIP_obj()',
    #     n_parameters='mip_obj.n_parameters = return_n_parameters',
    #     agg='mip_obj.agg = previous_mip_values',
    #     run_command='mip_obj.run()',
    # ),
    # Overfit=dict(
    #     init='overfit_obj = self.Overfit_obj()',
    #     cscv=dict(
    #         init='overfit_obj.cscv()',
    #         n_bins='overfit_obj.n_bins = int(n_bins)',
    #         objective='overfit_obj.objective = objective',
    #         PBO='overfit_obj.PBO = bool(PBO)',
    #         PDes='overfit_obj.PDes = bool(PDes)',
    #         SD='overfit_obj.SD = bool(SD)',
    #         POvPNO='overfit_obj.POvPNO = bool(POvPNO)',
    #     ),
    #     run_command='overfit_obj.run()'
    # ),
    # Neighbors=dict(
    #     init='neighbors_obj = self.Neighbors_obj()',
    #     n_neighbors='neighbors_obj.n_neighbors = n_neighbors',
    #     computations_type='neighbors_obj.computations_type = computations_type',
    #     run_command='neighbors_obj.run()',
    # ),
    Filters=dict(
        init='filters_obj = self.Filters_obj()',
        study_name='filters_obj.study_name = str(study_name)',
        output_path='filters_obj.output_path = str(output_path)',
        #
        Min_total_trades='filters_obj.quick_filters = int(quick_filters)',
        Profit_factor='filters_obj.Profit_factor = float(Profit_factor)',
        Expectancy='filters_obj.Expectancy = float(Expectancy)',
        Daily_drawdown='filters_obj.Daily_drawdown = float(Daily_drawdown)',
        Total_drawdown='filters_obj.Total_drawdown = float(Total_drawdown)',
        Profit_for_month='filters_obj.Profit_for_month = float(Profit_for_month)',
        Total_Win_Rate='filters_obj.Total_Win_Rate = float(Total_Win_Rate)',
        quick_filters='filters_obj.quick_filters = bool(quick_filters)',

        # delete_drawdown='filters_obj.delete_drawdown = drawdown_cmd',
        # delete_profit='filters_obj.delete_profit = profit_cmd',
        # delete_expectency='filters_obj.delete_expectency = expectency_cmd',
        # delete_loners='filters_obj.delete_loners = bool(delete_loners)',
        # ...
        run_command='filters_obj.run()',
    ),
    # Data_Manager=dict(
    #     init='data_manager_obj = self.Data_Manager_obj()',
    #     delete_first_month='data_manager_obj.delete_first_month = bool(delete_first_month)',
    #     delete_last_month='data_manager_obj.delete_last_month = bool(delete_last_month)',
    #     delete_max_drawdown_month='data_manager_obj.delete_max_drawdown_month = bool(delete_max_drawdown_month)',
    #     # ...
    #     n_split='data_manager_obj.n_split = int(n_split)',
    #     run_command='data_manager_obj.run()',
    # ),
)
