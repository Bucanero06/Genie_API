config_template = """dict(
    study_path = this_elsethis(study_path, None),
    output_path = this_elsethis(output_path, return_unique_name_for_path(default_output_path)),
    Min_total_trades = this_elsethis(Min_total_trades,int(1)),
    Profit_factor = this_elsethis(Profit_factor,-np.inf),
    Expectancy = this_elsethis(Expectancy,-np.inf),
    Daily_drawdown = this_elsethis(Daily_drawdown,-np.inf),
    Total_drawdown = this_elsethis(Total_drawdown,-np.inf),
    Profit_for_month = this_elsethis(Profit_for_month,-np.inf),
    Total_Win_Rate = this_elsethis(Total_Win_Rate,-np.inf),
    #
    quick_filters = this_elsethis(quick_filters,False),
)"""
