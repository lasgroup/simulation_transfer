linestyle_tuple = {
    'loosely dotted': (0, (1, 10)),
    'dotted': (0, (1, 1)),
    'densely dotted': (0, (1, 1)),
    'long dash with offset': (5, (10, 3)),
    'loosely dashed': (0, (5, 10)),
    'dashed': (0, (5, 5)),
    'dashdot': 'dashdot',
    'densely dashed': (0, (5, 1)),
    'loosely dashdotted': (0, (3, 10, 1, 10)),
    'dashdotted': (0, (3, 5, 1, 5)),
    'densely dashdotted': (0, (3, 1, 1, 1)),
    'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
    'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
}

METHODS = ['SVGD',
           'FSVGD',
           'Sim Model',
           'Sampled Prior Functions',
           'NP',
           'PACOH',

           'FSVGD[SimPrior[LF]=GP]',
           'FSVGD[SimPrior[HF]=GP]',

           'FSVGD[SimPrior[HF]=KDE]',
           'FSVGD[SimPrior[HF]=Nu-Method]',
           'FSVGD[SimPrior[HF]=SSGE]',

           'FSVGD[SimPrior[LF + GP]=GP]',
           'FSVGD[SimPrior[HF + GP]=GP]',

           'GreyBox[LF] + FSVGD',
           'GreyBox[HF] + FSVGD',

           'SysID[LF]',
           'SysID[HF]',
           ]

LINE_STYLES = {'SVGD': linestyle_tuple['densely dashdotdotted'],
               'FSVGD': linestyle_tuple['dashdot'],
               'Sim Model': linestyle_tuple['densely dashdotted'],
               'NP': linestyle_tuple['dashed'],
               'PACOH': linestyle_tuple['loosely dashed'],

               'FSVGD[SimPrior[LF]=GP]': linestyle_tuple['dotted'],
               'FSVGD[SimPrior[HF]=GP]': linestyle_tuple['dashdotdotted'],

               'FSVGD[SimPrior[HF]=KDE]': linestyle_tuple['dotted'],
               'FSVGD[SimPrior[HF]=Nu-Method]': linestyle_tuple['densely dashed'],

               'FSVGD[SimPrior[HF + GP]=GP]': linestyle_tuple['dashed'],
               'FSVGD[SimPrior[LF + GP]=GP]': linestyle_tuple['densely dashed'],

               'GreyBox[LF] + FSVGD': linestyle_tuple['dashdotted'],
               'GreyBox[HF] + FSVGD': linestyle_tuple['densely dotted'],

               'SysID[LF]': linestyle_tuple['long dash with offset'],
               'SysID[HF]': linestyle_tuple['densely dashdotted'],
               }

COLORS = {'SVGD': '#228B22',
          'FSVGD': '#9BCD4C',
          'Sim Model': '#AF5227',
          'NP': '#654321',
          'PACOH': '#808080',

          'FSVGD[SimPrior[LF]=GP]': "#e6d800",
          'FSVGD[SimPrior[HF]=GP]': "#dc0ab4",

          'FSVGD[SimPrior[HF]=KDE]': '#daa520',
          'FSVGD[SimPrior[HF]=Nu-Method]': '#8486E0',

          'FSVGD[SimPrior[HF + GP]=GP]': '#e60049',
          'FSVGD[SimPrior[LF + GP]=GP]': "#0bb4ff",

          'GreyBox[LF] + FSVGD': "#9b19f5",
          'GreyBox[HF] + FSVGD': "#00bfa0",

          'SysID[HF]': "#ffa300",
          'SysID[LF]': "#b3d4ff",
          }

plot_num_data_name_transfer = {
    'BNN_FSVGD': METHODS[1],
    'BNN_FSVGD_SimPrior_gp': METHODS[7],
    'BNN_FSVGD_SimPrior_kde': METHODS[8],
    'BNN_FSVGD_SimPrior_nu-method': METHODS[9],
    'BNN_SVGD': METHODS[0],
    'NP': METHODS[4],
    'PACOH': METHODS[5]
}

plot_real_regression_method_name_transfer = {
    'BNN_FSVGD_SimPrior_hf_gp': METHODS[12],
    'BNN_FSVGD': METHODS[1],
    'BNN_FSVGD_SimPrior_gp': METHODS[11],
    'BNN_FSVGD_SimPrior_gp_no_add_gp': METHODS[6],
    'GreyBox': METHODS[13],
    'SysID_hf': METHODS[16],
    'BNN_FSVGD_SimPrior_hf_gp_no_add_gp': METHODS[7],
    'BNN_SVGD': METHODS[0],
    'SysID': METHODS[15],
    'GreyBox_hf': METHODS[14],
}

plot_num_data_data_source_transfer = {
    'pendulum': 'Pendulum[HF simulator]',
    'racecar': 'Racecar[HF simulator]'
}

plot_num_data_metrics_transfer = {
    'nll': 'NLL',
    'rmse': 'RMSE'
}

plot_real_regression_data_source_transfer = {
    'real_racecar_v3': 'Racecar[Hardware]',
    'racecar_hf': 'Racecar[HF simulator]',
    'pendulum_hf': 'Pendulum[HF simulator]',
}

plot_real_regression_data_metrics_transfer = {
    'nll': 'NLL',
}

online_rl_name_transfer = {
    'No sim prior': METHODS[1],
    'Low fidelity prior': METHODS[11]
}

TRUE_FUNCTION_COLOR = 'black'
TRUE_FUNCTION_LINE_STYLE = linestyle_tuple['densely dashed']
TRUE_FUNCTION_LINE_WIDTH = 4

MEAN_FUNCTION_COLOR = 'blue'
MEAN_FUNCTION_LINE_STYLE = linestyle_tuple['densely dashdotted']
CONFIDENCE_ALPHA = 0.2
MEAN_FUNCTION_LINE_WIDTH = 4

SAMPLES_COLOR = 'green'
SAMPLES_LINE_STYLE = linestyle_tuple['densely dashdotdotted']
SAMPLES_ALPHA = 0.6
SAMPLES_LINE_WIDTH = 3

OBSERVATIONS_COLOR = 'RED'
OBSERVATIONS_LINE_WIDTH = 4

ONLINE_RL_LINEWIDTH = 4
