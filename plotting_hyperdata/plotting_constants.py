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
           'GreyBox + FSVGD',
           'Sim Model',
           'FSVGD[SimPrior=GP]',
           'FSVGD[SimPrior=KDE]',
           'FSVGD[SimPrior=Nu-Method]',
           'FSVGD[SimPrior=SSGE]',
           'Sampled Prior Functions',
           'NP',
           'PACOH',

           'FSVGD[SimPrior[HF + GP]=GP]',
           'FSVGD[SimPrior[LF + GP]=GP]',
           'FSVGD[SimPrior[LF]=GP]',
           'GreyBox[LF] + FSVGD',
           'SysID[HF]',
           'FSVGD[SimPrior[HF]=GP]',
           'SysID[LF]',
           'GreyBox[HF] + FSVGD',
           ]

plot_real_regression_method_name_transfer = {
    'BNN_FSVGD_SimPrior_hf_gp': METHODS[11],
     'BNN_FSVGD': METHODS[1],
     'BNN_FSVGD_SimPrior_gp': METHODS[12],
     'BNN_FSVGD_SimPrior_gp_no_add_gp': METHODS[13],
     'GreyBox': METHODS[14],
     'SysID_hf': METHODS[15],
     'BNN_FSVGD_SimPrior_hf_gp_no_add_gp': METHODS[16],
     'BNN_SVGD': METHODS[0],
     'SysID': METHODS[17],
     'GreyBox_hf': METHODS[18],
}

COLORS = {'SVGD': '#228B22',
          'FSVGD': '#9BCD4C',
          'GreyBox + FSVGD': '#CB3E3E',
          'Sim Model': '#AF5227',
          'FSVGD[SimPrior=GP]': '#DC6A73',
          'FSVGD[SimPrior=KDE]': '#daa520',
          'FSVGD[SimPrior=Nu-Method]': '#8486E0',
          'NP': '#654321',
          'PACOH': '#808080',

          'FSVGD[SimPrior[HF + GP]=GP]': '#e60049',
          'FSVGD[SimPrior[LF + GP]=GP]': "#0bb4ff",
          'FSVGD[SimPrior[LF]=GP]': "#e6d800",
          'GreyBox[LF] + FSVGD': "#9b19f5",
          'SysID[HF]': "#ffa300",
          'FSVGD[SimPrior[HF]=GP]': "#dc0ab4",
          'SysID[LF]': "#b3d4ff",
          'GreyBox[HF] + FSVGD': "#00bfa0",
          }

LINE_STYLES = {'SVGD': linestyle_tuple['densely dashdotdotted'],
               'FSVGD': linestyle_tuple['dashdot'],
               'GreyBox + FSVGD': linestyle_tuple['dashdotdotted'],
               'Sim Model': linestyle_tuple['densely dashdotted'],
               'FSVGD[SimPrior=GP]': linestyle_tuple['dashdotted'],
               'FSVGD[SimPrior=KDE]': linestyle_tuple['dotted'],
               'FSVGD[SimPrior=Nu-Method]': linestyle_tuple['densely dashed'],
               'NP': linestyle_tuple['dashed'],
               'PACOH': linestyle_tuple['loosely dashed'],

               'FSVGD[SimPrior[HF + GP]=GP]': linestyle_tuple['dashed'],
               'FSVGD[SimPrior[LF + GP]=GP]': linestyle_tuple['densely dashed'],
               'FSVGD[SimPrior[LF]=GP]': linestyle_tuple['dotted'],
               'GreyBox[LF] + FSVGD': linestyle_tuple['dashdotted'],
               'SysID[HF]': linestyle_tuple['densely dashdotted'],
               'FSVGD[SimPrior[HF]=GP]': linestyle_tuple['dashdotdotted'],
               'SysID[LF]': linestyle_tuple['long dash with offset'],
               'GreyBox[HF] + FSVGD': linestyle_tuple['densely dotted'],
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

plot_num_data_name_transfer = {
    'BNN_FSVGD': METHODS[1],
    'BNN_FSVGD_SimPrior_gp': METHODS[4],
    'BNN_FSVGD_SimPrior_kde': METHODS[5],
    'BNN_FSVGD_SimPrior_nu-method': METHODS[6],
    'BNN_SVGD': METHODS[0],
    'NP': METHODS[9],
    'PACOH': METHODS[10]
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
