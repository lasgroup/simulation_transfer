linestyle_tuple = {
    'loosely dotted': (0, (1, 10)),
    'dotted': (0, (1, 1)),
    'densely dotted': (0, (1, 1)),
    'long dash with offset': (5, (10, 3)),
    'loosely dashed': (0, (5, 10)),
    'dashed': (0, (5, 5)),
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
           'GreyBox',
           'Sim Model',
           'FSVGD[SimPrior=GP]',
           'FSVGD[SimPrior=KDE]',
           'FSVGD[SimPrior=Nu-Method]',
           'FSVGD[SimPrior=SSGE]',
           'Sampled Prior Functions'
           ]

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
