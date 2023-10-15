import pymannkendall as mk
from statsmodels.tsa.seasonal import seasonal_decompose

# hardcoding number of datapoints (=complete transitions) required for signal decomposition analysis to 15
num_transitions = 15
p_tresh = 0.00001

# slope thresholds defined in absolute values
driveTime_slope_floor = 0.5
other_slope_floor = 1.0
severity_threshold = 0.95
severity_threshold_mild = 0.60
confidence_threshold = 96
driveTime_severity_threshold = 0.90

def decompose_signal(series):
    result = seasonal_decompose(series, model='additive', period=num_transitions)
    return result.trend, result.seasonal

def mann_kendall_test(series):
    return mk.original_test(series)

def gen_trend_statistics(df, m):  # m is the column name in the df
    list_dict = []
    try:
        trend_out, seasonal_out = decompose_signal(df[m])
        mk_result = mann_kendall_test(trend_out)
        mk_result_og = mann_kendall_test(df[m])
        cause = None
    except Exception as e:
        mk_result = None
        cause = e
    list_dict.append({
    'trend': mk_result.trend if mk_result is not None else cause,
    'h': mk_result.h if mk_result is not None else None,
    'p': mk_result.p if mk_result is not None else None,
    'z': mk_result.z if mk_result is not None else None,
    'Tau': mk_result.Tau if mk_result is not None else None,
    's': mk_result.s if mk_result is not None else None,
    'var_s': mk_result.var_s if mk_result is not None else None,
    'slope': mk_result.slope if mk_result is not None else None,
    'intercept': mk_result.intercept if mk_result is not None else None,
    'og_slope' : mk_result_og.slope if mk_result is not None else None
    })
    return list_dict
        
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def calculate_confidence(v):
    # slope of original signal vs slope of extracted trend
    confidence_input = abs((v['og_slope'] - v['slope'])/v['og_slope'])
    confidence_score = sigmoid_derivative(confidence_input)*4*100

def calculate_severity():
    severity_score = sigmoid(abs(v['slope']))