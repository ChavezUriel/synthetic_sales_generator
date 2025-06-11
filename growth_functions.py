# %%
# growth_functions.py
"""
Defines various growth functions that can be used in the simulation.
Each function takes time 't' (in days) and parameters 'params' as input.
"""
import numpy as np

def constant_value(t, params):
    """ Returns a constant value. Params needs 'value'. """
    value = params.get('value', 0)
    noise = params.get('noise', 0.05 * abs(value))  # 5% of value by default
    poly_noise_factor = params.get('poly_noise_factor', 0.01)
    poly_noise_scale = params.get('poly_noise', poly_noise_factor * abs(value))
    shape = np.shape(t)
    result = value + np.random.normal(0, noise, shape)
    poly_coeffs = np.random.uniform(-poly_noise_scale, poly_noise_scale, 4)
    t_norm = np.linspace(-1, 1, shape[0]) if shape else np.array([-1])
    poly = np.polyval(poly_coeffs, t_norm)
    result = result + poly
    return np.maximum(result, 0)

def constant_growth(t, params):
    """ Simple constant rate. Params needs 'initial', 'rate_per_day'. """
    initial = params.get('initial', 0)
    rate_per_day = params.get('rate_per_day', 0)
    noise = params.get('noise', 0.05 * max(abs(initial), abs(rate_per_day)))
    poly_noise_factor = params.get('poly_noise_factor', 0.01)
    poly_noise_scale = params.get('poly_noise', poly_noise_factor * max(abs(initial), abs(rate_per_day)))
    shape = np.shape(t)
    result = initial + (rate_per_day * t)
    result = result + np.random.normal(0, noise, shape)
    poly_coeffs = np.random.uniform(-poly_noise_scale, poly_noise_scale, 4)
    t_norm = np.linspace(-1, 1, shape[0]) if shape else np.array([-1])
    poly = np.polyval(poly_coeffs, t_norm)
    result = result + poly
    return np.maximum(result, 0)

def linear_growth(t, params):
    """ Linear growth rate. Params needs 'initial', 'rate_per_year'. """
    initial = params.get('initial', 0)
    rate_per_year = params.get('rate_per_year', 0)
    rate_per_day = rate_per_year / 365.25
    noise = params.get('noise', 0.05 * max(abs(initial), abs(rate_per_year)))
    poly_noise_factor = params.get('poly_noise_factor', 0.01)
    poly_noise_scale = params.get('poly_noise', poly_noise_factor * max(abs(initial), abs(rate_per_year)))
    shape = np.shape(t)
    result = initial + (rate_per_day * t)
    result = result + np.random.normal(0, noise, shape)
    poly_coeffs = np.random.uniform(-poly_noise_scale, poly_noise_scale, 4)
    t_norm = np.linspace(-1, 1, shape[0]) if shape else np.array([-1])
    poly = np.polyval(poly_coeffs, t_norm)
    result = result + poly
    return np.maximum(result, 0)

def exponential_growth(t, params):
    """ Exponential growth. Params needs 'initial', 'rate_per_day'. """
    initial = params.get('initial', 0)
    rate_per_day_factor = params.get('rate_per_day', 0)
    effective_rate = 1.0 + rate_per_day_factor
    noise = params.get('noise', 0.05 * abs(initial))
    poly_noise_factor = params.get('poly_noise_factor', 0.01)
    poly_noise_scale = params.get('poly_noise', poly_noise_factor * abs(initial))
    shape = np.shape(t)
    result = initial * (effective_rate ** t)
    result = result + np.random.normal(0, noise, shape)
    poly_coeffs = np.random.uniform(-poly_noise_scale, poly_noise_scale, 4)
    t_norm = np.linspace(-1, 1, shape[0]) if shape else np.array([-1])
    poly = np.polyval(poly_coeffs, t_norm)
    result = result + poly
    return np.maximum(result, 0)

def piecewise_growth(t, params):
    """
    Piecewise growth function with noise.
    Params:
        'breakpoints': List of time points where the growth rate changes.
        'rates': List of growth rates corresponding to each interval.
        'initial': Initial value at the start of the time series.
        'noise': Optional Gaussian noise added to the growth curve.
    """
    breakpoints = params.get('breakpoints', [])
    rates = params.get('rates', [])
    initial = params.get('initial', 0)
    noise = params.get('noise', 0.05 * max(abs(initial), max([abs(r) for r in rates] + [0])))
    poly_noise_factor = params.get('poly_noise_factor', 0.01)
    poly_noise_scale = params.get('poly_noise', poly_noise_factor * max(abs(initial), max([abs(r) for r in rates] + [0])))
    
    # Ensure breakpoints and rates are of the same length
    if len(breakpoints) != len(rates):
        raise ValueError("Length of 'breakpoints' and 'rates' must be the same.")
    
    # Calculate the current value based on the piecewise function
    current_value = initial
    for i in range(len(breakpoints) - 1):
        t_start, t_end = breakpoints[i], breakpoints[i + 1]
        rate = rates[i]
        mask = (t >= t_start) & (t < t_end)
        current_value = np.where(mask, current_value + rate * (t - t_start), current_value)
    
    shape = np.shape(t)
    result = current_value
    result = result + np.random.normal(0, noise, shape)
    poly_coeffs = np.random.uniform(-poly_noise_scale, poly_noise_scale, 4)
    t_norm = np.linspace(-1, 1, shape[0]) if shape else np.array([-1])
    poly = np.polyval(poly_coeffs, t_norm)
    result = result + poly
    return np.maximum(result, 0)

def logarithmic_growth(t, params):
    """
    Logarithmic growth function with noise.
    Params:
        'value_after_one_year': The value at t=365 (e.g., after one year).
        'growth_speed': A factor that influences the growth rate.
        'noise': Optional Gaussian noise added to the growth curve.
    """
    year_days = 365
    multiplier = params.get('value_after_one_year', 1)
    growth_speed = params.get('growth_speed', 1)  # Not used in this function, but can be used for scaling

    absolute_growth_speed = max(1, multiplier * growth_speed)

    base = 1 + year_days / absolute_growth_speed


    noise = params.get('noise', 0.05 * abs(multiplier))
    poly_noise_factor = params.get('poly_noise_factor', 0.01)
    poly_noise_scale = params.get('poly_noise', poly_noise_factor * abs(multiplier))
    shape = np.shape(t)
    result = multiplier * np.emath.logn(base, t / absolute_growth_speed  + 1)
    result = result + np.random.normal(0, noise, shape)
    poly_coeffs = np.random.uniform(-poly_noise_scale, poly_noise_scale, 4)
    t_norm = np.linspace(-1, 1, shape[0]) if shape else np.array([-1])
    poly = np.polyval(poly_coeffs, t_norm)
    result = result + poly
    return np.maximum(result, 0)

def sigmoid_growth(t, params):
    """
    Sigmoid growth with configurable starting percentage, matching the provided equation.
    Params:
        'max_value' (mv): The maximum value reached as t increases (e.g., market capacity or max sales).
        'growth_rate' (gr): Controls how quickly the curve rises (higher = faster growth).
        'pc_start' (pcs): The starting percentage of max_value at t=0 (0 < pcs < 1).
    Implements:
        f(x) = mv / (1 + exp(-gr * (x - ln(1/pcs - 1)/gr))) - pcs * mv
    """
    mv = params.get('max_value', 1)
    gr = params.get('growth_rate', 1)
    pcs = params.get('pc_start', 0.01)  # Should be between 0 and 1
    # Avoid division by zero or log of zero
    pcs = np.clip(pcs, 1e-6, 1-1e-6)
    shift = np.log(1/pcs - 1) / gr
    noise = params.get('noise', 0.05 * abs(mv))
    poly_noise_factor = params.get('poly_noise_factor', 0.01)
    poly_noise_scale = params.get('poly_noise', poly_noise_factor * abs(mv))
    shape = np.shape(t)
    result = mv / (1 + np.exp(-gr * (t - shift))) - pcs * mv
    result = result + np.random.normal(0, noise, shape)
    poly_coeffs = np.random.uniform(-poly_noise_scale, poly_noise_scale, 4)
    t_norm = np.linspace(-1, 1, shape[0]) if shape else np.array([-1])
    poly = np.polyval(poly_coeffs, t_norm)
    result = result + poly
    return np.maximum(result, 0)

# Add more functions as needed (e.g., seasonal, specific arrays)



# %%
import numpy as np

# Test and plot all growth functions

# def test_growth_functions():
#     t = np.arange(0, 1000)  # Example time range

#     # Define parameters for each growth function
#     params_constant = {'value': 100, 'noise': 5}
#     params_constant_growth = {'initial': 50, 'rate_per_day': 0.1, 'noise': 2}
#     params_linear_growth = {'initial': 20, 'rate_per_year': 10, 'noise': 3}
#     params_exponential_growth = {'initial': 10, 'rate_per_day': 0.05, 'noise': 1}
#     params_logarithmic_growth = {
#         'value_after_one_year': 100,
#         'growth_speed': 0.01,
#         'noise': 5,
#         'poly_noise_factor': 0.5,
#     }
#     params_sigmoid_growth = {
#         'max_value': 200,
#         'growth_rate': 0.05,
#         'pc_start': 0.01,
#         'noise': 5,
#         'poly_noise_factor': 0.5,
#     }

#     # Calculate growth values
#     constant_value_array = constant_value(t, params_constant)
#     constant_growth_array = constant_growth(t, params_constant_growth)
#     linear_growth_array = linear_growth(t, params_linear_growth)
#     exponential_growth_array = exponential_growth(t, params_exponential_growth)
#     logarithmic_growth_array = logarithmic_growth(t, params_logarithmic_growth)
#     sigmoid_growth_array = sigmoid_growth(t, params_sigmoid_growth)

#     return (t,
#             constant_value_array,
#             constant_growth_array,
#             linear_growth_array,
#             exponential_growth_array,
#             logarithmic_growth_array,
#             sigmoid_growth_array)

# # Run the test and get the growth arrays
# t, constant_value_array, constant_growth_array, linear_growth_array, exponential_growth_array, logarithmic_growth_array, sigmoid_growth_array = test_growth_functions()

# # Plot the results
# import matplotlib.pyplot as plt
# plt.figure(figsize=(16, 12))

# names = [
#     ('Constant Value', constant_value_array),
#     ('Constant Growth', constant_growth_array),
#     ('Linear Growth', linear_growth_array),
#     ('Exponential Growth', exponential_growth_array),
#     ('Logarithmic Growth', logarithmic_growth_array),
#     ('Sigmoid Growth', sigmoid_growth_array)
# ]

# for i, (name, arr) in enumerate(names, 1):
#     plt.subplot(3, 2, i)
#     plt.plot(t, arr, label=name)
#     plt.title(name)
#     plt.xlabel('Time (days)')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid()

# plt.tight_layout()
# plt.show()
# # %%
