# The proposed numerical calculation of the solar system optimum generation capacity in Python. The input variables, including:
# First hour period [hour]
# Distance to solar noon [rad]
# Temperature [ºC]
# Wind direction  [ º]
# Wind speed [m/s]
# Sky cover [over 4]
# Visibility [km]
# Humidity [%]
# Average wind speed [m/s]
# Average pressure [inHg]
# The neural network model designed leverages deep architectures. It consists of five layers: a scaling layer with 8 neurons, two perceptron layers with 7 and 1 neuron, respectively, an unscaling layer with 1 neuron, and a bounding layer with 1 neuron.
# The optimal power generation is calculated and displayed based on the provided input environmental conditions, resulting in a power generation of at least 2.5 kJ.

import numpy as np


def scale_inputs(FHP, DSN, TMP, WND, WSP, SKC, VSB, HMD, AWS, APR):
    scaled_FHP = (FHP - 11.49279976) / 6.871900082
    scaled_DSN = (DSN - 0.5028570294) / 0.2976570129
    scaled_TMP = (TMP - 58.47909927) / 6.831540108
    scaled_WND = (WND - 24.95890045) / 6.907269955
    scaled_WSP = (WSP - 10.09780025) / 4.839719772
    scaled_SKC = (SKC - 1.988350034) / 1.411980033
    scaled_VSB = (VSB - 9.55739975) / 1.384310007
    scaled_HMD = (HMD - 73.51229858) / 15.08090019
    scaled_AWS = (AWS - 10.12919998) / 7.262030125
    scaled_APR = (APR - 30.01779938) / 0.1420429945
    return np.array(
        [
            scaled_FHP,
            scaled_DSN,
            scaled_TMP,
            scaled_WND,
            scaled_WSP,
            scaled_SKC,
            scaled_VSB,
            scaled_HMD,
            scaled_AWS,
            scaled_APR,
        ]
    )


def perceptron_layer_1(scaled_inputs, biases, weights):
    outputs = np.tanh(biases + np.dot(scaled_inputs, weights))
    return outputs


def perceptron_layer_2(outputs, weights):
    output = -0.0387 + np.dot(outputs, weights)
    return output


def unscale_output(output):
    unscaled_output = output * 10314.2998 + 6984.629883
    return unscaled_output


# Inputs
FHP, DSN, TMP, WND, WSP, SKC, VSB, HMD, AWS, APR = (
    11.49279976,
    0.5028570294,
    58.47909927,
    24.95890045,
    10.09780025,
    1.988350034,
    9.55739975,
    73.51229858,
    10.12919998,
    30.01779938,
)

# Biases and weights for perceptron layer 1 and 2
biases_1 = np.array(
    [-0.807717, 2.04105, -1.80318, -1.17135, -1.20332, 0.626849, -1.18963]
)
weights_1 = np.array(
    [
        [0.559397, -0.822155, 0.244786, -0.354197, -0.23055, 0.165954, 0.450259],
        [0.647947, 2.43219, -1.04416, -0.656248, -0.960755, 1.32087, -1.29097],
        [0.0276576, 0.0590992, -0.00892525, -0.102843, -0.18786, 0.182393, 0.144524],
        [0.47716, -0.0651514, 0.390159, 0.0811014, -0.0522886, 0.0098501, 0.134908],
        [0.353094, 0.134489, 0.323759, -0.0938557, -0.146358, 0.139196, 0.0916994],
        [-0.362904, 0.152256, -0.00599757, 0.156897, 0.205069, -0.198977, 0.305597],
        [-0.211327, -0.0255309, 0.0121547, -0.0791749, 0.107659, -0.0286878, -0.026318],
        [-0.447337, 0.201222, -0.32894, -0.300986, 0.592403, -0.305583, -0.0112834],
        [-0.353499, 0.137871, 0.152083, 0.155358, -0.161954, 0.0343114, 0.677691],
        [-0.496944, -0.304734, 0.00545484, 0.148069, -0.262045, 0.264277, -0.216455],
    ]
)
weights_2 = np.array(
    [-0.107249, -0.791044, 1.09248, -0.906219, -1.00434, -0.969682, -0.347533]
)

# Energy production approximation
scaled_inputs = scale_inputs(FHP, DSN, TMP, WND, WSP, SKC, VSB, HMD, AWS, APR)
outputs_1 = perceptron_layer_1(scaled_inputs, biases_1, weights_1)
output_2 = perceptron_layer_2(outputs_1, weights_2)
unscaled_output = unscale_output(output_2)

print("Energy production [J]:", unscaled_output)
