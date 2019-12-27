# Author: Radu Grosu (wrote the original MATLAB source code)
# Author: Hannes Brantner (improved the translated Python script)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# here we choose the initial values for the trainable parameters
def values_for_initialization(init_id):
    if init_id == 1:
        init_param = np.random.rand(6, 1)
    elif init_id == 2:
        init_param = 200 * np.random.rand(6, 1)
    else:
        init_param = np.zeros([6, 1])
    return init_param


# implementation of the sigmoid function
def sigmoid(x, sigma, mu):
    return 1 / (1 + np.exp(-sigma * (x - mu)))


# function returns the learned parameters and the history of the two potentials of the neurons of the last training epoch
def train(initial_values):
    training_epochs = 10
    simulation_time = 2499
    learning_rate = 0.01

    # constant parameters of the network
    # neuron parameters
    c = 50
    gL = +1
    eL = -70
    # synaptic parameters
    eX = 0
    eI = -90
    s = 0.5
    m = -40

    # training variables
    g_input2x = np.zeros([simulation_time, 1])  # g_input2x = conductance of the synaptic connection between input and neuron x
    g_input2y = np.zeros([simulation_time, 1])
    g_x2x = np.zeros([simulation_time, 1])
    g_y2y = np.zeros([simulation_time, 1])
    g_y2x = np.zeros([simulation_time, 1])
    g_x2y = np.zeros([simulation_time, 1])

    g_input2x[0:simulation_time - 1] = initial_values[0]
    g_input2y[0:simulation_time - 1] = initial_values[1]
    g_x2x[0:simulation_time - 1] = initial_values[2]
    g_y2y[0:simulation_time - 1] = initial_values[3]
    g_y2x[0:simulation_time - 1] = initial_values[4]
    g_x2y[0:simulation_time - 1] = initial_values[5]

    # resting membrane potential of x and y
    neuron_x = np.zeros([simulation_time, 1])
    neuron_y = np.zeros([simulation_time, 1])
    neuron_x[0:simulation_time - 1] = -70
    neuron_y[0:simulation_time - 1] = -70

    # load the training data
    data = pd.read_csv("training_data.csv")
    x_train = np.array(data.iloc[:, 0:2])
    y_train = np.array(data.iloc[:, 2:4])

    # training loop
    for epoch in range(training_epochs):
        # update the parameters for the each training iteration
        g_input2x[0:simulation_time - 1] = g_input2x[simulation_time - 1]
        g_input2y[0:simulation_time - 1] = g_input2y[simulation_time - 1]
        g_x2x[0:simulation_time - 1] = g_x2x[simulation_time - 1]
        g_y2y[0:simulation_time - 1] = g_y2y[simulation_time - 1]
        g_y2x[0:simulation_time - 1] = g_y2x[simulation_time - 1]
        g_x2y[0:simulation_time - 1] = g_x2y[simulation_time - 1]

        # simulation loop
        for t in range(simulation_time - 1):
            # the two neurons x and y are simulated
            neuron_x[t + 1] = neuron_x[t] + (gL * (eL - neuron_x[t]) + g_input2x[t] * sigmoid(x_train[t, 0], s, m) * (eX - neuron_x[t]) + g_x2x[t] * sigmoid(neuron_x[t], s, m) * (eX - neuron_x[t]) + g_y2x[t] * sigmoid(neuron_y[t], s, m) * (eI - neuron_x[t])) / c
            neuron_y[t + 1] = neuron_y[t] + (gL * (eL - neuron_y[t]) + g_input2y[t] * sigmoid(x_train[t, 1], s, m) * (eX - neuron_y[t]) + g_y2y[t] * sigmoid(neuron_y[t], s, m) * (eX - neuron_y[t]) + g_x2y[t] * sigmoid(neuron_x[t], s, m) * (eI - neuron_y[t])) / c

            # training parameters are getting tuned by gradient decent
            g_input2x[t + 1] = g_input2x[t] - learning_rate * (neuron_x[t + 1] - y_train[t + 1, 0]) * sigmoid(x_train[t, 0], s, m) * (eX - neuron_x[t]) / c
            g_input2y[t + 1] = g_input2y[t] - learning_rate * (neuron_y[t + 1] - y_train[t + 1, 1]) * sigmoid(x_train[t, 1], s, m) * (eX - neuron_y[t]) / c
            g_x2x[t + 1] = g_x2x[t] - learning_rate * (neuron_x[t + 1] - y_train[t + 1, 0]) * sigmoid(neuron_x[t], s, m) * (eX - neuron_x[t]) / c
            g_y2y[t + 1] = g_y2y[t] - learning_rate * (neuron_y[t + 1] - y_train[t + 1, 1]) * sigmoid(neuron_y[t], s, m) * (eX - neuron_y[t]) / c
            g_y2x[t + 1] = g_y2x[t] - learning_rate * (neuron_x[t + 1] - y_train[t + 1, 0]) * sigmoid(neuron_y[t], s, m) * (eI - neuron_x[t]) / c
            g_x2y[t + 1] = g_x2y[t] - learning_rate * (neuron_y[t + 1] - y_train[t + 1, 1]) * sigmoid(neuron_x[t], s, m) * (eI - neuron_y[t]) / c

            # set all negative parameter values to zero
            g_input2x[t + 1] = max(g_input2x[t + 1], 0)
            g_input2y[t + 1] = max(g_input2y[t + 1], 0)
            g_x2x[t + 1] = max(g_x2x[t + 1], 0)
            g_y2x[t + 1] = max(g_y2x[t + 1], 0)
            g_x2y[t + 1] = max(g_x2y[t + 1], 0)
            g_y2y[t + 1] = max(g_y2y[t + 1], 0)

        print(g_input2x[simulation_time - 1], g_input2y[simulation_time - 1], g_x2x[simulation_time - 1], g_y2y[simulation_time - 1], g_y2x[simulation_time - 1], g_x2y[simulation_time - 1])

    return g_input2x[simulation_time - 1], g_input2y[simulation_time - 1], g_x2x[simulation_time - 1], g_y2y[simulation_time - 1], g_y2x[simulation_time - 1], g_x2y[simulation_time - 1], neuron_y, neuron_x


# train the network
init_values = values_for_initialization(2)
aa, bb, cc, dd, ee, ff, neuron_yy, neuron_xx = train(init_values)

# plot the history of both neuron potentials
plt.figure()
plt.plot(neuron_yy, 'b', label='neuron_y')
plt.plot(neuron_xx, 'r', label='neuron_x')
plt.legend(loc='upper right')
plt.show()
