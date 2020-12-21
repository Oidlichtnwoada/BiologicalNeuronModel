# Author: Radu Grosu (wrote the original MATLAB source code)
# Author: Hannes Brantner (improved the translated Python script)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# here we choose the initial values for the trainable parameters
def values_for_initialization(init_id):
    if init_id == 0:
        init_param = [1.28, 1.28, 0.025, 0.025, 0.16, 0.16]
    elif init_id == 1:
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
    training_epochs = 1
    simulation_time = 2499
    learning_rate = 0

    # neuron parameters
    c = 1
    gL = 0.025
    eL = 0
    # synaptic parameters
    eX = 2
    eI = 0
    s = 100
    m = 0.5

    # training variables
    g_input2x = initial_values[0]
    g_input2y = initial_values[1]
    g_x2x = initial_values[2]
    g_y2y = initial_values[3]
    g_y2x = initial_values[4]
    g_x2y = initial_values[5]

    # resting membrane potential of x and y at the beginning
    neuron_x = np.ones([simulation_time, 1]) * eL
    neuron_y = np.ones([simulation_time, 1]) * (2 * g_y2y / (gL + g_y2y))

    # load the training data
    data = pd.read_csv("training_data.csv")
    x_train = np.array(data.iloc[:, 0:2])
    y_train = np.array(data.iloc[:, 2:4])

    # training loop
    for _ in range(training_epochs):

        # simulation loop
        for t in range(simulation_time - 1):
            # the two neurons x and y are simulated
            neuron_x[t + 1] = neuron_x[t] + (
                    gL * (eL - neuron_x[t]) + g_input2x * sigmoid(x_train[t, 0], s, m) * (eX - neuron_x[t]) + g_x2x * sigmoid(neuron_x[t], s, m) * (eX - neuron_x[t]) + g_y2x * sigmoid(neuron_y[t],
                                                                                                                                                                                        s, m) * (
                            eI - neuron_x[t])) / c
            neuron_y[t + 1] = neuron_y[t] + (
                    gL * (eL - neuron_y[t]) + g_input2y * sigmoid(x_train[t, 1], s, m) * (eX - neuron_y[t]) + g_y2y * sigmoid(neuron_y[t], s, m) * (eX - neuron_y[t]) + g_x2y * sigmoid(neuron_x[t],
                                                                                                                                                                                        s, m) * (
                            eI - neuron_y[t])) / c

            # training parameters are getting tuned by gradient decent
            g_input2x -= learning_rate * (neuron_x[t + 1] - y_train[t + 1, 0]) * sigmoid(x_train[t, 0], s, m) * (eX - neuron_x[t]) / c
            g_input2y -= learning_rate * (neuron_y[t + 1] - y_train[t + 1, 1]) * sigmoid(x_train[t, 1], s, m) * (eX - neuron_y[t]) / c
            g_x2x -= learning_rate * (neuron_x[t + 1] - y_train[t + 1, 0]) * sigmoid(neuron_x[t], s, m) * (eX - neuron_x[t]) / c
            g_y2y -= learning_rate * (neuron_y[t + 1] - y_train[t + 1, 1]) * sigmoid(neuron_y[t], s, m) * (eX - neuron_y[t]) / c
            g_y2x -= learning_rate * (neuron_x[t + 1] - y_train[t + 1, 0]) * sigmoid(neuron_y[t], s, m) * (eI - neuron_x[t]) / c
            g_x2y -= learning_rate * (neuron_y[t + 1] - y_train[t + 1, 1]) * sigmoid(neuron_x[t], s, m) * (eI - neuron_y[t]) / c

            # set all negative parameter values to zero
            g_input2x = max(g_input2x, 0)
            g_input2y = max(g_input2y, 0)
            g_x2x = max(g_x2x, 0)
            g_y2x = max(g_y2x, 0)
            g_x2y = max(g_x2y, 0)
            g_y2y = max(g_y2y, 0)

        print(g_input2x, g_input2y, g_x2x, g_y2y, g_y2x, g_x2y)

    return g_input2x, g_input2y, g_x2x, g_y2y, g_y2x, g_x2y, neuron_y, neuron_x, x_train[:, 0], x_train[:, 1]


# train the network
init_values = values_for_initialization(0)
aa, bb, cc, dd, ee, ff, neuron_yy, neuron_xx, x_inputs, y_inputs = train(init_values)

# plot the history of both neuron potentials
plt.figure()
plt.plot(neuron_xx, 'red', label='neuron_x_outputs')
plt.plot(x_inputs, 'green', label='neuron_x_inputs')
plt.plot(neuron_yy, 'blue', label='neuron_y_outputs')
plt.plot(y_inputs, 'grey', label='neuron_y_inputs')

plt.legend(loc='upper right')
plt.show()
