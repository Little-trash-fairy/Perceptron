from numpy import exp, array, dot, random


class Neuron:
    def __init__(self):
        self.weights = array([random.normal(), random.normal()])
        self.bias = random.normal()

    def activation_function(self, x):
        return 1 / (1 + exp(-x))

    def activation_function_derivative(self, x):
        return x * (1 - x)

    def think(self, input):
        return self.activation_function(dot(input, self.weights) + self.bias)

    def train(self, input, error):
        learn_rate = 0.1

        sum = self.think(input)

        self.weights[0] -= learn_rate * error * input[0] * self.activation_function_derivative(sum)
        self.weights[1] -= learn_rate * error * input[1] * self.activation_function_derivative(sum)
        self.bias -= learn_rate * error * self.activation_function_derivative(sum)


class NeuralNetwork:

    def __init__(self):
        self.intermediate_layer_1 = Neuron()
        self.intermediate_layer_2 = Neuron()

        self.outer_layer = Neuron()

    def think(self, input):
        return self.outer_layer.think(array([
            self.intermediate_layer_1.think(input),
            self.intermediate_layer_2.think(input)
        ]))

    def train(self, input, output):
        intermediate_layer_1 = self.intermediate_layer_1.think(input)
        intermediate_layer_2 = self.intermediate_layer_2.think(input)

        sum = self.think(input)
        error = -2 * (output - sum)

        self.intermediate_layer_1.train(input, error * self.outer_layer.weights[0])
        self.intermediate_layer_2.train(input, error * self.outer_layer.weights[1])

        self.outer_layer.train(array([intermediate_layer_1, intermediate_layer_2]), error)


inputs = array([
    [9, 0.5],
    [8, 0.3],
    [7.1, 0.01],

    [7, 2.5],
    [5, 0.6],
    [3.1, 0.8],

    [3, 25],
    [0.71, 2.52],
    [2.29, 15],

    [0.07, 25.1],
    [0.001, 70.19],
    [0.05, 100.23]
])

outputs = array([
    0.1,
    0.1,
    0.1,

    0.2,
    0.2,
    0.2,

    0.3,
    0.3,
    0.3,

    0.4,
    0.4,
    0.4
])

network = NeuralNetwork()

for epoch in range(10000):
    for input, output in zip(inputs, outputs):
        network.train(input, output)

elem_1 = array([8.1332, 0.3434])
elem_2 = array([0.0689, 80])

print("elem_1: %.3f" % network.think(elem_1))
print("elem_2: %.3f" % network.think(elem_2))
