# from https://gitlab.com/ntnu-tdat3025/cnn/mnist/blob/master/compute2.py
import numpy as np
import tensorflow as tf

(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train_, (-1, 28, 28, 1))  # tf.layers.conv2d takes 4D input
y_train = np.zeros((y_train_.size, 10))
y_train[np.arange(y_train_.size), y_train_] = 1

batches = 600  # Divide training data into batches to speed up optimization
x_train_batches = np.split(x_train, batches)
y_train_batches = np.split(y_train, batches)

x_test = np.reshape(x_test_, (-1, 28, 28, 1))
y_test = np.zeros((y_test_.size, 10))
y_test[np.arange(y_test_.size), y_test_] = 1


class ConvolutionalNeuralNetworkModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32)

        conv1 = tf.layers.conv2d(self.x, filters=32, kernel_size=[5, 5], strides=[1, 1], padding='same')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], padding='same')

        # Logits
        logits = tf.layers.dense(tf.layers.flatten(pool1), units=10)

        # Predictor
        f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))


model = ConvolutionalNeuralNetworkModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.AdamOptimizer(0.001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(20):
    for batch in range(batches):
        session.run(minimize_operation, {model.x: x_train_batches[batch], model.y: y_train_batches[batch]})

    print("epoch", epoch)
    print("accuracy", session.run(model.accuracy, {model.x: x_test, model.y: y_test}))

session.close()
