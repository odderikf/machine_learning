# from https://gitlab.com/ntnu-tdat3025/rnn/generate-characters/blob/master/generate1.py
import numpy as np
import tensorflow as tf


class LongShortTermMemoryModel:
    def __init__(self, encoding_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.nn.rnn_cell.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [])  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, encoding_size])  # Shape: [batch_size, max_time, encoding_size]
        self.y = tf.placeholder(tf.float32, [None, None, encoding_size])  # Shape: [batch_size, max_time, encoding_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, encoding_size]))
        b = tf.Variable(tf.random_normal([encoding_size]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.einsum('bts,se->bte', lstm, W), b)  # b: batch, t: time, s: state, e: encoding

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


char_encodings = [
    [1, 0, 0, 0, 0],  # ' '
    [0, 1, 0, 0, 0],  # 'h'
    [0, 0, 1, 0, 0],  # 'e'
    [0, 0, 0, 1, 0],  # 'l'
    [0, 0, 0, 0, 1],  # 'o'
]
encoding_size = np.shape(char_encodings)[1]

index_to_char = [' ', 'h', 'e', 'l', 'o']

x_train = [char_encodings[0], char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4]]  # ' hello'
y_train = [char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0]]  # 'hello '

model = LongShortTermMemoryModel(encoding_size)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

# Initialize model.in_state
zero_state = session.run(model.in_state, {model.batch_size: 1})

for epoch in range(500):
    session.run(minimize_operation, {model.batch_size: 1, model.x: [x_train], model.y: [y_train], model.in_state: zero_state})

    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", session.run(model.loss, {model.batch_size: 1, model.x: [x_train], model.y: [y_train], model.in_state: zero_state}))

        # Generate characters from the initial characters ' h'
        state = session.run(model.in_state, {model.batch_size: 1})
        text = ' h'
        y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[0]]], model.in_state: state})  # ' '
        y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[1]]], model.in_state: state})  # 'h'
        text += index_to_char[y.argmax()]
        for c in range(50):
            y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[y.argmax()]]], model.in_state: state})
            text += index_to_char[y[0].argmax()]
        print(text)

session.close()
