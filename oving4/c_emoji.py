# from https://gitlab.com/ntnu-tdat3025/rnn/generate-characters/blob/master/generate1.py
#%% setup
import numpy as np
import tensorflow as tf


x_raw = []
y_raw = []

with open('oving4/emoji_full.csv', 'r') as file:
    i = 0
    for line in file:
        if i >= 50: break
        i += 1
        if line[0] == '#': continue
        y, x = line.split(',')
        x = x.strip()
        y = y.strip()
        x_raw.append(x)
        y_raw.append(y)

emoji_list = list(set(y_raw))
training_size = len(x_raw)
encoding_size = len(emoji_list)
emoji_to_index = {y: i for i, y in enumerate(emoji_list)}
emoji_encodings = np.eye(encoding_size)


alphabet = 'abcdefghijklmnopqrstuvwxyz 0123456789'
character_encoding_size = len(alphabet)
char_to_index = {y: i for i, y in enumerate(alphabet)}
character_encodings = np.eye(character_encoding_size)

x_train = [[character_encodings[char_to_index[c]] for c in s] for s in x_raw]
sequence_lengths = [len(x) for x in x_train]
max_seq_length = max(sequence_lengths)
for x in x_train:
    while len(x) < max_seq_length:
        x.append([0 for _ in range(character_encoding_size)])
y_train = [emoji_encodings[emoji_to_index[c]] for c in y_raw]


class LongShortTermMemoryModel:
    def __init__(self):
        cell_state_size = 128
        self.cell = tf.nn.rnn_cell.LSTMCell(cell_state_size)

        self.batch_size = tf.placeholder(tf.int32, [])
        self.sequence_lengths = tf.placeholder(tf.float32, [None])
        self.x = tf.placeholder(tf.float32, [None, max_seq_length, character_encoding_size])  # Shape: [batch_size, max_time, encoding_size]
        self.y = tf.placeholder(tf.float32, [None, encoding_size])  # Shape: [batch_size, max_time, encoding_size]
        self.in_state = self.cell.zero_state(self.batch_size, tf.float32)

        self.W = tf.Variable(tf.random_normal([cell_state_size, encoding_size]))
        self.b = tf.Variable(tf.random_normal([encoding_size]))

        self.lstm, self.out_state = tf.nn.dynamic_rnn(self.cell, self.x, initial_state=self.in_state, sequence_length=self.sequence_lengths)

        logits = tf.nn.bias_add(tf.matmul(self.out_state.h, self.W), self.b)

        self.f = tf.nn.softmax(logits)

        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        self.min_op = tf.train.RMSPropOptimizer(0.05).minimize(self.loss)


def predict(text):
    y = ''
    state = session.run(model.in_state, {model.batch_size: 1})

    x_padded = [[character_encodings[char_to_index[c]] for c in s] for s in [text]]
    for x in x_padded:
        while len(x) < max_seq_length:
            x.append([0 for _ in range(character_encoding_size)])

    y, state = session.run([model.f, model.out_state],
                           {
                               model.batch_size: 1,
                               model.x: x_padded,
                               model.sequence_lengths: [len(text)],
                               model.in_state: state
                           })

    return emoji_list[y[0].argmax()]


model = LongShortTermMemoryModel()

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())


#%% zs
# Initialize model.in_state
zero_state = session.run(model.in_state, {model.batch_size: training_size})

#%% train
for epoch in range(500):
    session.run(model.min_op, {model.batch_size: training_size,
                               model.x: x_train, model.y: y_train,
                               model.in_state: zero_state,
                               model.sequence_lengths: sequence_lengths})


#%% test
text = input('Please input a word:\t').split()[0].lower()
while not text == 'exit':
    print("Program returns:", predict(text))
    text = input('Please input a word:\t').split()[0].lower()
print('\nExiting program.')

#%% close
session.close()
