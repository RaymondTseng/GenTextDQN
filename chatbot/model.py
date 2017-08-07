# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf

from textdata import Batch
import numpy as np
from collections import deque
import random

GAMMA = 0.99
OBSERVE = 100
EPSILON = 0.001
REPLAY_MEMORY = 10000
MIN_BLEU_SCORE = 0.6
UPDATE_TIME = 100

class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args, textData):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        self.replayMemory = deque()
        self.timeStep = 0

        # Construct the graphs
        self.encoderInputs, self.decoderInputs, self.QValues, self.encoDecoCell, \
        self.decoderOutputs, self.W, self.b = self.buildNetwork('')

        self.encoderInputsT, self.decoderInputsT, self.QValuesT, self.encoDecoCellT, \
        self.decoderOutputsT, self.WT, self.bT = self.buildNetwork('T')

        self.createTrainingMethod()

        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.session.run(tf.initialize_all_variables())

    def createTrainingMethod(self):
        with tf.name_scope('placeholder_training_method'):
            self.actionInput = tf.placeholder(tf.float32, [None, self.
                                              textData.getVocabularySize()],
                                              name='action_inputs')
            self.actionStep = tf.placeholder(tf.int32, [None], name='action_step')
            self.yInput = tf.placeholder(tf.float32, [None], name='y_inputs')
        indices_shape = tf.shape(self.actionStep)
        indices = tf.concat([tf.reshape(tf.range(indices_shape[0]), [indices_shape[0], 1]),
                             tf.reshape(self.actionStep, [indices_shape[0], 1])], axis=1)
        QValues = tf.gather_nd(tf.transpose(self.QValues, perm=[1, 0, 2]), indices)
        Q_Action = tf.reduce_sum(tf.multiply(QValues, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    # Creation of the rnn cell
    def create_rnn_cell(self):
        encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
            self.args.hiddenSize,
        )
        if not self.args.test:  # TODO: Should use a placeholder instead
            encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                encoDecoCell,
                input_keep_prob=1.0,
                output_keep_prob=self.args.dropout
            )
        return encoDecoCell

    def create_multi_rnn_cell(self):
        return tf.contrib.rnn.MultiRNNCell(
                [self.create_rnn_cell() for _ in range(self.args.numLayers)],
        )

    def buildNetwork(self, name):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)
        W = self.weight_variable([self.args.hiddenSize, self.textData.getVocabularySize()])

        b = self.bias_variable([self.textData.getVocabularySize()])



        with tf.variable_scope('multi_lstm' + name):
            encoDecoCell = self.create_multi_rnn_cell()

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder' + name):
            encoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='encode_inputs') for _ in range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder' + name):
            decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='decode_inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)


        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        with tf.variable_scope('embedding_rnn_seq2seq' + name, reuse=None):
            decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
                decoderInputs,  # For training, we force the correct output (feed_previous=False)
                encoDecoCell,
                self.textData.getVocabularySize(),
                self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
                embedding_size=self.args.embeddingSize,  # Dimension of each word
                output_projection=(W, b),
                feed_previous=bool(self.args.test)  # When we test (self.args.test), we use previous output as next input (feed_previous)
            )

        # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): Should speed up
        # training and reduce memory usage. Other solution, use sampling softmax

        # For testing only
        Qvalues = None
        if self.args.test:
            QValues = 1

        # For training only
        else:
            QValues = [tf.matmul(decoderOutput, W) + b for decoderOutput in decoderOutputs]
        return encoderInputs, decoderInputs, QValues, encoDecoCell, decoderOutputs, W, b

    def initStatus(self, batch):
        feedDict = {}
        for i in range(self.args.maxLengthEnco):
            feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
        for i in range(self.args.maxLengthDeco):
            feedDict[self.decoderInputs[i]] = batch.decoderSeqs[i]
        QValues = self.session.run(self.QValues, feed_dict=feedDict)
        encodeSentences = batch.encoderSeqs
        targetSentences = batch.decoderSeqs
        decodeSentences = [tf.argmax(decodeSentence, axis=1).eval() for decodeSentence in QValues]

        return encodeSentences, targetSentences, decodeSentences

    def getAction(self, encodeSentence, decodeSentence):
        feedDict = {}
        for i in range(self.args.maxLengthEnco):
            feedDict[self.encoderInputs[i]] = [encodeSentence[i]]
        for i in range(self.args.maxLengthDeco):
            feedDict[self.decoderInputs[i]] = [decodeSentence[i]]
        timeStep = random.randrange(self.args.maxLengthDeco)
        QValues = self.session.run(self.QValues, feed_dict=feedDict)
        actionIndex = 0
        if random.random() <= EPSILON:
            actionIndex = random.randrange(self.textData.getVocabularySize())
        else:
            actionIndex = np.argmax(QValues[timeStep])
        return actionIndex, timeStep

    def setPerception(self, encodeSentence, decodeSentence, nextDecodeSentence,
                      actionIndex, step, reward):
        self.replayMemory.append(((encodeSentence, decodeSentence), (actionIndex, step), reward,
                                  (encodeSentence, nextDecodeSentence)))
        loss = None
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            loss = self.trainQNetwork()

        self.timeStep += 1
        return loss


    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, self.args.batchSize)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        feedDict = {}
        for j in range(self.args.maxLengthEnco):
            feedDict[self.encoderInputsT[j]] = [nextState_batch[i][0][j] for i in range(self.args.batchSize)]
        for j in range(self.args.maxLengthDeco):
            feedDict[self.decoderInputsT[j]] = [nextState_batch[i][1][j] for i in range(self.args.batchSize)]
        # QValues ==> [maxLengthDeco, batchSize, vocabularySize]
        QValues = self.session.run(self.QValuesT, feed_dict=feedDict)
        feedDict.clear()
        actions = []
        all_y = []
        steps = []
        for i in range(0, self.args.batchSize):
            reward = reward_batch[i]
            actionIndex, step = action_batch[i]
            steps.append(step)
            action = np.zeros(self.textData.getVocabularySize())
            action[actionIndex] = 1
            actions.append(action)
            if reward > MIN_BLEU_SCORE:
                all_y.append(reward)
            else:
                all_y.append(reward + GAMMA * np.max(QValues[step][i]))

        feedDict = {self.actionInput: actions,
                    self.actionStep: steps,
                    self.yInput: all_y}
        for j in range(self.args.maxLengthEnco):
            feedDict[self.encoderInputs[j]] = [state_batch[i][0][j] for i in range(self.args.batchSize)]
        for j in range(self.args.maxLengthDeco):
            feedDict[self.decoderInputs[j]] = [state_batch[i][1][j] for i in range(self.args.batchSize)]


        cost = self.session.run(self.cost, feed_dict=feedDict)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'save/' + 'network' + '-dqn', global_step=self.timeStep)

        return cost


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def copyTargetQNetwork(self):
        with tf.variable_scope('multi_lstm') as scope:
            scope.reuse_variables()
            self.encoDecoCellT = self.create_multi_rnn_cell()
        copyOperation = [self.WT.assign(self.W), self.bT.assign(self.b)]
        self.session.run(copyOperation)



