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

        # Construct the graphs
        self.encoderInputs, self.decoderInputs, self.QValues, self.encoDecoCell, \
        self.decoderOutputs, self.W, self.b = self.buildNetwork()

        self.encoderInputsT, self.decoderInputsT, self.QValuesT, self.encoDecoCellT, \
        self.decoderOutputsT, self.WT, self.bT = self.buildNetwork()

        self.createTrainingMethod()

        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.textData.getVocabularySize])
        self.actionStep = tf.placeholder("int", [None])
        self.yInput = tf.placeholder("float", [None])
        indices_shape = tf.shape(self.actionStep)
        indices = tf.concat([tf.reshape(tf.range(indices_shape[0]), [indices_shape[0], 1]),
                             tf.reshape(self.actionStep, [indices_shape[0], 1])], axis=1)
        QValues = tf.gather_nd(self.QValues, indices)
        Q_Action = tf.reduce_sum(tf.multiply(QValues, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
        W = tf.get_variable(
            'weights',
            (self.args.hiddenSize, self.textData.getVocabularySize()),
            # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
            dtype=self.dtype
        )

        b = tf.get_variable(
            'bias',
            self.textData.getVocabularySize(),
            initializer=tf.constant_initializer(),
            dtype=self.dtype
        )



        # Creation of the rnn cell
        def create_rnn_cell():
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
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)


        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
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
        QValues = self.session.run((self.QValues,), feed_dict=feedDict)
        encodeSentences = batch.encoderSeqs
        targetSentences = batch.decoderSeqs
        decodeSentences = tf.reduce_max(QValues, axis=2)

        return encodeSentences, targetSentences, decodeSentences

    def getAction(self, encodeSentence):
        feedDict = {}
        for i in range(self.args.maxLengthEnco):
            feedDict[self.encoderInputs[i]] = encodeSentence[i]
        timeStep = random.randrange(self.args.maxLengthDeco)
        QValues = self.session.run((self.QValues,), feed_dict=feedDict)
        action = np.zeros(self.textData.getVocabularySize())
        actionIndex = 0
        if random.random() <= EPSILON:
            actionIndex = random.randrange(self.textData.getVocabularySize)
        else:
            actionIndex = np.argmax(QValues[timeStep])
        action[actionIndex] = 1
        return action, timeStep


    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
