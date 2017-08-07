# -*- coding:utf-8 -*-

import numpy as np


def bleuScore(decodeSentence, targetSentence, actionIndex, timeStep):
    nextDecodeSentence = decodeSentence
    nextDecodeSentence[timeStep] = actionIndex
    status = np.equal(targetSentence, nextDecodeSentence).astype(int)
    result = np.sum(status).astype(float)
    result /= len(status)
    return nextDecodeSentence, result

