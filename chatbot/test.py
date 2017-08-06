# -*- coding:utf-8 -*-

import tensorflow as tf

# # B = 3
# # N = 4
# # M = 2
# # [B x N x 3]
# data = tf.constant([
#     [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#     [[100, 101, 102], [103, 104, 105], [106, 107, 108], [109, 110, 111]],
#     [[200, 201, 202], [203, 204, 205], [206, 207, 208], [209, 210, 211]],
#     ])
#
# # [B x M]
# indices = tf.constant([
#     [0, 2],
#     [1, 3],
#     [3, 2],
#     ])
#
# indices_shape = tf.shape(indices)
# # (B, M)
#
# # tf.range(indices_shape[0]) ==> [0, 1, 2, ... , B]
# # [indices_shape[0], 1] ==> [B, 1]
# # [1, indices_shape[1]] ==> [1, M]
# indices_help = tf.tile(tf.reshape(tf.range(indices_shape[0]), [indices_shape[0], 1]), [1, indices_shape[1]])
# # indices_help ==> [B, M]
# # tf.expand_dims(indices_help, 2) ==> [B, M, 1]
# # tf.expand_dims(indices, 2) ==> [B, M, 1]
# indices_ext = tf.concat([tf.expand_dims(indices_help, 2), tf.expand_dims(indices, 2)], axis = 2)
# # indices_ext ==> [B, M, 2]
# new_data = tf.gather_nd(data, indices_ext)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print ('indices_help')
#     print sess.run(indices_help)
#     print ('indices_ext')
#     print sess.run(indices_ext)
#     print('data')
#     print(sess.run(data))
#     print('\nindices')
#     print(sess.run(indices))
#     print('\nnew_data')
#     print(sess.run(new_data))

from numpy import random
import numpy as np
data = random.random(size=(5, 12, 10))
data = tf.convert_to_tensor(data)
step = np.array([5,8,10,1,3])
step = tf.convert_to_tensor(step, dtype=tf.int32)
indices_shape = tf.shape(step)
range_step = tf.range(indices_shape[0])
indices = tf.concat([tf.reshape(tf.range(indices_shape[0]), [indices_shape[0], 1]),
                    tf.reshape(step, [indices_shape[0], 1])], axis=1)
new_data = tf.gather_nd(data, indices)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(new_data)

