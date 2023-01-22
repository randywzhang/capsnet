"""
This is an implementation of the capsule network architecture
described by Hinton, Sabour, and Frosst in their 2017 paper
'Dynamic Routing Between Capsules'

link:
https://arxiv.org/pdf/1710.09829.pdf

'There are many possible ways to implement the general idea
of capsules. The aim of this paper is not to explore this whole
space but simply to show that one fairly straightforward
implementation works well and that dynamic routing helps.'
    - Hinton, Sabour, Frosst
"""

import keras.backend as K
from keras import layers
import tensorflow as tf

"""
Keras customized layer

https://www.tutorialspoint.com/keras/keras_customized_layer.htm
"""
class CapsuleLayer(layers.Layer):

    def __init__(self, num_capsules, capsule_dimension, num_routings, **kwargs):
        self.num_capsules = num_capsules
        self.capsule_dimension = capsule_dimension
        self.num_routings = num_routings
        super(CapsuleLayer, self).__init__(**kwargs)

    """
    build method for custom Keras layer
    """
    def build(self, input_shape):
        # build must support linking to both capsule and non capsule layers
        # if input_shape is 2-D (vector.shape = (1, x)) we need to add a 3rd dimension
        #
        # if input_shape is 3-D we have input.shape = (1, input_caps_dims, num_input_caps)
        # we want to create a weight matrix such that multiplying the input by the matrix
        # will give us the activations of the current layer capsules of shape
        # (1, caps_dims, num_caps)
        # TODO: determine the shape of the weight matrix that allows for easy multiplication
        self.weights = self.add_weight(name='weights',
                                       shape=(input_shape[1], input_shape[2],
                                              self.num_capsules, self.capsule_dimension))
        super(CapsuleLayer, self).build(input_shape)

    """
    call method for custom Keras layer
    """
    def call(self, input_data):
        # TODO: implement call method for custom Keras layer
        pass

    """
    compute_output_shape method for custom Keras layer
    """
    def compute_output_shape(self, input_shape):
        # TODO: implement compute_output_shape method for custom Keras layer
        pass

    def routing(self, inputs):
        pass


"""
squash function as described in the paper (Eqn. 1)

@param s - the tensor of total inputs to layer j,
@return - v, the vector outputs of the capsules in 
    the current layer
"""
def squash(s):
    # calculate squared norms of sj, the inputs to capsule j, for
    # all capsules j in current layer
    # tf.square performs element-wise square on inputs.
    # TODO: determine s.shape and which axis to sum over
    squared_norms = tf.reduce_sum(tf.square(s), keepdims=True)

    # v = |s|^2 / (1 + |s|^2) * s / |s|            (Eqn. 1)
    # the addition of K.epsilon is to prevent division by 0
    return s * squared_norms / (1 + squared_norms) / (tf.sqrt(squared_norms) + K.epsilon())
