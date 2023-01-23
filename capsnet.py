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
from keras import initializers, layers
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
        # TODO: figure out how to attach to non-capsule layer
        # build must support linking to both capsule and non capsule layers
        # if input_shape is 2-D (vector.shape = (1, x)) we need to add a 3rd dimension
        #
        # if input_shape is 3-D we have input.shape = (1, input_caps_dims, num_input_caps)
        # we want to create a weight matrix such that multiplying the input by the matrix
        # will give us the activations of the current layer capsules of shape
        # (1, caps_dims, num_caps)
        # TODO: determine the shape of the weight matrix that allows for easy multiplication
        # scratch notes:
        # for each connection between an input capsule xi with vector shape [id, 1]
        # and an output capsule yj with shape [od, 1] we need a weight matrix Wij
        # with shape [od, id] so that Wij * xj -> shape [od, 1]
        # There will be a matrix ([on, in]) of matrices ([od, id])
        # input looks like [in, id, 1] output looks like [on, od, 1]
        # weights look like [on, in, od, id]
        # W * i -> [on, in, od, 1]
        # reduce sum along in axis (axis 1) results in [on, od, 1]      **Note: We don't want to reduce sum
        #                                                                       we want to apply routing
        # 'each capsule in the [6 × 6] grid is sharing their
        # weights with each other' (p.4, Hinton et. al)
        # Each capsule belonging to the the same feature detector
        # shares the same weights.
        # TODO: split input_capsule_dimensions to reflect this

        self.input_num_capsules = input_shape[1]
        self.input_capsule_dimension = input_shape[2]

        self.weights = self.add_weight(name='weights',
                                       shape=(self.num_capsules, self.input_num_capsules,
                                              self.capsule_dimension, self.input_capsule_dimension),
                                       initializer=weight_initializer,
                                       trainable=True)

        super(CapsuleLayer, self).build(input_shape)

    """
    call method for custom Keras layer
    
    @param input_data - input to the capsule layer. If the input is
        from another capsule layer, it will have shape (num_caps, caps_dims, 1)
    """
    def call(self, input_data):
        # TODO: implement call method for custom Keras layer
        # First multiply weight matrix and inputs
        u_hat = tf.matmul(self.weights, input_data)
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


"""
softmax function (Eqn. 3)
"""
def softmax(b):
    return tf.exp(b) / tf.reduce_sum(tf.exp(b))


"""
Custom weight initializer

https://keras.io/api/layers/initializers/#creating-custom-initializers
"""
def weight_initializer(shape, dtype=None):
    initializer = initializers.initializers_v1.RandomNormal
    return initializer(shape, dtype)
