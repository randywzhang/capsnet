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
Primary Capsule Layer doesn't route between previous layer so
we just need to reshape a layer into capsules.

The paper uses capsules for CV so we reshape a convolutional
layer.

@param num_channels - the number of unique capsule weights, used to
    calculate num_capsules. If reshaping a conv2D layer, 
    num_capsules = num_channels * shape[0] * shape[1] where shape is
    conv2D_output.shape
@param layer_to_shape - contains conv2D metadata as tuple (kernel, stride, padding)
"""
def PrimaryCapsuleLayer(inputs, capsule_dimension, num_channels, layer_to_shape):
    # TODO: shape the first capsule layer
    # TODO [REACH]: shape the first capsule layer for an arbitrary layer
    pass

"""
Keras customized layer

https://www.tutorialspoint.com/keras/keras_customized_layer.htm
"""
class CapsuleLayer(layers.Layer):

    def __init__(self, num_capsules, capsule_dimension, num_routings=3, **kwargs):
        self.num_capsules = num_capsules
        self.capsule_dimension = capsule_dimension
        # route at least once
        if num_routings < 1:
            num_routings = 1
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
        #
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
        # 'each capsule in the [6 ?? 6] grid is sharing their                    which is a sort of weighted
        # weights with each other' (p.4, Hinton et. al)                         reduce sum
        # Each capsule belonging to the the same feature detector
        # shares the same weights.
        # TODO: split input_capsule_dimensions to reflect this

        self.input_num_capsules = input_shape[1]
        self.input_capsule_dimension = input_shape[2]

        self.capsule_weights = self.add_weight(name='capsule_weights',
                                               shape=(self.num_capsules, self.input_num_capsules,
                                                      self.capsule_dimension, self.input_capsule_dimension),
                                               initializer=initializers.initializers_v1.RandomNormal,
                                               trainable=True)

        super(CapsuleLayer, self).build(input_shape)

    """
    call method for custom Keras layer
    
    @param input_data - input to the capsule layer. If the input is
        from another capsule layer, it will have shape (in_num_caps, in_caps_dims, 1)
    """
    def call(self, input_data, **kwargs):
        # TODO: get rid of extra dimensions of length 1, are they extra? see next TODO
        # TODO: test batch training, input_data shape = (num_caps, caps_dims, batch_size)
        # First multiply weight matrix and inputs to obtain prediction vectors
        u_hat = tf.matmul(self.capsule_weights, input_data)

        # apply routing algorithm on predictions
        capsule_activations = self.routing(u_hat)

        return capsule_activations

    """
    compute_output_shape method for custom Keras layer
    """
    def compute_output_shape(self, input_shape):
        # TODO: implement compute_output_shape method for custom Keras layer
        pass

    """
    routing algorithm described on p.3 (Procedure 1)
    
    @param u_hat - prediction vectors of layer i to layer j
        with shape (num_caps, in_num_caps, caps_dim, 1)
    """
    def routing(self, u_hat):
        # one coupling logit for each capsule connection
        logits = tf.zeros([self.num_capsules, self.input_num_capsules])

        for r in range(self.num_routings):
            # for all capsules i in input layer, calculate the softmax vector
            coupling_coeff = softmax(logits)

            # for all capsules j in current layer, calculate sj
            #
            # sj = sum ( c_i * u_hat_i )
            #
            # we want the output shape to remain the same, simply scaling the
            # prediction vectors by the coupling coefficients
            #
            # this means that for each logit[j] vector with dimension [in_num_caps]
            # we multiply against the corresponding matrix u_hat[j] which results
            # in a vector with dimension [caps_dims]
            #
            # after summing across all i, s dimensions should be [num_caps, caps_dims]
            # TODO: figure out a more efficient way to do this, tf function
            # TODO: look into tf.multiply()
            s = [tf.squeeze(tf.matmul(c, tf.squeeze(u_hat_j)))
                 for c, u_hat_j in zip(tf.expand_dims(coupling_coeff, 1), u_hat)]

            # squash s
            v = squash(s)

            # update logits
            #
            # if the prediction vector and activation agree, increase the coupling
            # between the input capsule and capsule j
            #
            # TODO: verify that we index i and j correctly throughout forward pass
            # bij += u_hat_ij * vj
            # bi += u_hat_i * vj
            # squeeze and expand_dims to get correct shapes for matmul
            # u_hat_i dimensions: [in_num_caps, caps_dims, 1] -> [in_num_caps, caps_dims]
            # v dimensions: [caps_dims] -> [caps_dims, 1]
            # u_hat_i * v dimensions: [in_num_caps, 1]
            # delta_b dimensions: [num_caps, in_num_caps, 1] -> [num_caps, in_num_caps]
            # TODO: tf function for this, tf.multiply()??
            delta_b = [tf.squeeze(tf.matmul(tf.squeeze(u_hat_i), vj))
                       for u_hat_i, vj in zip(u_hat, tf.expand_dims(v, -1))]
            logits += delta_b

        # add back dimension required for weight matmul and return activations
        return tf.expand_dims(v, -1)


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
    return tf.exp(b) / tf.reduce_sum(tf.exp(b), axis=-1, keepdims=True)


"""
Margin loss for digit existence (p. 3)

implementation of loss function described in section 3
"""
def margin_loss(vk, tk):
    # loss function variables
    m_plus = 0.9
    m_minus = 0.1
    lbda = 0.5

    # tk = 1 if the digit is present, 0 otherwise
    return tk * tf.square(max(0, m_plus - tf.norm(vk))) \
           + lbda * (1 - tk) * tf.square(max(0, tf.norm(vk) - m_minus))


"""
derivative of margin loss for back propagation
"""
def margin_loss_dx(x, y):
    pass


"""
activation derivatives
"""
def reLu_prime(a):
    return a > 0


def sigmoid_prime(a):
    return tf.exp(-a) / tf.square((1 + tf.exp(-a)))


"""
Custom weight initializers
"""
def weight_initializer(shape, dtype=None):
    # TODO: https://keras.io/api/layers/initializers/#creating-custom-initializers
    initializer = initializers.initializers_v1.RandomNormal
    return initializer(shape, dtype)
