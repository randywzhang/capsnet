"""
Author: Randy Zhang

This is an implementation of the capsule network architecture
described by Hinton, Sabour, and Frosst in their 2017 paper
'Dynamic Routing Between Capsules'

link:
https://arxiv.org/pdf/1710.09829.pdf
"""

from keras import initializers, layers

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

    def routing(self, inputs):
        pass


"""
squash function as described in the paper (Eqn. 1)

@param input_vectors - the prediction vectors u_hat from the capsules in
    the previous layer, where u_hat is obtained by multiplying the weight
    matrices by the activations u in the previous layer (Eqn. 2)
@return - vj, the vector output of the current capsule
"""
def squash(input_vectors):
    pass
