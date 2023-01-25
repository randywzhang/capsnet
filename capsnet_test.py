import unittest
import capsnet
import tensorflow as tf

"""
Test parameters
"""
routings = 3
num_capsules = 10
capsule_dimension = 8
output_shape = [num_capsules, capsule_dimension, 1]
input_num_capsules = 6
input_capsule_dimension = 4
input_shape = [1, input_num_capsules, input_capsule_dimension]
input_data_shape = [input_num_capsules, input_capsule_dimension, 1]

class TestCapsNet(unittest.TestCase):

    def setup(self):
        self.test_layer = capsnet.CapsuleLayer(num_capsules, capsule_dimension, routings)
        self.test_layer.build(input_shape)

        self.test_input_data = tf.random.normal(input_data_shape)

    def test_build(self):
        self.setup()
        expected_shape = tf.TensorShape([num_capsules, input_num_capsules, capsule_dimension, input_capsule_dimension])

        # UUT
        weight_shape = self.test_layer.capsule_weights.shape

        # Verify
        self.assertEqual(weight_shape, expected_shape)

    def test_call(self):
        self.setup()

        # UUT
        output = self.test_layer.call(self.test_input_data)

        # Verify
        self.assertEqual(output.shape, output_shape)


if __name__ == "__main__":
    unittest.main()
