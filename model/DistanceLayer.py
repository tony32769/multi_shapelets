from keras.engine.topology import Layer
import tensorflow as tf


class DistanceLayer(Layer):
    """
    Custom layer to compute shapelet-timeseries minimum distance matrix
    """
    def __init__(self, output_dim, shapelets, **kwargs):
        self.output_dim = output_dim
        self.shapelets = shapelets
        super(DistanceLayer, self).__init__(**kwargs)

    def get_shapelets(self, boh):
        return tf.convert_to_tensor(self.shapelets, dtype=tf.float32)

    def build(self, input_shape):
        # Create trainable shapelets variables for this layer.

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.shapelets.shape[1], self.shapelets.shape[2]),
                                      initializer=self.get_shapelets,
                                      trainable=True)
        del self.shapelets
        super(DistanceLayer, self).build(input_shape)

    def call(self, data):
        print(self.kernel.shape)
        tl, nc = data.shape[1], data.shape[2]
        ls = self.kernel.shape[1]
        ns = self.kernel.shape[0]


        D = tf.reshape(data,(-1,1,data.shape[1],data.shape[2]))
        A = tf.extract_image_patches(D,[1,1,ls,1],[1,1,1,1],[1,1,1,1],padding='VALID') #extract subsequences of
                                                                                        # length ls from each timeseries
        A = tf.reshape(A,(-1,tl - ls + 1,ls,nc))
        print(A.shape)

        P = tf.squared_difference(A, tf.reshape(self.kernel, [ns, 1, 1,
                                                              ls, nc]))

        o = tf.reduce_sum(P,
                          axis=(4, 3))
        o = tf.reshape(tf.reduce_min(o, axis=2), [ns, -1])

        o = tf.transpose(o)

        o = o / ls.value

        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
