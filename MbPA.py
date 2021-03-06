import tensorflow as tf
from tensorflow.contrib import layers
from memory.memory import Memory
from ops import conv2d, linear
class MbPA:

    def __init__(self, sess, args):
        with tf.variable_scope(args.model_name):
            self.args = args
            self.learning_rate = args.learning_rate
            self.session = sess

            self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
            # self.trainable = tf.placeholder(tf.int32, shape=(), name="trainable")
            self.memory_sample_batch = tf.placeholder(tf.int16, shape=(), name="memory_sample_batch")

            self.embed = self.embedding(self.x)

            self.M = Memory(args.memory_size, self.embed.get_shape()[-1], self.y.get_shape()[-1])
            embs_and_values = tf.py_func(self.get_memory_sample, [self.memory_sample_batch],
                                         [tf.float64, tf.float64])

            self.memory_batch_x = tf.to_float(embs_and_values[0])
            self.memory_batch_y = tf.to_float(embs_and_values[1])
            self.xa = tf.concat(values=[self.embed, self.memory_batch_x], axis=0)
            self.ya = tf.concat(values=[self.y, self.memory_batch_y], axis=0)

            self.y_ = self.output_network(self.xa)

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ya,
                                                                                        logits=self.y_))
            self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
            self.correct_prediction = tf.equal(tf.argmax(self.ya, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            self.session.run(tf.global_variables_initializer())

    def train(self, xs, ys, memory_sample_batch):
        # print(memory_sample_batch)
        embeds, _ = self.session.run([self.embed, self.optim],
                         feed_dict={
                             self.x: xs,
                             self.y: ys,
                             self.memory_sample_batch: memory_sample_batch
                         })
        return embeds

    def test(self, xs_test, ys_test):
        acc = self.session.run(
            self.accuracy,
            feed_dict={
                self.x: xs_test,
                self.y : ys_test,
                self.memory_sample_batch: 0
            }
        )
        return acc

    def get_memory_sample(self, batch_size):
        x, y = self.M.sample(batch_size)
        return x, y

    def add_to_memory(self, xs, ys):
        if self.args.sample_add == "normal":
            self.M.add(xs, ys)
        elif self.args.sample_add == "lru":
            self.M.add_lru(xs, ys)
        elif self.args.sample_add == "rand":
            self.M.add_rand(xs, ys)
        elif self.args.sample_add == "knn":
            self.M.add_knn(xs, ys)
        elif self.args.sample_add == "knn_lru":
            self.M.add_knn_lru(xs, ys)
        else:
            raise Exception("error sample adding type, pleace choose in ['normal', 'lru', 'rand']")


    @staticmethod
    def embedding(x):
        out = tf.reshape(x, [-1, 28, 28, 1])
        # convs = [(16, 8, 4), (32, 4, 2)]
        # with tf.variable_scope("conv1"):
            # out = layers.convolution2d(inputs=out,
            #                            num_outputs=16,
            #                            kernel_size=8,
            #                            stride=4,
            #                            trainable=trainable)
            # out = tf.nn.relu(out)
            # out = tf.nn.max_pool(out, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        with tf.variable_scope("conv2"):
            # out = layers.convolution2d(inputs=out,
            #                            num_outputs=32,
            #                            kernel_size=4,
            #                            stride=2,
            #                            trainable=trainable)
            # out = tf.nn.relu(out)
            # out = tf.nn.max_pool(out, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
            embed = layers.flatten(out)
        return embed

    @staticmethod
    def output_network(embed):
        out = embed
        with tf.variable_scope("fc_1"):
            out = layers.fully_connected(
                inputs=out,
                num_outputs=1024
            )
            out = tf.nn.relu(out)
        with tf.variable_scope("fc_2"):
            out = layers.fully_connected(
                inputs=out,
                num_outputs=10
            )
        return out


