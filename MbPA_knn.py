import tensorflow as tf
from tensorflow.contrib import layers
from memory.memory import Memory
class MbPA_KNN:
    def __init__(self, sess, args):
        self.args = args
        self.session = sess
        with tf.variable_scope(self.args.name):
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
            # self.trainable = tf.placeholder(tf.int32, shape=(), name="trainable")
            self.embed = self.embedding(self.x)

            self.M = Memory(self.args.memory_size, self.embed.get_shape()[-1], self.y.get_shape()[-1])
            self.y_ = self.output_network(self.embed)

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y,
                logits=self.y_
            ))

            self.optim = tf.train.GradientDescentOptimizer(self.args.learning_rate).minimize(self.cross_entropy)
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            self.session.run(tf.global_variables_initializer())

    def train(self, xs, ys):
        embeds, _ = self.session.run([self.embed, self.optim],
                                     feed_dict={
                                         self.x: xs,
                                         self.y: ys
                                     })

    def test(self, xs_test, ys_test):
        acc = self.session.run(
            self.accuracy,
            feed_dict={
                self.x: xs_test,
                self.y: ys_test
            }
        )
        return acc

    def get_memory_sample(self, xs, k=256):
        x, y, dist = self.M.sample_knn(xs, k)
        return x, y, dist

    def add_to_memory(self, xs, ys):
        if self.args.sample_add == "knn":
            self.M.add_knn(xs, ys)
        elif self.args.sample_add == "knn_lru":
            self.M.add_knn_lru(xs, ys)
        else:
            raise Exception("error sample adding type, pleace choose in ['normal', 'lru', 'rand']")

    @staticmethod
    def embedding(x):
        out = tf.reshape(x, [-1, 28, 28, 1])
        with tf.variable_scope(name_or_scope="conv1"):
            out = layers.convolution2d(
                inputs=out,
                num_outputs=16,
                kernel_size=8,
                stride=4
            )
            out = tf.nn.relu(out)
        with tf.variable_scope(name_or_scope="conv2"):
            out = layers.convolution2d(
                inputs=out,
                num_outputs=32,
                kernel_size=4,
                stride=2
            )
            out = tf.nn.relu(out)

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