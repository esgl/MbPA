import tensorflow as tf
from tensorflow.contrib import layers
from memory.memory import Memory
from ops import conv2d, linear, clipped_error
class MbPA_KNN:
    def __init__(self, sess, args):
        self.args = args
        self.session = sess
        with tf.variable_scope(self.args.name):
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
            # self.trainable = tf.placeholder(tf.int32, shape=(), name="trainable")
            self.embed = self.embedding(self.x)

            self.embed_dim = self.embed.get_shape()[-1]
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
        # print("xs:", xs)
        embeds, _ = self.session.run([self.embed, self.optim],
                                     feed_dict={
                                         self.x: xs,
                                         self.y: ys
                                     })
        return embeds

    def test(self, xs_test, ys_test):
        acc = self.session.run(
            self.accuracy,
            feed_dict={
                self.x: xs_test,
                self.y: ys_test
            }
        )

        self.x_test = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="x_test")
        self.y_test = tf.placeholder(tf.float32, shape=[None, 10], name="y_test")

        self.y_test_ = self.output_network(self.x_test)
        self.cross_entropy_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_test,
            logits=self.y_test_
        ))
        self.optim_test = tf.train.GradientDescentOptimizer(self.args.learning_rate).minimize(self.cross_entropy_test)
        self.correct_prediction_test = tf.equal(tf.argmax(self.y_test, 1), tf.argmax(self.y_test_, 1))
        self.accuracy_test = tf.reduce_mean(tf.cast(self.correct_prediction_test, tf.float32))
        for xs_test_i in range(len(xs_test)):
            xs_test_ = xs_test[xs_test_i]
            test_sample = self.get_memory_sample(xs_test)
            self.session.run(
                self.optim_test,
                feed_dict={
                    self.x_test: test_sample[0],
                    self.y_test: test_sample[1]
                }
            )
        xs_test_embed = self.session.run(
            self.embed,
            feed_dict={
                self.x: xs_test
            }
        )
        acc = self.session.run(
            self.accuracy_test,
            feed_dict={
                self.x_test: xs_test_embed,
                self.y_test: ys_test
            }
        )
        return acc

    def get_memory_sample(self, xs, k=256):
        x, y, dist = self.M.sample_knn(xs, k)
        return x, y, dist

    def add_to_memory(self, xs, ys):
        # print(xs)
        if self.args.sample_add == "knn":
            self.M.add_knn(xs, ys)
        elif self.args.sample_add == "knn_lru":
            self.M.add_knn_lru(xs, ys)
        else:
            raise Exception("error sample adding type, pleace choose in ['normal', 'lru', 'rand']")

    def update_test_network(self):
        for name in self.w.keys():
            self.t_w_input[name] = tf.placeholder(tf.float32, self.w[name].get_shape().as_list(), name=name)
            self.t_w_assign_op[name] = self.test_w[name].assign(self.t_w_input[name])
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_iput[name]: self.w[name].eval})

    def test_output_network(self, xs_test_embed):
        out = xs_test_embed
        with tf.variable_scope("test_fc"):
            out, self.test_w["l3_w"], self.test_w["l3_b"] = linear(
                input_=out,
                output_size=1024,
                activation_fn=tf.nn.relu,
                name="fc_1"
            )
            out, self.test_w["l4_w"], self.test_w["l4_b"] = linear(
                input_=out,
                output_size=10,
                name="fc_2"
            )
        return out

    # @staticmethod
    def op_embedding(self, x):
        out = tf.reshape(x, [-1, 28, 28, 1])
        with tf.variable_scope("conv"):
            out, self.w["l1_w"], self.w["l1_b"] = conv2d(
                x=self.out,
                output_dim=16,
                kernel_size=[8,8],
                stride=[4, 4],
                activation_fn=tf.nn.relu,
                name="conv1"
            )
            out, self.w["l2_w"], self.w["l2_b"] = conv2d(
                x=self.out,
                output_dim=32,
                kernel_size=[4,4],
                stride=[2,2],
                activation_fn=tf.nn.relu,
                name="conv2"
            )
            embed = layers.flatten(out)
        return embed

    def op_output_network(self, embed):
        out = embed
        with tf.variable_scope("fc"):
            out, self.w["l3_w"], self.w["l3_b"] = linear(
                input_=out,
                output_size=1024,
                activation_fn=tf.nn.relu,
                name="fc_1"
            )
            out, self.w["l4_w"], self.w["l4_b"] = linear(
                input_=out,
                output_size=10,
                name="fc_2"
            )
        return out

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

    @property
    def memory_length(self):
        return self.M.length