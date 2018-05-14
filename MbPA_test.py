import tensorflow as tf
from tensorflow.contrib import layers
from memory.memory import Memory
from ops import conv2d, linear
import numpy as np
from tqdm import tqdm

class MbPA_KNN_Test:
    def __init__(self, sess, args):
        self.args = args
        self.session = sess
        self.w = {}
        self.eval_w = {}
        with tf.variable_scope(self.args.model_name):
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
            with tf.variable_scope("training"):
                with tf.variable_scope("embedding"):
                    self.out = tf.reshape(self.x, [-1, 28, 28, 1])
                    with tf.variable_scope("conv"):
                        self.out, self.w["l1_w"], self.w["l1_b"] = conv2d(
                            x=self.out,
                            output_dim=16,
                            kernel_size=[8, 8],
                            stride=[4, 4],
                            activation_fn=tf.nn.relu,
                            name="conv1"
                        )
                        self.out, self.w["l2_w"], self.w["l2_b"] = conv2d(
                            x=self.out,
                            output_dim=32,
                            kernel_size=[4, 4],
                            stride=[2, 2],
                            activation_fn=tf.nn.relu,
                            name="conv2"
                        )
                        self.embed = layers.flatten(self.out)
                        self.embed_dim = self.embed.get_shape()[-1]

                with tf.variable_scope("fc"):
                    self.out = self.embed
                    self.out, self.w["l3_w"], self.w["l3_b"] = linear(
                        input_=self.out,
                        output_size=1024,
                        activation_fn=tf.nn.relu,
                        name="fc_1"
                    )
                    self.out, self.w["l4_w"], self.w["l4_b"] = linear(
                        input_=self.out,
                        output_size=10,
                        name="fc_2"
                    )
                    self.y_ = self.out


                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y,
                    logits=self.y_
                ))

                self.optim = tf.train.GradientDescentOptimizer(self.args.learning_rate).minimize(self.cross_entropy)
                self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            self.M = Memory(self.args.memory_size, self.embed.get_shape()[-1], self.y.get_shape()[-1])
            # print("self.embed_dim: ", self.embed.get_shape().as_list())
            with tf.variable_scope("prediction"):
                self.x_eval = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="x_test")
                self.y_eval = tf.placeholder(tf.float32, shape=[None, 10], name="y_test")
                with tf.variable_scope("test_fc"):
                    self.out = self.x_eval
                    self.out, self.eval_w["l3_w"], self.eval_w["l3_b"] = linear(
                        input_=self.out,
                        output_size=1024,
                        activation_fn=tf.nn.relu,
                        name="fc_1"
                    )
                    self.out, self.eval_w["l4_w"], self.eval_w["l4_b"] = linear(
                        input_=self.out,
                        output_size=10,
                        name="fc_2"
                    )
                self.y_eval_ = self.out
                self.cross_entropy_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y_eval,
                    logits=self.y_eval_
                ))
                self.optim_eval = tf.train.GradientDescentOptimizer(self.args.learning_rate).minimize(
                    self.cross_entropy_eval)
                self.correct_prediction_eval = tf.equal(tf.argmax(self.y_eval, 1), tf.argmax(self.y_eval_, 1))
                self.accuracy_eval = tf.reduce_mean(tf.cast(self.correct_prediction_eval, tf.float32))

            with tf.variable_scope("training_to_prediction"):
                self.t_w_input = {}
                self.t_w_assign_op = {}
                for name in self.eval_w.keys():
                    self.t_w_input[name] = tf.placeholder(tf.float32, self.w[name].get_shape().as_list(), name=name)
                    self.t_w_assign_op[name] = self.eval_w[name].assign(self.t_w_input[name])
            self.session.run(tf.global_variables_initializer())

    def update_training_to_prediction(self):
        for name in self.eval_w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def train(self, xs, ys):
        embeds, _ = self.session.run([self.embed, self.optim],
                                     feed_dict={
                                         self.x: xs,
                                         self.y: ys
                                     })
        return embeds

    def get_memory_sample(self, xs, k=512):
        xs, ys, dist = self.M.sample_knn_test(xs, k)
        return xs, ys, dist

    def add_to_memory(self, xs, ys):
        if self.args.sample_add == "knn":
            self.M.add_knn(xs, ys)
        elif self.args.sample_add == "knn_lru":
            self.M.add_knn_lru(xs, ys)
        else:
            raise Exception("error sample adding type, pleace choose in ['normal', 'lru', 'rand']")

    def test(self, xs_test, ys_test):
        # self.update_training_to_prediction()
        test_embed = self.session.run(
            self.embed,
            feed_dict={
                self.x: xs_test
            }
        )
        acc = []
        for i in tqdm(range(len(test_embed))):
            self.update_training_to_prediction()
            xs_test_embed_ = test_embed[i]
            xs_test_embed_sample, ys_test_sample, _ = self.get_memory_sample(xs_test_embed_)
            self.session.run(self.optim_eval,
                             feed_dict={
                                 self.x_eval: xs_test_embed_sample,
                                 self.y_eval: ys_test_sample
                             })
            acc_ = self.session.run(self.accuracy_eval,
                feed_dict={
                    self.x_eval: [xs_test_embed_],
                    self.y_eval: [ys_test[i]]
                }
            )
            # print("acc_:", acc_)
            acc.append(acc_)
        acc = np.sum(np.array(acc)) / len(acc)
        return acc

    @property
    def memory_length(self):
        return self.M.length
