import tensorflow as tf
from tensorflow.contrib import layers
from memory.memory import Memory
from ops import conv2d, linear
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.data import Dataset
class MbPA_KNN_Test:
    def __init__(self, sess, args):
        self.args = args
        self.session = sess
        self.w = {}
        self.eval_w = {}
        with tf.variable_scope(self.args.model_name):
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
            self.memory_sample_batch = tf.placeholder(tf.int16, shape=(), name="memory_sample_batch")
            with tf.variable_scope("training"):
                with tf.variable_scope("embedding"):
                    self.out = tf.reshape(self.x, [-1, 28, 28, 1])
                    with tf.variable_scope("conv"):
                        # self.out, self.w["l1_w"], self.w["l1_b"] = conv2d(
                        #     x=self.out,
                        #     output_dim=16,
                        #     kernel_size=[8, 8],
                        #     stride=[4, 4],
                        #     activation_fn=tf.nn.relu,
                        #     name="conv1"
                        # )
                        # self.out, self.w["l2_w"], self.w["l2_b"] = conv2d(
                        #     x=self.out,
                        #     output_dim=32,
                        #     kernel_size=[4, 4],
                        #     stride=[2, 2],
                        #     activation_fn=tf.nn.relu,
                        #     name="conv2"
                        # )
                        self.embed = layers.flatten(self.out)
                        self.embed_dim = self.embed.get_shape()[-1]
                self.M = Memory(self.args.memory_size, self.embed.get_shape()[-1], self.y.get_shape()[-1])
                embs_and_values = tf.py_func(self.get_memory_sample,
                                             [self.memory_sample_batch],
                                             [tf.float64, tf.float64])
                self.memory_batch_x = tf.to_float(embs_and_values[0])
                self.memory_batch_y = tf.to_float(embs_and_values[1])
                self.xa = tf.concat(values=[self.embed, self.memory_batch_x], axis=0)
                self.ya = tf.concat(values=[self.y, self.memory_batch_y], axis=0)
                with tf.variable_scope("fc"):
                    self.out = self.xa
                    # self.out, self.w["l3_w"], self.w["l3_b"] = linear(
                    #     input_=self.out,
                    #     output_size=1024,
                    #     activation_fn=tf.nn.relu,
                    #     name="fc_1"
                    # )
                    self.out, self.w["l4_w"], self.w["l4_b"] = linear(
                        input_=self.out,
                        output_size=10,
                        name="fc_2"
                    )
                    self.ya_ = self.out


                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.ya,
                    logits=self.ya_
                ))

                self.optim = tf.train.GradientDescentOptimizer(self.args.learning_rate).minimize(self.cross_entropy)
                self.correct_prediction = tf.equal(tf.argmax(self.ya, 1), tf.argmax(self.ya_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            self.session.run(tf.global_variables_initializer())

    def update_training_to_prediction(self):
        for name in self.eval_w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def train(self, xs, ys, memory_sample_batch):
        embeds, _ = self.session.run([self.embed, self.optim],
                                     feed_dict={
                                         self.x: xs,
                                         self.y: ys,
                                         self.memory_sample_batch: memory_sample_batch
                                     })
        return embeds

    def get_memory_sample(self, batch_size):
        xs, ys = self.M.sample(batch_size)
        return xs, ys

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

    def test(self, xs_test, ys_test):
        # self.update_training_to_prediction()
        acc = self.session.run(
            self.accuracy,
            feed_dict={
                self.x: xs_test,
                self.y: ys_test,
                self.memory_sample_batch: 0
            }
        )
        return acc

    @property
    def memory_length(self):
        return self.M.length
