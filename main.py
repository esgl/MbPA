import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from MbPA import MbPA
from MbPA_test import MbPA_KNN_Test
import time
from tqdm import tqdm
import logging
from args import set_args

from utils import plot_result

def main(_):
    with tf.Session() as sess:
        # print("\nParamters used: ", args, "\n")
        logger.info("\nParamters used: {}\n".format(args))

        args.model_name = "mlp"
        baseline_model = MbPA(sess, args)
        args.model_name = "mbpa"
        mbpa_model = MbPA(sess, args)

        args.model_name = "mbpa_test"
        mbpa_test_model = MbPA_KNN_Test(sess, args)
        mnist = input_data.read_data_sets("mnist/", one_hot=True)

        task_permutation = []
        for task in range(args.num_tasks_to_run):
            task_permutation.append(np.random.permutation(784))

        # print("\nBaseline MLP training...\n")
        logger.info("\nBaseline MLP training...\n")
        start = time.time()
        performance_baseline = training(baseline_model, mnist, task_permutation, False)
        # performance_baseline = training_knn(baseline_model, mnist, task_permutation, False)
        end = time.time()
        time_needed_baseline = round(end - start)
        logger.info("Training time elapased: {}s".format(time_needed_baseline))

        # print("\nMemory-based parameter Adaptation....\n")
        logger.info("\nMemory-based parameter Adaptation....\n")
        start = time.time()
        mbpa_performance = training(mbpa_model, mnist, task_permutation, True)
        # mbpa_performance = training_knn(mbpa_model, mnist, task_permutation, True)
        end = time.time()
        time_needed_baseline = round(end - start)
        # print("Training time elapased: ", time_needed_baseline, "s")
        logger.info("Training time elapased: {}s".format(time_needed_baseline))

        # print("\nMemory-based parameter Adaptation....\n")
        logger.info("\nMemory-based test parameter Adaptation....\n")
        start = time.time()
        # mbpa_performance = training(mbpa_model, mnist, task_permutation, True)
        mbpa_test_performance = training_knn(mbpa_test_model, mnist, task_permutation, True)
        end = time.time()
        time_needed_baseline = round(end - start)
        print("Training time elapased: ", time_needed_baseline, "s")
        logger.info("Training time elapased: {}s".format(time_needed_baseline))
        plot_result(args.num_tasks_to_run, performance_baseline, mbpa_performance, mbpa_test_performance,
                    (args.log.split("/")[-1]).split(".")[0])


def training_knn(model, mnist, task_permutation, use_memory=False):
    last_performance = []
    # num_example = mnist.train.num_examples
    # perm0 = np.arange(num_example)
    # np.random.shuffle(perm0)
    # mnist_train_images = np.reshape(mnist.train.images[perm0[:10000]], [-1, 28, 28, 1])
    # mnist_train_labels = mnist.train.labels[perm0[:10000]]
    # mnist_train = DataSet(mnist_train_images, mnist_train_labels)
    mnist_train = mnist.train
    for task in range(args.num_tasks_to_run):
        logger.info("\nTraining task:{}/{}".format(task + 1, args.num_tasks_to_run))
        for i in tqdm(range(10000), dynamic_ncols=True):
            # batch = mnist.train.next_batch(args.batch_size)
            batch = mnist_train.next_batch(args.batch_size)
            batch = (batch[0][:, task_permutation[task]], batch[1])
            if use_memory:
                embeddings = model.train(batch[0], batch[1])
                # print("embedding.shape:{}", np.shape(embeddings))
                # print("value shape: {}", np.shape(batch[1]))
                # print("embeddings:", embeddings)
                if i % args.memory_each == 0:
                    model.add_to_memory(embeddings, batch[1])
            else:
                model.train(batch[0], batch[1])
        acc_ = model.test(batch[0], batch[1])
        acc_ = acc_ * 100
        logger.info("training Accuracy: {}".format(acc_))
        logger.info("memory length:{}".format(model.memory_length))
        average_acc = []
        for test_task in range(task + 1):
            test_images = mnist.test.images

            test_images = test_images[:, task_permutation[test_task]]
            acc = model.test(test_images[:1000], mnist.test.labels[:1000])
            acc = acc * 100
            average_acc.append(acc)
            if args.num_tasks_to_run == task + 1:
                last_performance.append(acc)
            # print("Testing, task: ", test_task + 1, " \tAccuracy: ", acc)
            logger.info("Testing, task: {}\tAccuracy: {}".format(test_task + 1, acc))
        logger.info("average accuracy: {}".format(np.mean(average_acc)))
    return last_performance

def training(model, mnist, task_permutation, use_memory=False):
    last_performance = []
    # num_example = mnist.train.num_examples
    # perm0 = np.arange(num_example)
    # np.random.shuffle(perm0)
    # mnist_train_images = np.reshape(mnist.train.images[perm0[:10000]], [-1, 28, 28, 1])
    # mnist_train_labels = mnist.train.labels[perm0[:10000]]
    # mnist_train = DataSet(mnist_train_images, mnist_train_labels)
    #
    mnist_train = mnist.train
    for task in range(args.num_tasks_to_run):
        # print("\nTraining task:", task + 1, "/", args.num_tasks_to_run)
        logger.info("\nTraining task:{}/{}".format(task + 1, args.num_tasks_to_run))
        for i in tqdm(range(10000), dynamic_ncols=True):
            # batch = mnist.train.next_batch(args.batch_size)
            batch = mnist_train.next_batch(args.batch_size)
            batch = (batch[0][:, task_permutation[task]], batch[1])
            if use_memory:

                embeddings = model.train(batch[0], batch[1], args.batch_size)
                if i % args.memory_each == 0:
                    model.add_to_memory(embeddings, batch[1])
            else:
                model.train(batch[0], batch[1], 0)
        acc_ = model.test(batch[0], batch[1])
        acc_ = acc_ * 100
        logger.info("training Accuracy: {}".format(acc_))

        average_acc = []
        for test_task in range(task + 1):
            test_images = mnist.test.images

            test_images = test_images[:, task_permutation[test_task]]
            acc = model.test(test_images[:1000], mnist.test.labels[:1000])
            acc = acc * 100
            average_acc.append(acc)
            if args.num_tasks_to_run == task + 1:
                last_performance.append(acc)
            # print("Testing, task: ", test_task + 1, " \tAccuracy: ", acc)
            logger.info("Testing, task: {}\tAccuracy: {}".format(test_task + 1, acc))
        logger.info("average accuracy: {}".format(np.mean(average_acc)))
    return last_performance

if __name__ == "__main__":

    args = set_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log, mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    tf.app.run()
