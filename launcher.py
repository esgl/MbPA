import numpy as np
from tqdm import tqdm
import logging
from args import set_args
from MbPA_test import MbPA_KNN_Test
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# import time

def main():
    # last_performance = []
    sess = tf.Session()
    for task in range(args.num_tasks_to_run):
        args.model_name = "mbpa_test"
        mbpa_test_model = MbPA_KNN_Test(sess, args)
        mnist = input_data.read_data_sets("mnist/", one_hot=True)
        task_permutation = np.random.permutation(784)
        logger.info("\nTraining task:{}/{}".format(task + 1, args.num_tasks_to_run))
        for i in tqdm(range(10000)):
            batch = mnist.train.next_batch(args.batch_size)
            batch = (batch[0][:, task_permutation[task]], batch[1])
            if True:
                embeddings = mbpa_test_model.train(batch[0], batch[1])
                # print("embedding.shape:{}", np.shape(embeddings))
                # print("value shape: {}", np.shape(batch[1]))
                # print("embeddings:", embeddings)
                # if i % args.memory_each == 0:
                mbpa_test_model.add_to_memory(embeddings, batch[1])
            else:
                mbpa_test_model.train(batch[0], batch[1])
        logger.info("memory length:{}".format(mbpa_test_model.memory_length))
        for test_task in range(task + 1):
            test_images = mnist.test.images

            test_images = test_images[:, task_permutation[test_task]]
            acc = mbpa_test_model.test(test_images, mnist.test.labels)
            acc = acc * 100
            # if args.num_tasks_to_run == task + 1:
            #     last_performance.append(acc)
            # print("Testing, task: ", test_task + 1, " \tAccuracy: ", acc)
            logger.info("Testing, task: {}\tAccuracy: {}".format(test_task + 1, acc))



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