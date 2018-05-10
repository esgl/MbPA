import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from MbPA import MbPA
import time
from tqdm import tqdm
import logging

def plot_result(num_tasks_to_run, baseline_mlp, memoryadaoted):
    import matplotlib.pyplot as plt
    tasks = range(1, num_tasks_to_run + 1)
    plt.plot(tasks, baseline_mlp[::-1])
    plt.plot(tasks, memoryadaoted[::-1])
    plt.legend(["Baseline-MLP", "RMA"], loc="lower right")
    plt.xlabel("Number of Tasks")
    plt.ylabel("Accuracy (%)")
    plt.ylim([1, 100])
    plt.xticks(tasks)
    plt.show()



def main(_):
    with tf.Session() as sess:
        print("\nParamters used: ", args, "\n")
        logger.info("\nParamters used: {}\n".format(args))

        args.name = "mlp"
        baseline_model = MbPA(sess, args)
        args.name = "mbpa"
        mbpa_model = MbPA(sess, args)
        mnist = input_data.read_data_sets("mnist/", one_hot=True)

        task_permutation = []
        for task in range(args.num_tasks_to_run):
            task_permutation.append(np.random.permutation(784))

        print("\nBaseline MLP training...\n")
        logger.info("\nBaseline MLP training...\n")
        start = time.time()
        performance_baseline = training(baseline_model, mnist, task_permutation, False)
        end = time.time()
        time_needed_baseline = round(end - start)
        print("Training time elapased: ", time_needed_baseline, "s")
        logger.info("Training time elapased: {}s".format(time_needed_baseline))

        print("\nMemory-based parameter Adaptation....\n")
        logger.info("\nMemory-based parameter Adaptation....\n")
        start = time.time()
        mbpa_performance = training(mbpa_model, mnist, task_permutation, True)
        end = time.time()
        time_needed_baseline = round(end - start)
        print("Training time elapased: ", time_needed_baseline, "s")
        logger.info("Training time elapased: {}s".format(time_needed_baseline))
        plot_result(args.num_tasks_to_run, performance_baseline, mbpa_performance)


def training(model, mnist, task_permutation, use_memory=False):
    last_performance = []
    for task in range(args.num_tasks_to_run):
        print("\nTraining task:", task + 1, "/", args.num_tasks_to_run)
        logger.info("\nTraining task:{}/{}".format(task + 1, args.num_tasks_to_run))
        for i in tqdm(range(10000)):
            batch = mnist.train.next_batch(args.batch_size)
            batch = (batch[0][:, task_permutation[task]], batch[1])
            if use_memory:

                embeddings = model.train(batch[0], batch[1], args.batch_size)
                if i % args.memory_each == 0:
                    model.add_to_memory(embeddings, batch[1])
            else:
                model.train(batch[0], batch[1], 0)

        for test_task in range(task + 1):
            test_images = mnist.test.images

            test_images = test_images[:, task_permutation[test_task]]
            acc = model.test(test_images, mnist.test.labels)
            acc = acc * 100
            if args.num_tasks_to_run == task + 1:
                last_performance.append(acc)
            print("Testing, task: ", test_task + 1, " \tAccuracy: ", acc)
            logger.info("Testing, task: {}\tAccuracy: {}".format(test_task + 1, acc))

    return last_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks-to-run", type=int, default=20,
                        help="Number of task to run")
    parser.add_argument("--memory-size", type=int, default=15000,
                        help="Memory size")
    parser.add_argument("--memory-each", type=int, default=1000,
                        help="Add to memory after these number of steps")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Size of batch for updates")
    parser.add_argument("--learning-rate", type=float, default=0.5,
                        help="Learning rate")
    parser.add_argument("--memory_using_start", type=int, default=1000,
                        help="using memory after n step ")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("logs/log.txt")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    tf.app.run()
