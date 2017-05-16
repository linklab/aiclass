import tensorflow as tf
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#%matplotlib inline

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

print(type(mnist.train.images), mnist.train.images.shape)
print(type(mnist.train.labels), mnist.train.labels.shape)

print(type(mnist.validation.images), mnist.validation.images.shape)
print(type(mnist.validation.labels), mnist.validation.labels.shape)

print(type(mnist.test.images), mnist.test.images.shape)
print(type(mnist.test.labels), mnist.test.labels.shape)


class VanillaNN:
    def __init__(self):
        self.epoch_list             = []
        self.train_error_list       = []
        self.validation_error_list  = []
        self.test_accuracy_list     = []

    def setData(self, n_input, n_classes, trainData, validationData, testData):
        self.n_input     = n_input            # 784
        self.n_classes   = n_classes          # 10
        self.trainData   = trainData
        self.validationData = validationData
        self.testData = testData

    #Create model
    def makeModel(self, learning_rate):
        self.learning_rate = learning_rate

        #tf Graph input
        self.x = tf.placeholder(tf.float32, (None, self.n_input))
        self.y = tf.placeholder(tf.float32, (None, self.n_classes))

        self.weight = tf.Variable(tf.zeros([self.n_input, self.n_classes]))
        self.bias = tf.Variable(tf.zeros([self.n_classes]))
        self.pred = tf.add(tf.matmul(self.x, self.weight), self.bias)

        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.error)

        self.prediction_ground_truth = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction_ground_truth, tf.float32))

    #Learning
    def learning(self, batch_size, training_epochs):
        self.batch_size      = batch_size
        self.training_epochs = training_epochs
        self.init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(self.init)
            self.total_batch = int(math.ceil(self.trainData.num_examples/float(self.batch_size)))
            print("Total batch: %d" % self.total_batch)

            for epoch in range(self.training_epochs):
                for i in range(self.total_batch):
                    batch_images, batch_labels = self.trainData.next_batch(self.batch_size)
                    _ = sess.run(self.optimizer, feed_dict={self.x: batch_images, self.y: batch_labels})

                self.epoch_list.append(epoch)

                # Train Error Value
                t_error_value = sess.run(self.error, feed_dict={self.x: self.trainData.images, self.y: self.trainData.labels})
                self.train_error_list.append(t_error_value)

                # Validation Error Value
                v_error_value = sess.run(self.error, feed_dict={self.x: self.validationData.images, self.y: self.validationData.labels})
                self.validation_error_list.append(v_error_value)

                # Test Accuracy Value
                accuracy_value = sess.run(self.accuracy, feed_dict={self.x: self.testData.images, self.y: self.testData.labels})
                self.test_accuracy_list.append(accuracy_value)
                print("epoch: %d, test_accuracy: %f" % (epoch, accuracy_value))

            self.drawErrorValues()

            self.drawFalsePrediction(sess, 10)

            print("Training % Test finished!")


    def drawErrorValues(self):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.plot(self.epoch_list, self.train_error_list, 'r', label='Train')
        plt.plot(self.epoch_list, self.validation_error_list, 'g', label='Validation')
        plt.ylabel('Total Error')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(122)
        plt.plot(self.epoch_list, self.test_accuracy_list, 'b', label='Test')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.yticks(np.arange(0.0, 1.0, 0.05))
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.savefig('error_accuracy_values.png')
        plt.show()

    def drawFalsePrediction(self, sess, numPrintImages):
        ground_truth = sess.run(tf.argmax(self.y, 1), feed_dict={self.y: self.testData.labels})
        prediction = sess.run(tf.argmax(self.pred, 1), feed_dict={self.x: self.testData.images})

        fig = plt.figure(figsize=(20, 5))
        j = 1
        for i in range(self.testData.num_examples):
            if (j > numPrintImages):
                break;
            if (prediction[i] != ground_truth[i]):
                print("Error Index: %s, Prediction: %s, Ground Truth: %s" % (i, prediction[i], ground_truth[i]))
                img = np.array(self.testData.images[i])
                img.shape = (28, 28)
                plt.subplot(1, numPrintImages, j)
                plt.imshow(img, cmap='gray')
                j += 1
        plt.savefig("false_prediction.png")

if __name__ == "__main__":
    #Parameter: Batch_size, Training_epochs, learning_rate
    vanilla = VanillaNN()
    vanilla.setData(n_input = 784,
                    n_classes = 10,
                    trainData = mnist.train,
                    validationData = mnist.validation,
                    testData = mnist.test)
    vanilla.makeModel(learning_rate = 0.001)
    vanilla.learning(batch_size = 100, training_epochs = 100)