import tensorflow as tf
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])


x_image = tf.reshape(x, [-1, 28, 28, 1])
print(x_image.get_shape())


keep_prob = tf.placeholder(tf.float32)

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
d_h_pool1 = tf.nn.dropout(h_pool1, keep_prob=keep_prob)
print(h_conv1)
print(h_pool1)
print(d_h_pool1)


W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(d_h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
d_h_pool2 = tf.nn.dropout(h_pool2, keep_prob=keep_prob)
print(h_conv2)
print(h_pool2)
print(d_h_pool2)


h_pool2_flat = tf.reshape(d_h_pool2, [-1, 7 * 7 * 64])


W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
d_h_fc1 = tf.nn.dropout(h_fc1, keep_prob=keep_prob)


W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
pred = tf.matmul(d_h_fc1, W_fc2) + b_fc2


error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(error)


init = tf.global_variables_initializer()

# Parameters
training_epochs = 50
learning_rate = 0.001
batch_size = 100

# Calculate accuracy with a Test model
prediction_ground_truth = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction_ground_truth, tf.float32))


def printLossAccuracyForTestData(epoch, sess, test_error_value_list, test_accuracy_list):
    accuracy_value, error_value = sess.run((accuracy, error), 
                                           feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.5})
    test_error_value_list.append(error_value)
    test_accuracy_list.append(accuracy_value)
    print("epoch: %d, test_error_value: %f, test_accuracy: %f" % ( epoch, error_value, accuracy_value ))

def drawErrorValues(epoch_list, train_error_value_list, validation_error_value_list, test_error_value_list, test_accuracy_list):
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(epoch_list, train_error_value_list, 'r', label='Train')
    plt.plot(epoch_list, validation_error_value_list, 'g', label='Validation')
    plt.plot(epoch_list, test_error_value_list, 'b', label='Test')
    plt.ylabel('Total Error')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.subplot(122)
    plt.plot(epoch_list, test_accuracy_list, 'b', label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.yticks(np.arange(min(test_accuracy_list), max(test_accuracy_list), 0.05))
    plt.grid(True)
    plt.legend(loc='lower right')            
    plt.show()

def drawFalsePrediction(sess, numPrintImages):
    ground_truth = sess.run(tf.argmax(y, 1), feed_dict={y: mnist.test.labels})
    prediction = sess.run(tf.argmax(pred, 1), feed_dict={x: mnist.test.images, keep_prob: 0.5})

    fig = plt.figure(figsize=(20, 5))
    j = 1
    for i in range(mnist.test.num_examples):
        if (j > numPrintImages):
            break;
        if (prediction[i] != ground_truth[i]):
            print("Error Index: %s, Prediction: %s, Ground Truth: %s" % (i, prediction[i], ground_truth[i]))
            img = np.array(mnist.test.images[i])
            img.shape = (28, 28)
            plt.subplot(1, numPrintImages, j)
            plt.imshow(img)
            j += 1    

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(math.ceil(mnist.train.num_examples/float(batch_size)))

    print("total batch: %d" % total_batch)

    epoch_list                  = []
    train_error_value_list      = []
    validation_error_value_list = []
    test_error_value_list       = []
    test_accuracy_list          = []

    # Training cycle
    for epoch in range(training_epochs):

        # Loop over all batches
        for i in range(total_batch):
            batch_images, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_images, y: batch_labels, keep_prob: 0.5})

        epoch_list.append(epoch)

        # Train Error Value
        t_error_value = sess.run(error, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.5})
        train_error_value_list.append(t_error_value)

        # Validation Error Value
        v_error_value = sess.run(error, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 0.5})
        validation_error_value_list.append(v_error_value)

        printLossAccuracyForTestData(epoch, sess, test_error_value_list, test_accuracy_list)

    drawErrorValues(epoch_list, train_error_value_list, validation_error_value_list, test_error_value_list, test_accuracy_list)

    drawFalsePrediction(sess, 10)
            
    print("Optimization Finished!")

