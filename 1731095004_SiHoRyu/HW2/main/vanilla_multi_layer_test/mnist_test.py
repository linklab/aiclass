import datasource.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from scipy import stats

vector = np.array([1, 0, 4, 2, 3])
num_classes = 5
t = np.zeros((vector.size, num_classes))
for idx, row in enumerate(t):
    row[vector[idx]] = 1
print(t, end="\n\n")

data = mnist.MNIST_Data()
#data = mnist.Fashion_MNIST_Data()

print("data.train_input.shape:", data.train_input.shape, data.train_input.dtype)
print("data.train_target.shape:", data.train_target.shape, data.train_target.dtype)

print("data.validation_input.shape:", data.validation_input.shape, data.validation_input.dtype)
print("data.validation_target.shape:", data.validation_target.shape, data.validation_target.dtype)

print("data.test_input.shape:", data.test_input.shape, data.test_input.dtype)
print("data.test_target.shape:", data.test_target.shape, data.test_target.dtype)

print()

print(data.train_target[0])
print(data.train_target[1])
print(data.train_target[2])
print(data.train_target[2])

flatten_list = []
for figure in data.train_input:
    for pixel in figure:
        flatten_list.append(pixel)
print("data.train_input:", stats.describe(flatten_list))

# fig = plt.figure(figsize=(20, 5))
# for i in range(5):
#     idx = randint(0, data.num_train_data)
#     print(idx, ":", data.labels[np.argmax(data.train_target[idx])])
#     img = data.train_input[idx]
#     img = img.reshape(28, 28)
#     img.shape = (28, 28)
#     plt.subplot(150 + (i+1))
#     plt.imshow(img, cmap='gray')
#
# plt.show()
