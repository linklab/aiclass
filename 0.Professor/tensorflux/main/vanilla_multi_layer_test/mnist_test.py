import datasource.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
from random import randint

vector = np.array([1, 0, 4, 2, 3])
num_classes = 5
t = np.zeros((vector.size, num_classes))
for idx, row in enumerate(t):
    row[vector[idx]] = 1
print(t, end="\n\n")

#data = mnist.MNIST_Data()
data = mnist.Fashion_MNIST_Data()

print("data.train_input.shape:", data.train_input.shape)
print("data.train_target.shape:", data.train_target.shape)

print("data.validation_input.shape:", data.validation_input.shape)
print("data.validation_target.shape:", data.validation_target.shape)

print("data.test_input.shape:", data.test_input.shape)
print("data.test_target.shape:", data.test_target.shape)

print()

fig = plt.figure(figsize=(20, 5))
for i in range(5):
    idx = randint(0, data.num_train_data)
    print(idx, ":", data.labels[np.argmax(data.train_target[idx])])
    img = data.train_input[idx]
    img = img.reshape(28, 28)
    img.shape = (28, 28)
    plt.subplot(150 + (i+1))
    plt.imshow(img, cmap='gray')

plt.show()
