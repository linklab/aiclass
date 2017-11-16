import datasource.mnist as mnist
import matplotlib.pyplot as plt
from random import randint


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
    print(idx, ":", data.labels[data.train_target[idx]])
    img = data.train_input[idx]
    img = img.reshape(28, 28)
    img.shape = (28, 28)
    plt.subplot(150 + (i+1))
    plt.imshow(img, cmap='gray')

plt.show()