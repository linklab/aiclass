
# coding: utf-8

# # numpy example

# In[3]:

#numpy example
import numpy as np
a = np.zeros((2,2)); b = np.ones((2,2))
np.sum(b,axis=1)
a.shape
np.reshape(a,(1,4))


# # tensorflow begin

# In[78]:

#tensorflow begin
import tensorflow as tf
sess = tf.Session()
a = tf.zeros((2,2)); b = tf.ones((2,2))
sess.run(tf.reduce_sum(b,axis = 1))
a.get_shape()
sess.run(tf.reshape(a,(1,4)))


# # sess.run()

# In[14]:

# sess.run()
sess = tf.Session()
a = np.zeros((2,2)); ta = tf.zeros((2,2))
print(a)
print(ta)
print(sess.run(ta))


# # computation graph & Tensorflow session

# In[19]:

# computation graph & Tensorflow session
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b
sess = tf.Session()
print(sess.run(c))
with tf.Session() as sess:
    print(sess.run(c))
    print(c.eval())


# # tensorflwo variables

# In[22]:

#tensorflwo variables
w = tf.Variable(tf.zeros((2,2)), name ="weight")
with tf.Session() as sess:
    print(sess.run(w))


# In[25]:

#tensorflwo variables
w = tf.Variable(tf.random_normal([5,2], stddev =0.1),name = "weight")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))


# # updating Variables

# In[27]:

#updating Variables
state = tf.Variable(0, name = "conunter")
new_value = tf.add(state,tf.constant(1))
update = tf.assign(state, new_value)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# # Fetching Varialbes

# In[38]:

#Fetching Varialbes
x1 = tf.constant(1)
x2 = tf.constant(2)
x3 = tf.constant(3)
temp = tf.add(x2, x3)
mul = tf.multiply(x1, temp)

with tf.Session() as sess:
    result1, result2 = sess.run([mul, temp])
    print(result1, result2)


# # tensorflow placeholder

# In[39]:

#tensorflow placeholder
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a,b)
mul = tf.multiply(a,b)
with tf.Session() as sess:
    print(sess.run(add, feed_dict = {a:2, b:3}))
    print(sess.run(mul, feed_dict = {a:2, b:3}))    


# In[42]:

#tensorflow placeholder
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)


# In[44]:

#tensorflow placeholder
import numpy as np

matrix1 = tf.placeholder(tf.float32, [1, 2])
matirx2 = tf.placeholder(tf.float32, [2, 1])
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    mv1 = np.array([[3., 3.]])
    mv2 = np.array([[2.],[2.]])
    result = sess.run(product, feed_dict = {matrix1: mv1, matrix2: mv2})
    print(result)


# # Example - MNIST with MLP

# In[56]:

#Example - MNIST with MLP
import tensorflow as tf
# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

learning_rate = 0.001
max_steps = 15000
batch_size = 128

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

def MLP(inputs):
    w_1 = tf.Variable(tf.random_normal([784, 256]))
    b_1 = tf.Variable(tf.zeros([256]))
    
    w_2 = tf.Variable(tf.random_normal([256, 256]))
    b_2 = tf.Variable(tf.zeros([256]))
    
    w_out = tf.Variable(tf.random_normal([256,10]))
    b_out = tf.Variable(tf.zeros([10]))
    
    h_1 = tf.add(tf.matmul(inputs, w_1),b_1)
    h_1 = tf.nn.relu(h_1)
    
    h_2 = tf.add(tf.matmul(h_1,w_2),b_2)
    h_2 = tf.nn.relu(h_2)
    
    out = tf.add(tf.matmul(h_2,w_out),b_out)
    
    return out

net = MLP(x)

#define loss and opimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net,y))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

#initializing the variables
init_op = tf.global_varialbes_initializer()

sess = tf.Session()
sess.run(init_op)

# train model
for step in range(max_steps):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    
    loss = sess.run([opt,loss_op],feed_dict = {x:batch_x, y:batch_y})
    
    if (step+1) % 1000 == 0:
        print("[{}/{}] loss:{:.3f}".format(step+1,max_steps,loss))
        print("Optimization Finished!")
        
# test model
correct_prediction = tf.equal(tf.argmax(net,1),tf.argmax(y,1))

#calculate accuracy
accuracy = tf.reduce_mean(tf.cost(corrent_prediction,tf.float32))
print("Train accuracy: {:.3f}". format(sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels})))
print("Train accuracy: {:.3f}". format(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})))


# # tf.Variable_scope()

# In[58]:

#tf.Variable_scope()
var1 = tf.Variable([1],name="var")
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        var2 = tf.Variable([1],name="var")
        var3 = tf.Variable([1],name="var")
        
print("var1 : {}".format(var1.name))
print("var2 : {}".format(var2.name))
print("var3 : {}".format(var3.name))


# # tf.get_variable()

# In[ ]:

#tf.get_variable()
#1
var1 = tf.Variable([1],name="var")
with tf.variable_scope("foo"):
    with tf.variable_scope("bar") as scp:
        var2 = tf.Variable([1],name="var")
        scp.reuse_variables()
        var3 = tf.Variable([1],name="var")
        
print("var1 : {}".format(var1.name))
print("var2 : {}".format(var2.name))
print("var3 : {}".format(var3.name))

#2
var1 = tf.get_variable("var", [1])
with tf.variable_scope("foo"):
    with tf.variable_scope("bar") as scp:
        var2 = tf.get_variable("var", [1])
        scp.reuse_variables()
        var3 = tf.get_variable("var", [1])
        
print("var1 : {}".format(var1.name))
print("var2 : {}".format(var2.name))
print("var3 : {}".format(var3.name))

#Parameter sharing
with tf.variable_scope("foo"):
    with tf.variable_scope("bar") as scp:
        var1 = tf.get_variable("var", [1])
        scp.reuse_variables()
        var2 = tf.get_variable("var", [1])
        
with tf.variable_scope("bar",reuse = True):
        var3 = tf.get_variable("var", [1])
        
print("var1 : {}".format(var1.name))
print("var2 : {}".format(var2.name))
print("var3 : {}".format(var3.name))


# # Wrappers

# In[81]:

# Wrappers
from tensorflow.contrib.layers import variance_scaling_initializer 
he_init = variance_scaling_initializer() 

def conv(bottom, num_filter, ksize=3, stride=1, padding="SAME", scope=None): 
    bottom_shape = bottom.get_shape().as_list()[3] 
    
    with tf.variable_scope(scope or "conv"): 
        W = tf.get_variable("W", [ksize, ksize, bottom_shape, num_filter], initializer=he_init) 
        b = tf.get_variable("b", [num_filter], initializer=tf.constant_initializer(0))
        
        x = tf.nn.conv2d(bottom, W, strides=[1, stride, stride, 1], padding=padding) 
        x = tf.nn.relu(tf.nn.bias_add(x, b)) 
        
    return x

def maxpool(bottom, ksize=2, stride=2, padding="SAME", scope=None):
    
    with tf.variable_scope(scope or "maxpool"): 
        pool = tf.nn.max_pool(bottom, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding) 
    
    return pool

def fc(bottom, num_dims, scope=None): 
    
    bottom_shape = bottom.get_shape().as_list() 
    if len(bottom_shape) > 2: 
        bottom = tf.reshape(bottom, [-1, reduce(lambda x, y: x*y, bottom_shape[1:])]) 
        bottom_shape = bottom.get_shape().as_list() 
        
    with tf.variable_scope(scope or "fc"): 
        W = tf.get_variable("W", [bottom_shape[1], num_dims], initializer=he_init) 
        b = tf.get_variable("b", [num_dims], initializer=tf.constant_initializer(0)) 
        out = tf.nn.bias_add(tf.matmul(bottom, W), b) 
    
    return out 

    def fc_relu(bottom, num_dims, scope=None): 
        
        with tf.variable_scope(scope or "fc"):
            out = fc(bottom, num_dims, scope="fc") 
            relu = tf.nn.relu(out) 
            
        return relu


# # All Together

# In[84]:

# All Together
keep_prob = tf.placeholder(tf.float32, None) 

def conv_net(x, keep_prob): 
    x = tf.reshape(x, shape=[-1, 28, 28, 1]) 
    
    conv1 = conv(x, 32, 5, scope="conv_1") 
    conv1 = maxpool(conv1, scope="maxpool_1") 
    conv2 = conv(conv1, 64, 5, scope="conv_2") 
    conv2 = maxpool(conv2, scope="maxpool_2")
    
    fc1 = fc_relu(conv2, 1024, scope="fc_1") 
    fc1 = tf.nn.dropout(fc1, keep_prob) 
    
    out = fc(fc1, 10, scope="out") 
    return out


# # Layers

# In[107]:

# Layers

import tensorflow as tf 
slim = tf.contrib.slim 

input = ... 
with tf.name_scope('conv1_1') as scope: 
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights') 
    conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME') 
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases') 
    bias = tf.nn.bias_add(conv, biases) 
    conv1 = tf.nn.relu(bias, name=scope) 

# 1. simple network generation with slim 
net = ... 
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool3')

# 1. cleaner by repeat operation: 
net = ... 
net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3') 
net = slim.max_pool(net, [2, 2], scope='pool3')

# 2. Verbose way: 
x = slim.fully_connected(x, 32, scope='fc/fc_1') 
x = slim.fully_connected(x, 64, scope='fc/fc_2') 
x = slim.fully_connected(x, 128, scope='fc/fc_3')
                    
# 2. Equivalent, TF-Slim way using slim.stack: 
slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')


# # argscope

# In[91]:

# argscope

he_init = slim.variance_scaling_initializer() 
xavier_init = slim.xavier_initializer() 

with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                    activation_fn=tf.nn.relu, weights_initializer=he_init, 
                    weights_regularizer=slim.l2_regularizer(0.0005)): 
    
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'): 
        net = slim.conv2d(inputs, 64, [11, 11], 4, scope='conv1') 
        net = slim.conv2d(net, 256, [5, 5], weights_initializer=xavier_init, scope='conv2') 
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')


# # Losses

# In[100]:

# Losses
# Define the loss functions and get the total loss. 
loss1 = slim.losses.softmax_cross_entropy(pred1, label1) 
loss2 = slim.losses.mean_squared_error(pred2, label2) 

# The following two lines have the same effect: 
total_loss = loss1 + loss2 
slim.losses.get_total_loss(add_regularization_losses=False) 

# If you want to add regularization loss 
reg_loss = tf.add_n(slim.losses.get_regularization_losses())
total_loss = loss1 + loss2 + reg_loss 

# or 
total_loss = slim.losses.get_total_loss()


# # Save/Restore

# In[102]:

# Save/Restore
def save(self, ckpt_dir, global_step=None): 
    if self.config.get("saver") is None: 
        self.config["saver"] = \ 
            tf.train.Saver(max_to_keep=30)
            
    saver = self.config["saver"]
    sess = self.config["sess"] 
    
    dirname = os.path.join(ckpt_dir, self.name) 
    
    if not os.path.exists(dirname): 
        os.makedirs(dirname) 
    saver.save(sess, dirname, global_step)
    
    def load_latest_checkpoint(self, ckpt_dir, exclude=None): 
        path = tf.train.latest_checkpoint(ckpt_dir) 
        if path is None: 
            raise AssertionError("No ckpt exists in {0}.".format(ckpt_dir)) 
            print("Load {} save file".format(path)) 
            self._load(path, exclude) 
    
    def load_from_path(self, ckpt_path, exclude=None): 
        self._load(ckpt_path, exclude) 
        
    def _load(self, ckpt_path, exclude): 
        init_fn = slim.assign_from_checkpoint_fn(ckpt_path, slim.get_variables_to_restore(exclude=exclude), ignore_missing_vars=True) 
        init_fn(self.config["sess"])


# # Vgg_16

# In[105]:

def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
        
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
        
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points

vgg_16.default_image_size = 224


# In[104]:

X = tf.placeholder(tf.float32, [None, 224, 224, 3], name="X") 
y = tf.placeholder(tf.int32, [None, 8], name="y") is_training = tf.placeholder(tf.bool, name="is_training") 
with slim.arg_scope(vgg.vgg_arg_scope()): net, end_pts = vgg.vgg_16(X, is_training=is_training,num_classes=1000) 
    with tf.variable_scope("losses"): 
        cls_loss = slim.losses.softmax_cross_entropy(net, y) 
        reg_loss = tf.add_n(slim.losses.get_regularization_losses()) 
        loss_op = type_loss + reg_loss 
    
    with tf.variable_scope("opt"): 
        opt = tf.train.AdamOptimizer(0.001).minimize(loss_op) 
    
    self.load_from_path(ckpt_path=VGG_PATH, exclude=["vgg_16/fc8"]) ...


# # Tensor Board & slim example

# In[110]:

#Tensor Board & slim example

import tensorflow as tf 
slim = tf.contrib.slim 

# Import MINST data 
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

max_steps = 10000 
batch_size = 128 
lr = 0.001 
keep_prob = 0.5 
weight_decay = 0.0004 
logs_path = "/tmp/tensorflow_logs/example" 

def my_arg_scope(is_training, weight_decay):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, 
                        weights_regularizer=slim.l2_regularizer(weight_decay), 
                        weights_initializer=slim.variance_scaling_initializer(), 
                        biases_initializer=tf.zeros_initializer, 
                        stride=1, padding="SAME"): 
        with slim.arg_scope([slim.dropout], is_training=is_training) as arg_sc: 
            return arg_sc
        
def my_net(x, keep_prob, outputs_collections="my_net"): 
    x = tf.reshape(x, shape=[-1, 28, 28, 1]) 
    
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=outputs_collections): 
        net = slim.conv2d(x, 64, [3, 3], scope="conv1") 
        net = slim.max_pool2d(net, [2, 2], scope="pool1") 
        net = slim.conv2d(net, 128, [3, 3], scope="conv2") 
        net = slim.max_pool2d(net, [2, 2], scope="pool2") 
        net = slim.conv2d(net, 256, [3, 3], scope="conv3") 
        
        # global average pooling 
        net = tf.reduce_mean(net, [1, 2], name="pool3", keep_dims=True) 
        net = slim.dropout(net, keep_prob, scope="dropout3") 
        net = slim.conv2d(net, 1024, [1, 1], scope="fc4") 
        net = slim.dropout(net, keep_prob, scope="dropout4") 
        net = slim.conv2d(net, 10, [1, 1], activation_fn=None, scope="fc5") 
        
    end_points = \ 
        slim.utils.convert_collection_to_dict(outputs_collections) 
    return tf.reshape(net, [-1, 10]), end_points

    x = tf.placeholder(tf.float32, [None, 784]) 
    y = tf.placeholder(tf.float32, [None, 10]) 
    is_training = tf.placeholder(tf.bool) 
    
    with slim.arg_scope(my_arg_scope(is_training, weight_decay)): 
        net, end_pts = my_net(x, keep_prob) 
        pred = slim.softmax(net, scope="prediction") 
        
    with tf.variable_scope("losses"): 
        cls_loss = slim.losses.softmax_cross_entropy(net, y) 
        reg_loss = tf.add_n(slim.losses.get_regularization_losses()) 
        loss_op = cls_loss + reg_loss 
        
    with tf.variable_scope("Adam"): 
        opt = tf.train.AdamOptimizer(lr) 
        
        # Op to calculate every variable gradient 
        grads = tf.gradients(loss_op, tf.trainable_variables()) 
        grads = list(zip(grads, tf.trainable_variables())) 
        
        # Op to update all variables according to their gradient 
        apply_grads = opt.apply_gradients(grads_and_vars=grads) 
        
    with tf.variable_scope("accuracy"): 
        correct_op = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1)) 
        acc_op = tf.reduce_mean(tf.cast(correct_op, tf.float32))

    # Create a summary to monitor loss and accuracy 
    summ_loss = tf.summary.scalar("loss", loss_op) 
    summ_acc = tf.summary.scalar("accuracy_test", acc_op) 
    
    # Create summaries to visualize weights and grads 
    for var in tf.trainable_variables(): tf.summary.histogram(var.name, var, collections=["my_summ"]) 
        for grad, var in grads: tf.summary.histogram(var.name + "/gradient", grad, collections=["my_summ"]) 
            
            summ_wg = tf.summary.merge_all(key="my_summ") 
            
            sess = tf.Session() 
            sess.run(tf.global_variables_initializer()) 
            summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
            
    for step in range(max_steps): batch_X, batch_y = mnist.train.next_batch(batch_size) 
        _, loss, plot_loss, plot_wg = sess.run([apply_grads, loss_op, summ_loss, summ_wg], 
                                               feed_dict={x: batch_X, y: batch_y, is_training: True}) 
        summary_writer.add_summary(plot_loss, step) 
        summary_writer.add_summary(plot_wg, step) 
        
        if (step+1) % 100 == 0: 
            plot_acc = sess.run(summ_acc, feed_dict={x: mnist.test.images, y: mnist.test.labels, is_training: False}) 
            summary_writer.add_summary(plot_acc, step) 
            print("Optimization Finished!")
            
            test_acc = sess.run(acc_op, feed_dict={x: mnist.test.images, y: mnist.test.labels, is_training: False}) 
            print("Test accuracy: {:.3f}".format(test_acc))


# In[ ]:



