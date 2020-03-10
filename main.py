import tensorflow as tf
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from numpy import genfromtxt
data = genfromtxt('Ed.csv', delimiter=',')
#
r=np.random.choice(np.linspace(0,50,51),51).astype(int)
data=data[r] #shuffled

train_example_num = 40
train_data = data[:train_example_num,1:]
train_labels = data[:train_example_num,0]
test_data = data[train_example_num:,1:]
test_labels = data[train_example_num:,0]

train_min = np.min(train_data,0)
train_data = train_data-train_min
test_data = test_data-train_min

train_max = np.max(train_data,0)
train_data = train_data/train_max
test_data = test_data/train_max




def weight_variable_xavier(shape):
    initial = tf.truncated_normal(shape, stddev=np.sqrt(1.0/(shape[0]) ))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lrelu(x,leak=0.1):
    return tf.maximum(x,leak*x)

#def get_batch(bsize):
#    batch=[0]*2
#    r=np.random.choice(np.linspace(0,train_example_num,train_example_num-1),bsize)
#    batch[0] = train_data[r]
#    batch[1] = train_labels[r]
#    return batch

    
    

x = tf.placeholder(tf.float32, shape=[None, 6]) # input placeholder
y_truth = tf.placeholder(tf.float32, shape=[None,1]) # output placeholder



# Network
w_fc1 = weight_variable_xavier([6,10]) # hidden layer 1 
b_fc1 = bias_variable([10])
h_fc1 = lrelu(tf.matmul(x,w_fc1)+b_fc1)

w_fc2 = weight_variable_xavier([10,10]) # # hidden layer 2
b_fc2 = bias_variable([10])
h_fc2 =  lrelu(tf.matmul(h_fc1,w_fc2)+b_fc2)

w_fc3 = weight_variable_xavier([10,1]) # output layer 
b_fc3 = bias_variable([1])
y_pred = tf.matmul(h_fc2,w_fc3)+b_fc3


MSE = tf.reduce_mean(tf.square(y_pred-y_truth)) # 

train_step = tf.train.AdamOptimizer(0.001).minimize(MSE)

# Init TF session
sess = tf.InteractiveSession()    
# Init TF variables
tf.global_variables_initializer().run()


#batch size
#bsize = 50 
iterations = 10000

for i in range(iterations):
    #batch = get_batch(bsize)
    train_step.run(feed_dict={x: train_data, y_truth: np.expand_dims(train_labels,1)})
    if i % 100==0:
        training_error = MSE.eval(feed_dict={x: train_data, y_truth: np.expand_dims(train_labels,1)})
        test_error = MSE.eval(feed_dict={x: test_data, y_truth: np.expand_dims(test_labels,1)})
        print('step %d, training error %g' % (i, training_error))
        print('test error %g' % test_error)
        #print(y_pred.eval(feed_dict={x: test_data}))
        
        

