import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#show tensorflow version. tensorflow 2.0 is now combined with keras
print(tf.__version__)

#use numpy to generate a (10000,1) array with uniformly random data from -10 to 10
observation = 10000
xs = np.random.uniform(-10,10,(observation,1))
zs = np.random.uniform(-10,10,(observation,1))

#stack 1-D arrays as columns into a 2-D array
generated_inputs = np.column_stack((xs,zs))

#add noise and set target to be 2*xs - 3zs + 5 + noise
noise = np.random.uniform(-1,1,(observation,1))
generated_targets = 2*xs - 3*zs + 5 + noise

#save np arrays into npz file. TF_intro is filename, and inputs and targets are array tags
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)
#load from npz. now generated_inputs becomes training_data['input']
training_data = np.load('TF_intro.npz')

#define the neural network model
input_size = 2 #xs,zs
output_size = 1 #y
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(output_size,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
                                                  bias_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)
                                                 )
                            ])
custom_opimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=custom_opimizer, loss='huber_loss')
#model.compile(optimizer='sgd', loss='mean_squared_error') is okay too

#actually train the model
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)

#show result weights
model.layers[0].get_weights()

#do prediction with trained model. we should use test data but here we use same data
a = model.predict_on_batch(training_data['inputs']).numpy().round(1)
b = training_data['targets'].round(1)
#plot the result and we expect see a y=x line
plt.plot(a,b)