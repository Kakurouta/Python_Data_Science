import numpy as np
import tensorflow as tf

#popular datasets can now be downloaded by tensorflow_datasets and are preprocessed
import tensorflow_datasets as tfds

#STEP1: loading data

#as_supervised: bool, if True, the returned tf.data.Dataset will have a 2-tuple structure (input, label) according to builder.info.supervised_keys. 
#If False, the default, the returned tf.data.Dataset will have a dictionary with all the features.
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
#mnist_dataset2 = tfds.load(name='mnist',as_supervised=False)

#get training and testing set
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

#STEP2: define the number of records of training, validation, andtest data

#get total number of records of training and testing set from mnist_info
#train:60000 -> train:54000 + validation:6000, then test:10000 records
num_validation_samples = 0.1*mnist_info.splits['train'].num_examples
num_test_samples = mnist_info.splits['test'].num_examples
#cast this number from float to int64
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
num_test_samples = tf.cast(num_test_samples, tf.int64)

#STEP3: feature scaling

#total 256 shades of gray. scale each pixel to [0,1]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

#apply training and testing data to scale function
scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

#STEP4: split training and validation data

BUFFER_SIZE = 10000
#shuffle the whole training data
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
#take first 10% of shuffled training data out as validation data
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
#take remaining 90% (skip first 10%) as training data
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

#STEP5: transfer data format to batch

#tensorflow trains in batch, thus transfer our data into batch
BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

#use next(iter(batch_data)) to get data of next batch. all validation data got in this case
validation_inputs, validation_targets = next(iter(validation_data))

#STEP6: define neural network model

#image size is 28*28 pixels
input_size = 784
#results from 0 to 9
output_size = 10
hidden_layer_size = 200

#activation function: relu: f(x)=max(x,0), softmax:probabilities, used for classification
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28,28,1)),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')    
                            ])

#optimizer: SGD -momentum on learning rate-> Adagrad -simplified computing-> Adadelta -special case, RMS on learning rate computing-> RMSprop
#           Adagrad(good at sparse gradient) + RMSprop(good at unstable target) =  Adam(Adaptive Moment Estimation)
#loss: sparse:lots of zeroes, categorical: classification problem (identify number 0-9), corssentropy: loss function
#metrics: used to judge the performance
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#STEP7: actually train the model!

NUM_EPOCHS = 5
#validation_steps = total_validation_samples // validation_batch_size
model.fit(train_data, epochs = NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), validation_steps=1, verbose=2)

#STEP8: input test data to see how good the model is
test_loss, test_accuracy = model.evaluate(test_data)