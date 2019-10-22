import numpy as np
from sklearn import preprocessing

#load csv file into np array
raw_csv = np.loadtxt('Example_TF.csv', delimiter = ',')

#drop first column (customer id) and last column (target)
unscaled_inputs_all = raw_csv[:,1:-1]
#last column is the target that we want to predict (buy or not)
targets_all = raw_csv[:,-1]


#to prevent bias, we choose equal records that has target 0 (dont buy) and 1 (buy)
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

#delete excessive records in inputs and targets by row (axis=0)
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

#feature scaling
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

#shuffle data: make random index array then rearrange data by this order
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

#a simple example of how this shuffle works
#a = np.arange(5)
#np.random.shuffle(a)
#b =np.array([9,8,7,6,5])
#c = b[a]
#a,c

#split train-vaildation-test (0.8-0.1-0.1) data in both inputs and targets
samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

#save to npz file
np.savez('Audiobooks_data_train', inputs=train_inputs,targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs,targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs,targets=test_targets)

#load from npz file
npz = np.load('Audiobooks_data_train.npz')

train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz2 = np.load('Audiobooks_data_validation.npz')
validation_inputs = npz2['inputs'].astype(np.float)
validation_targets = npz2['targets'].astype(np.int)

npz3 = np.load('Audiobooks_data_test.npz')
test_inputs = npz3['inputs'].astype(np.float)
test_targets = npz3['targets'].astype(np.int)

#define neural network model
input_size = 10 #10 columns of customer parameter
output_size = 2 #buy or not(1 or 0)
hidden_layer_size = 20 #each hidden layer has 20 nodes

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax') #classification problem: buy or not
                            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model
batch_size = 100
max_epochs = 100
#number of epochs with no improvement after which training will be stopped
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)

model.fit(train_inputs, train_targets, batch_size = batch_size, epochs = max_epochs,
          callbacks = [early_stopping],
          validation_data = (validation_inputs, validation_targets), verbose = 2)

#evaluate test data with the model
test_loss, test_accuracy =  model.evaluate(test_inputs, test_targets)