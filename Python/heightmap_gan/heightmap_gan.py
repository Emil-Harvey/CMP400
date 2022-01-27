import os
import tensorflow as tf
from array import array
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.python.ops.distributions.kullback_leibler import cross_entropy
import time
import matplotlib.pyplot as plt

DISABLE_GPU_USAGE = True  #False#

hgt_filenames = []
heightmap_directory = 'Heightmaps/L32/'
for latitude in range(44, 48):
	for longditude in range(7, 10):
		hgt_filenames.append(heightmap_directory + 'N' + str(latitude) + f'E{longditude:03}' + '.hgt')

if  (DISABLE_GPU_USAGE):
	try:
		# Disable all GPUS
		tf.config.set_visible_devices([], 'GPU')
		visible_devices = tf.config.get_visible_devices()
		for device in visible_devices:
			assert device.device_type != 'GPU'
	except:
		# Invalid device or cannot modify virtual devices once initialized.
		pass
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), "Physical,",
	  len(tf.config.list_logical_devices('GPU')), "Logical")


#def slice_into_subimages(arr, nrows, ncols): #  https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays


def OpenAndReadHeightmap(filename):
	data = 0

	print("Loading Heightmap ({})...".format(filename))
	# if file is a .hgt:
	if filename.endswith('.hgt'):
		f = open(filename, 'rb')
		format = 'h'  # h stands for signed short
		row_length = 1201
		column_length = row_length  # heightmap is square
		data = array(format)
		data.fromfile(f, row_length * column_length)
		data.byteswap()
		f.close()
	#else if file = tiff:
	elif filename.endswith('.tif'):
		#f = open(filename, 'rb')
		data = plt.imread(filename)

	else:
		print('UNKNOWN FILE FORMAT. Sorry')

	# --testing--
	#for value in data:
	#    print(value)
	# -----------
	rank_2_tensor = 0

	desired_row_length = 1200
	desired_column_length = desired_row_length

	if len(data) == 1201 * 1201:
		#reduce the array to 1200x1200 - delete the last row/column.
		del data[1200 * 1201:]
		del data[::1201]


		# make a rank 1 tensor  (1D array) and fill the tensor with the heightmap data
		rank_1_tensor = tf.convert_to_tensor(data)
		# convert to rank 2 (2D array)
		rank_2_tensor = tf.reshape(rank_1_tensor, [desired_row_length, desired_column_length, 1])

	else:
		# make a rank 1 tensor  (1D array) and fill the tensor with the heightmap data
		rank_1_tensor = tf.convert_to_tensor(data)
		# convert to rank 3 (3D array)
		from math import sqrt
		print('sqrt(',len(data), ') / desired_row_length =')
		print(sqrt(len(data)), ' / ', desired_row_length,' =')
		subdivisions = int(sqrt(len(data)) / desired_row_length)
		print(subdivisions)
		rank_3_tensor = tf.reshape(rank_1_tensor, [desired_row_length, desired_column_length,subdivisions, 1])
		# convert to rank 2 (2D array)
		rank_2_tensor = rank_3_tensor[:,:,0,:]#tf.reshape(rank_1_tensor, [desired_row_length, desired_column_length, 1])

	plt.imshow(rank_2_tensor[:, :], cmap="viridis")
	plt.show()

	# slice into a hundred 120 by 120 sub-images
	sub_image_res = 120
	array3D = [[[0 for k in range(sub_image_res)] for j in range(sub_image_res)] for i in range(100)]

	for index in range(100):
		row_index = (index % 10) * 120
		column_index = int(index / 10) * 120
		array3D[index] = rank_2_tensor[row_index:row_index + 120, column_index:column_index + 120]

	rank_3_tensor = tf.convert_to_tensor(array3D)

	#print(rank_3_tensor.shape)

	'''# print the tensor row by row
    print('\n------\n'.join(['\n'.join([''.join(['{:5}'.format(item)
                                                 for item in row])
                                        for row in sub_image])
                             for sub_image in rank_3_tensor]))
    #'''

	print("...Finished Loading.")
	#rank_3_tensor =tf.reshape(rank_3_tensor, [10, None,120,120,1])
	return rank_3_tensor


'''
# https://towardsdatascience.com/developing-a-dcgan-model-in-tensorflow-2-0-396bc1a101b2 # not used
'''


# https://www.tensorflow.org/tutorials/generative/dcgan

def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(30 * 30 * 256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((30, 30, 256)))
	assert model.output_shape == (None, 30, 30, 256)  # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 30, 30, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 60, 60, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 120, 120, 1)  ###

	return model


def make_discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
							input_shape=[120, 120, 1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model


# clear the session to ensure memory is freed up
tf.keras.backend.clear_session()
# let the GAN grow GPU memory allocation when more is needed
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# use mixed precision to speed-up training by using 16 bit floats instead of 32 bits where possible
# - less memory usage
# tensorflow automatically performs loss scaling (moving loss values
# closer to 1 to avoid rounding errors) with mixed_float16
# https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
''' only works on GPUs of 7.0 or higher
mixed_precision.set_global_policy('mixed_float16')
'''

generator = make_generator_model()
# The discriminator is a CNN-based image classifier.
discriminator = make_discriminator_model()

#decision = discriminator(generated_image)  # negative values for fake images, positive values for real images
#print (decision)

# Define loss functions and optimizers for both models.
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Discriminator loss
#
# This method quantifies how well the discriminator is able to distinguish
# real images from fakes. It compares the discriminator's predictions on
# real images to an array of 1s, and the discriminator's predictions on
# fake (generated) images to an array of 0s.
#
def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss


# Generator loss
#
# The generator's loss quantifies how well it was able to trick the
# discriminator. Intuitively, if the generator is performing well,
# the discriminator will classify the fake images as real (or 1).
# Here, compare the discriminators decisions on the generated images to
# an array of 1s.
#
def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)


loss_history = []

# The Adam optimization algorithm is an extension of stochastic gradient descent.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Save checkpoints
#
# to save and restore models, which can be helpful in case a long running training task is interrupted.
#
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_v02")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
								 discriminator_optimizer=discriminator_optimizer,
								 generator=generator,
								 discriminator=discriminator)
# checkpoint manager, helpful for multiple checkpoints
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)

# Define the training loop
BUFFER_SIZE = 60000
BATCH_SIZE = 3
EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 16

# this seed will be reused overtime (so it's easier
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# The training loop begins with generator receiving a random seed as input. That
# seed is used to produce an image. The discriminator is then used to classify
# real images (drawn from the training set) and fakes images (produced by the
# generator). The loss is calculated for each of these models, and the gradients
# are used to update the generator and discriminator.

# The `tf.function` annotation causes the function to be "compiled".
@tf.function
def train_step(images):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	#print('LOSS:\n Generator -', gen_loss, "\n Discriminator -", disc_loss)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

	return gen_loss


def train(dataset, epochs):
	# restore from latest checkpoint if possible
	checkpoint.restore(manager.latest_checkpoint)
	if manager.latest_checkpoint:
		print("[Checkpoint Manager]\t Restored from {}".format(manager.latest_checkpoint))
	else:
		print("[Checkpoint Manager]\t Initializing from scratch.")

	overall_start_time = time.time()
	for epoch in range(epochs):
		start = time.time()

		for image_batch in dataset:
			print("\t\ttraining image batch...")
			loss_history.append(train_step(image_batch))

		# Produce images for the GIF as you go
		##display.clear_output(wait=True)

		# Save the model every 15 epochs
		if (epoch + 1) % 300 == 0:
			manager.save()
		#print('LOSS:', loss.numpy())
		#generate_and_save_images(generator,
		#                         epoch + 1,
		#                         seed)

		print('Epoch  {}  took {} sec'.format(epoch + 1, time.time() - start))
	print('Training for {} Epochs took {} sec'.format(epochs, time.time() - overall_start_time))

	# Generate after the final epoch
	##display.clear_output(wait=True)
	generate_and_save_images(generator,
							 epochs,
							 seed)
	plt.show()
	plt.plot(loss_history)
	plt.xlabel('Batch #')
	plt.ylabel('Loss [entropy]')
	plt.show()


def generate_and_save_images(model, epoch, test_input):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4, 4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gist_rainbow')
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


#plt.show()


#using the GAN's trained generator, create a heightmap and export/save to a readable file.
def generate_heightmap(model=generator, input_noise=tf.random.normal([1, noise_dim]), save=False, filename=None):
	# restore from latest checkpoint if possible
	checkpoint.restore(manager.latest_checkpoint)
	if manager.latest_checkpoint:
		print("[Checkpoint Manager]\t Using trained model.".format(manager.latest_checkpoint))
	else:
		print("[Checkpoint Manager]\t No trained model found.")

	generated_heightmap = model(input_noise, training=False)
	#print(generated_heightmap.shape)
	plt.imshow(generated_heightmap[0, :, :, 0], cmap='gray', interpolation='none', resample=False)
	plt.show()
	#plt.imshow(generated_heightmap[0, :, :, 0], cmap='gray', vmin=0, vmax=1, interpolation='none',resample=False)
	#plt.show()
	if save:
		if filename is None:
			#save the array as a PNG image using PyPlot:
			name = 'heightmap_{}.png'.format(int(input_noise[0, 0] * 1000))
			name = 'heightmap_x.png'
			print(name)
			plt.savefig(name)

			#save the array as a .float (texture) file format
			name = 'heightmap_{}.float'.format(int(input_noise[0, 0] * 1000))
			print(name)
			exported_file = open(name, mode='wb')  #  create new/overwrite file
			generated_heightmap[0, :, :, 0].numpy().tofile(exported_file)
			exported_file.close()
		else:
			plt.savefig(filename)
	print('done')


#
# fix checkpoint error (Exception ignored)


# Train the model
# Call the train() method defined above to train the generator and
# discriminator simultaneously. Note, training GANs can be tricky.
# It's important that the generator and discriminator do not overpower
# each other (e.g., that they train at a similar rate).
#

def train_from_files(epochs=200):
	print(hgt_filenames)
	#heightmap_tensors = [OpenAndReadHeightmap(name) for name in hgt_filenames]
	f_names = ['Heightmaps/dem_tif_n60w180/n60w155_dem.tif', 'Heightmaps/dem_tif_n60w180/n60w160_dem.tif', 'Heightmaps/dem_tif_n60w180/n65w180_dem.tif']

	heightmap_tensors = [OpenAndReadHeightmap(name) for name in f_names]
	train_dataset = tf.data.Dataset.from_tensor_slices(heightmap_tensors)

	print('\n\tInput T to train...')
	user_input = input()  #'k'#
	if user_input == 't' or user_input == 'T':
		train(train_dataset, epochs)
	elif user_input == 'e' or user_input == 'E':
		print('\n\tInput the number of EPOCHS to train.')
		#epochs = int(input())
		train(train_dataset, int(input()))

	while user_input == 'g' or user_input == 'G':
		generate_heightmap(model=generator)
		user_input = input()
##
##
