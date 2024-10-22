import os
import tensorflow as tf
from array import array
from tensorflow.keras import layers
#from tensorflow.keras import mixed_precision
from tensorflow.python.ops.distributions.kullback_leibler import cross_entropy
import time
import matplotlib.pyplot as plt
from PIL import Image

INPUT_DATA_RES = 256  # 120
I2_DATA_RES = int(INPUT_DATA_RES / 2)  # 120 #128
I4_DATA_RES = int(INPUT_DATA_RES / 4)  # 60 #64
I8_DATA_RES = int(INPUT_DATA_RES / 8)  # 30 #31
I16_DATA_RES = int(INPUT_DATA_RES / 16)  # 15 #16
I32_DATA_RES = int(INPUT_DATA_RES / 32)  # 7.5 #8
I64_DATA_RES = int(INPUT_DATA_RES / 64)  # 3.75 #4
DISABLE_GPU_USAGE = False  # True  #

if (DISABLE_GPU_USAGE):
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


# tf.test.gpu_device_name()






# https://www.tensorflow.org/tutorials/generative/dcgan

def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(I64_DATA_RES * I64_DATA_RES * 256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((I64_DATA_RES, I64_DATA_RES, 256)))
	assert model.output_shape == (None, I64_DATA_RES, I64_DATA_RES, 256)  # Note: None is the batch size

	# the following is significantly adapted from
	nch = 512
	div = [2, 2, 4, 4, 8, 16]
	num_repeats = 1

	div = [nch / elem for elem in
		   div]  # .... <-- 7 layers  (div = [256, 256, 128,128,64,64,32] )(div = [120, 120, 60,60,30,30,15 ] )
	for n in div:
		for r in range(num_repeats + 1):
			# print('n=', n)
			model.add(layers.Conv2DTranspose(n, (5, 5), strides=(1, 1), padding='same',
											 use_bias=False))  # layer = Conv2DLayer(layer, num_filters=n, filter_size=h, pad='same', nonlinearity=linear)
			model.add(layers.BatchNormalization())  # layer = BatchNormLayer(layer)
			model.add(layers.LeakyReLU())  # layer = NonlinearityLayer(layer, nonlinearity=LeakyRectify(0.2))

		model.add(layers.UpSampling2D(size=(2, 2), interpolation='bilinear'))

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
	# print( model.output_shape)
	assert model.output_shape == (None, INPUT_DATA_RES, INPUT_DATA_RES, 1)  ###

	return model


def make_discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
							input_shape=[INPUT_DATA_RES, INPUT_DATA_RES, 1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
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
#'''

generator = make_generator_model()
# The discriminator is a CNN-based image classifier.
discriminator = make_discriminator_model()

# decision = discriminator(generated_image)  # negative values for fake images, positive values for real images
# print (decision)

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


gen_loss_history = []
disc_loss_history = []

# The Adam optimization algorithm is an extension of stochastic gradient descent.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Save checkpoints
#
# to save and restore models, which can be helpful in case a long running training task is interrupted.
#
checkpoint_dir = os.path.normpath('D:/LocalWorkDir/1800480/training_checkpoints')  # './training_checkpoints')#
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_v09")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
								 discriminator_optimizer=discriminator_optimizer,
								 generator=generator,
								 discriminator=discriminator)
# checkpoint manager, helpful for multiple checkpoints
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)

# Define the training loop
BUFFER_SIZE = 4000
BATCH_SIZE = 8
EPOCHS = 82
noise_dim = 100  # size of input noise
num_examples_to_generate = 16  # when previewing the 'results' after training; with pyplot

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

	# print('LOSS:\n Generator -', gen_loss, "\n Discriminator -", disc_loss)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

	return gen_loss, disc_loss


def train(dataset, epochs, loss_graph_enabled=True, use_checkpoint=True):
	# print('len', len(dataset))

	if use_checkpoint:
		print('loading checkpoints...')
		# restore from latest checkpoint if possible
		# print(manager.checkpoints)
		# print(manager.latest_checkpoint)
		checkpoint.restore(manager.latest_checkpoint)
	if manager.latest_checkpoint:
		print("[Checkpoint Manager]\t Restored from {}".format(manager.latest_checkpoint))
	else:
		print("[Checkpoint Manager]\t Initializing from scratch.")

	overall_start_time = time.time()
	for epoch in range(epochs):
		start = time.time()

		for image_batch in dataset:
			# print("\t\ttraining image batch...")
			if loss_graph_enabled:
				gen_loss, disc_loss = train_step(image_batch)
				gen_loss_history.append(gen_loss)
				disc_loss_history.append(disc_loss)
			else:
				train_step(image_batch)

		# Produce images for the GIF as you go
		##display.clear_output(wait=True)

		# Save the model every X epochs
		if (epoch + 1) % 50 == 0:
			manager.save()
		if (epoch + 1) % 15 == 0:
			# print('LOSS:', loss.numpy())
			generate_and_save_images(generator,
									 epoch + 1,
									 seed)

		print('Epoch  {} ({} batches) took {} min {} sec'.format(epoch + 1, len(dataset),
																 int((time.time() - start) / 60.0),
																 (time.time() - start) % 60))
	print('Training for {} Epochs took {} sec'.format(epochs, time.time() - overall_start_time))

	if input('would you like to save? type \'y\': ') == 'y':
		print('Saving checkpoint...')
		manager.save()

	# Generate after the final epoch
	##display.clear_output(wait=True)
	generate_and_save_images(generator,
							 epochs,
							 seed)
	plt.show()
	# generator
	plt.plot(gen_loss_history)
	plt.xlabel('Batch #')
	plt.ylabel('Generator Loss [entropy]')
	plt.show()
	# discriminator
	plt.plot(disc_loss_history)
	plt.xlabel('Batch #')
	plt.ylabel('Discriminator Loss [entropy]')
	plt.show()


def generate_and_save_images(model, epoch, test_input):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(10, 10))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predictions[i, :, :, 0], cmap='gray', interpolation='none', resample=False)  # * 127.5 + 127.5
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d} (v09.6).png'.format(epoch))


# plt.show()


# using the GAN's trained generator, create a heightmap and export/save to a readable file.
def generate_heightmap(model=generator, input_noise=tf.random.normal([1, noise_dim])):
	print('Generating...')
	# restore from latest checkpoint if possible
	checkpoint.restore(manager.latest_checkpoint)
	if manager.latest_checkpoint:
		print("[Checkpoint Manager]\t Using trained model {}.".format(manager.latest_checkpoint))
	else:
		print("[Checkpoint Manager]\t No trained model found.")

	print(int(input_noise[0, 0] * 1000))

	generated_heightmap = model(input_noise, training=False)
	print(generated_heightmap.shape)
	decision = discriminator(generated_heightmap)  # negative values for fake images, positive values for real images
	print('\t\tthe discriminator gives this a score of:', decision[0, 0])

	plt.imshow(generated_heightmap[0, :, :, 0], cmap='gray', interpolation='none', resample=False)  #
	plt.show()
	# plt.imshow(generated_heightmap[0, :, :, 0], cmap='gray', interpolation='none',vmin=0, vmax=1, resample=False) #
	#plt.axis('off')  # remove axes

	user_input = input('Would you like to save the image? [y/n] ')
	save = (user_input == 'y' or user_input == 'Y')
	
	if save:
		filename = None
		user_input = input('Please enter a file name, including .png at the end: ')
		if user_input[-4:] == '.png':
			filename = user_input
		normalise = input('Normalise values? [y/n]: ') == 'y'

		import numpy
		output_list = numpy.array(list(generated_heightmap[0, :, :, 0]))  # .astype(tf.uint32)
		# normalise values in array (stretch to 0 and 1) if chosen. then multiply all values to fit range (2^16)
		output_array = (numpy.multiply((output_list - output_list.min())/ (output_list.max() - output_list.min()) if normalise else output_list, 65535)).astype(numpy.uint32)
		output = Image.fromarray(output_array, 'I')

		if filename is None:
			# save the array as a PNG image using PIL
			name = 'heightmap_{}.png'.format(int(input_noise[0, 0] * 1000))
			output.save(name)
			print('> Saved as', name)
		else:
			output.save(filename)
			print('Saved.')

	# plt.show()
	print('done')


#
## fix checkpoint error/warnings (Exception ignored)


# Train the model
# Call the train() method defined above to train the generator and
# discriminator simultaneously.
# It's important that the generator and discriminator do not overpower
# each other (e.g. that they train at a similar rate).
#

def train_from_files(epochs=200):
	print('\n\tInput T to train... \n or E or G...')
	user_input = input()

	while user_input == 'g' or user_input == 'G':
		tf.random.set_seed(int(time.time()))
		generate_heightmap(model=generator, input_noise=tf.random.normal([1, noise_dim]))
		user_input = input()

	#print(hgt_filenames)
	#heightmap_tensors = [OpenAndReadHeightmap(name) for name in hgt_filenames]
	f_names = ['Heightmaps/all_hgts/G46/N24E090.hgt','Heightmaps/all_hgts/G46/N24E091.hgt']#['Heightmaps/dem_tif_n60w180/n60w155_dem.tif', 'Heightmaps/dem_tif_n60w180/n60w160_dem.tif', 'Heightmaps/dem_tif_n60w180/n65w180_dem.tif']

	heightmap_tensors = [OpenAndReadHeightmap(name) for name in f_names]
	train_dataset = tf.data.Dataset.from_tensor_slices(heightmap_tensors)


	if user_input == 't' or user_input == 'T':
		train(train_dataset, epochs)
	elif user_input == 'e' or user_input == 'E':
		print('\n\tInput the number of EPOCHS to train.')
		#epochs = int(input())
		train(train_dataset, int(input())) #'''

##
##
