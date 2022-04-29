import heightmap_gan as gan
from array import array
#EPOCHS = 100

#hgt_filenames = []
#heightmap_directory = 'Heightmaps/L32/'
#for latitude in range(44,48):
#    for longditude in range(7,10):
#        hgt_filenames.append(heightmap_directory + 'N' + str(latitude) + f'E{longditude:03}' + '.hgt')
def GetFilenames(directory = 'Heightmaps/'):
    filenames = []
    import os

    for root, dirs, files in os.walk(directory):
        for filename in files:
            #print(root + '/' + filename)
            #print(root + filename)
            filenames.append(root + '/' + filename)

    return filenames

def OpenAndReadHeightmap(filename, preview_data=False, analysing_data=False):
    if preview_data or analysing_data:
        print("Loading Heightmap", filename) #

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
        #print('.hgt')
        data = gan.plt.imread(filename)
        #print(data[:500])
    else:
        print('UNKNOWN FILE FORMAT. Sorry')
        return

    from numpy import sqrt

    if len(data) == 1201 * 1201:
        input_resolution = int(sqrt(len(data)))
        nearest_slicable_resolution = gan.INPUT_DATA_RES * int(input_resolution/gan.INPUT_DATA_RES)
        nsr = nearest_slicable_resolution
        #reduce the array to NSRxNSR - delete the extra rows/columns.
        amount_to_crop = input_resolution - nsr

        #'''
        del data[nsr * input_resolution:]
        col_to_delete = input_resolution
        for column in range(amount_to_crop):
            del data[::col_to_delete]
            col_to_delete -= 1
        #'''
        #gan.tf.keras.layers.Cropping2D(cropping=(amount_to_crop))(data)

        # make a rank 1 tensor  (1D array) and fill the tensor with the heightmap data
        rank_1_tensor = gan.tf.convert_to_tensor(data)
        #print(rank_1_tensor.shape)
        # convert to rank 2 (2D array)
        rank_2_tensor = gan.tf.reshape(rank_1_tensor, [nsr, nsr, 1])

    else:
        # make a rank 2 tensor  (2D array) and fill the tensor with the heightmap data
        rank_2_tensor = gan.tf.convert_to_tensor(data)[:, :, 0]
        #print(rank_2_tensor.shape)#6000x6000 or 6000x6000x1

    if preview_data:
        print(filename,':  matrix:', rank_2_tensor.shape, )
        print('showing preview:')
        gan.plt.imshow(rank_2_tensor[:, :], cmap="terrain")  #viridis") #inferno") #
        gan.plt.show()



    # slice into [a hundred][or 2500] 120 by 120 sub-images
    sub_image_res = gan.INPUT_DATA_RES#300#120
    number_of_sub_images = int((len(rank_2_tensor[0]) / sub_image_res) ** 2)
    if preview_data: print('The data will be sliced into ', number_of_sub_images, ' sub-images of size ', sub_image_res, 'x',
                           sub_image_res, '.')
    array3D = [[[0 for k in range(sub_image_res)] for j in range(sub_image_res)] for i in range(number_of_sub_images)]

    rows_columns = int(sqrt(number_of_sub_images))

    for index in range(number_of_sub_images):
        row_index = (index % rows_columns) * sub_image_res
        column_index = int(index / rows_columns) * sub_image_res
        array3D[index] = rank_2_tensor[row_index:row_index + sub_image_res, column_index:column_index + sub_image_res]

    rank_3_tensor = gan.tf.convert_to_tensor(array3D)

    max_altitude = 8850.0  # normalise height values to be between 0 [0.000000001] and 1. use sqrt to create a larger deviation between common (low) heights (avoid negatives/NaN):
    if analysing_data: ## create a histogram by grouping all elevation values into 100 groups, rounded down
        histogram_tensor = gan.tf.cast(gan.tf.maximum(gan.tf.cast(rank_3_tensor, gan.tf.float32) / max_altitude, 1e-9) * 100, gan.tf.int32)
        histogram_tensor = gan.tf.reshape(histogram_tensor, [256*256*16])
        ##histogram = gan.tf.histogram_fixed_width(rank_3_tensor,[0,100],100,gan.tf.int32)
        return histogram_tensor#histogram

    rank_3_tensor = gan.tf.cast(gan.tf.sqrt(gan.tf.maximum(gan.tf.cast(rank_3_tensor, gan.tf.float32) / max_altitude, 1e-9)), gan.tf.float32)



    if preview_data:
        print(rank_3_tensor.shape)


        gan.plt.imshow(rank_3_tensor[0, :, :], cmap="inferno")  #terrain")  #viridis") #
        gan.plt.show()

        '''gan.plt.figure(figsize=(10, 10))  # set image dimensions in inches
        gan.plt.imshow(rank_3_tensor[0, :, :], cmap='gist_rainbow', interpolation='none',
                       resample=False)  # vmin=0, vmax=1,
        gan.plt.axis('off')  # remove axes
        gan.plt.show()'''


    '''# print the tensor row by row
        print('\n------\n'.join(['\n'.join([''.join(['{:5}'.format(item)
                                                     for item in row])
                                            for row in sub_image])
                                 for sub_image in rank_3_tensor[0:16]]))
        #'''

    if preview_data:
        print("...Finished Loading.")
    #else:
    #    print("|", end="")

    return rank_3_tensor


def analyse_dataset():
    histogram_list = [0 for _ in range(100)]
    count = 0
    for name in GetFilenames('Heightmaps/all_hgts/'):
        #bucket_array = OpenAndReadHeightmap(name, analysing_data=True)
        #for bucket in range(100):
        #    histogram_list[bucket] += gan.tf.reduce_sum(gan.tf.cast(gan.tf.equal(bucket_array, bucket), gan.tf.int32)).numpy() #bucket_array.count(bucket)
        #print('100:',histogram_tensors[99])
        count += 1

    print(count, 'total source DEM files')
    count = count * 256 * 256 * 16
    print('count:', count)
    print(histogram_list)
    print('normalised:')
    histogram_percentage = [100* bucket/count for bucket in histogram_list]
    print(histogram_percentage)


dataset_path = 'D:/LocalWorkDir/1800480/training_dataset'
no_saved_dataset = True ###False ###

def train_from_input(EPOCHS=gan.EPOCHS, viewInputs=False):

    print('\n\tInput T to train ('+str(EPOCHS)+'); E to set the number of epochs; C to load a checkpoint ...')
    user_input = input('  -->')

    print('loading dataset from files...')
    if no_saved_dataset:
        heightmap_tensors = [OpenAndReadHeightmap(name, preview_data=viewInputs) for name in GetFilenames('Heightmaps/all_hgts/')] #dem_n30e000/')] #
        #f_names = ['Heightmaps/all_hgts/G46/N27E090.hgt','Heightmaps/all_hgts/G46/N24E091.hgt']
        #heightmap_tensors = [OpenAndReadHeightmap(name,preview_data=True) for name in f_names]
        data_size = len(heightmap_tensors) * len(heightmap_tensors[0])

        hmt = gan.tf.convert_to_tensor(heightmap_tensors)
        hmt = gan.tf.reshape(hmt, [data_size, 256, 256, 1])
        hmt = gan.tf.random.shuffle(hmt)
        #print(hmt.shape)

        train_dataset = gan.tf.data.Dataset.from_tensor_slices(hmt)
        train_dataset = train_dataset.shuffle(gan.BUFFER_SIZE, reshuffle_each_iteration=True).batch(gan.BATCH_SIZE).prefetch(4).cache()


        #gan.tf.data.experimental.save(train_dataset, dataset_path)  # save the dataset to a file


        print('\n                                --------------\n'
              'total size of training dataset: ', hmt.shape ,'images\n' #
              '                ~ comprised of: ', len(train_dataset),'batches\n'
              '                ~           of: ', gan.BATCH_SIZE,'images\n'
              '                                --------------')
    else:
        print('loaded tensorflow dataset - ', dataset_path)
        # load saved dataset
        train_dataset = gan.tf.data.TFRecordDataset(dataset_path)

    if user_input == 't' or user_input == 'T':
        gan.train(train_dataset, EPOCHS)
    elif user_input == 'e' or user_input == 'E':
        print('\n\tInput the number of EPOCHS to train.')
        EPOCHS = int(input())
        gan.train(train_dataset, EPOCHS)
    elif user_input == 'c' or user_input == 'c':
        print('\n\tInput the number of EPOCHS to train.')
        EPOCHS = int(input())
        gan.train(train_dataset, EPOCHS, use_checkpoint=True)

    return True

def Main():

    while True:

        print('\n\t\tCMP400\t-\tDeep Convolutional Generative Adversarial Network'
              '\n\t----------------------------------------------------------------'
              '\nPlease enter a command (key) -'
              '\n\tG: Generate A Heightmap'
              '\n\tT: Train from Dataset'
              '\n\tA: Analyse Dataset'
              '\n\tO: Other'
              '\n\tQ: Quit'
              '\n\t...')
        user_input = input('\n\t--> ')

        if user_input == 't' or user_input == 'T':
            train_from_input()
        elif user_input == 'tv' or user_input == 'TV':
            train_from_input(viewInputs=True)
        elif user_input == 'g' or user_input == 'G':
            gan.tf.random.set_seed(int(gan.time.time()))
            gan.generate_heightmap(input_noise=gan.tf.random.normal([1, gan.noise_dim]))
        elif user_input == 'a' or user_input == 'A':
            analyse_dataset()
        elif user_input == 'o' or user_input == 'O':
            gan.train_from_files()
        elif user_input == 'q':
            break
        else:
            break

Main()