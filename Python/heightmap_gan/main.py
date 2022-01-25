import heightmap_gan as gan
from array import array
EPOCHS = 100

hgt_filenames = []
heightmap_directory = 'Heightmaps/L32/'
for latitude in range(44,48):
    for longditude in range(7,10):
        hgt_filenames.append(heightmap_directory + 'N' + str(latitude) + f'E{longditude:03}' + '.hgt')


def OpenAndReadHeightmap(filename):
    print("Loading Heightmap...")
    f = open(filename, 'rb')
    format = 'h'  # h stands for signed short
    row_length = 1201
    column_length = row_length  # heightmap is square
    data = array(format)
    data.fromfile(f, row_length * column_length)
    data.byteswap()
    f.close()

    #reduce the array to 1200x1200 - delete the last row/column.
    del data[1200*1201:]
    del data[::1201]
    new_row_length = 1200
    new_column_length = new_row_length

    # make a rank 1 tensor  (1D array) and fill the tensor with the heightmap data
    rank_1_tensor = gan.tf.convert_to_tensor(data)
    # convert to rank 2 (2D array)
    rank_2_tensor = gan.tf.reshape(rank_1_tensor, [new_row_length, new_column_length, 1])

    #plt.imshow(rank_2_tensor[:, :], cmap="viridis")
    #plt.show()

    # slice into a hundred 120 by 120 sub-images
    sub_image_res = 120
    array3D = [[[0 for k in range(sub_image_res)] for j in range(sub_image_res)] for i in range(100)]

    for index in range(100):
        row_index = (index % 10) * 120
        column_index = int(index / 10) * 120
        array3D[index] = rank_2_tensor[row_index:row_index +120, column_index:column_index +120]

    rank_3_tensor = gan.tf.convert_to_tensor(array3D)

    #print(rank_3_tensor.shape)

    print("...Finished Loading.")

    return rank_3_tensor


def TrainFromInput(EPOCHS=100):
    heightmap_tensors = [OpenAndReadHeightmap(name) for name in hgt_filenames]

    train_dataset = gan.tf.data.Dataset.from_tensor_slices(heightmap_tensors)


    print('\n\tInput T to train...')
    user_input = input()
    if user_input == 't' or user_input == 'T':
        gan.train(train_dataset, EPOCHS)
    elif user_input == 'e' or user_input == 'E':
        print('\n\tInput the number of EPOCHS to train.')
        EPOCHS = int(input())
        gan.train(train_dataset, EPOCHS)

    return True

def Main():
    print('\n\t\tCMP400\t-\tDeep Convolutional Generative Adversarial Network'
          '\n\t----------------------------------------------------------------'
          '\nPlease enter a command (key) -'
          '\n\tG: Generate A Heightmap'
          '\n\tT: Train from Dataset'
          '\n\tQ: Quit'
          '\n\t...')
    user_input = input()
    if user_input == 't' or user_input == 'T':
        TrainFromInput()
    elif user_input == 'g' or user_input == 'G':
        user_input = input('Would you like to save the image? [y/n]')
        Save = (user_input == 'y' or user_input == 'Y')
        if Save:
            user_input = input('Please enter a file name, including .png at the end:')
            if user_input[-4:] == '.png':
                gan.generate_heightmap(save=Save, filename=user_input)
            else:
                gan.generate_heightmap(save=Save)
        else:
            gan.generate_heightmap()


Main()