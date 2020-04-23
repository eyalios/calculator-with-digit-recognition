from PIL import Image
import os
import numpy as np
class calc_utils:
    batch_size = 128
    def __init__(self):
         self.size_of_vec = 784
         # set size of datas.
         self.total_data = 3000 * 10
         self.test_data = 500 * 10


    def get_data(self):
        # if data was already retrived and saved just load it.
        if os.path.isfile('C:/Users/USER/playground/test_set.npy'):
            self.X = np.load('x_set.npy')
            self.Y = np.load('y.npy')
            self.TestSet = np.load('test_set.npy')
            self.TestSetY = np.load('test_setY.npy')
        # open data sets train x,y and test x,y
        else:
            self.X = np.zeros((self.total_data, self.size_of_vec))
            self.Y = np.ones(self.total_data, dtype=np.int8)
            self.TestSet = np.zeros((self.test_data, self.size_of_vec))
            self.TestSetY = np.ones(self.test_data, dtype=np.int8)
            files_path = 'C:/Users/USER/Downloads/mnistasjpg/trainingSet/trainingSet/'
            x_index = 0
            train_index = 0
            # from each digit (10 digits) get samples.
            for i in range(10):
                cur_folder = files_path + str(i)
                counter = 0
                # for each digit get total_data/10 samples for train data
                for file in os.listdir(cur_folder):
                    if (counter < self.total_data / 10):
                        counter += 1
                        filename = os.fsdecode(file)
                        # preprocess images to fit model.
                        self.X[x_index] = self.get_image_and_vectorize(os.path.join(cur_folder, filename))
                        self.Y[x_index] = (i)
                        x_index += 1
                    # for each digit get test_data/10 samples.
                    elif (counter < (self.test_data + self.total_data) / 10):
                        counter += 1
                        filename = os.fsdecode(file)
                        # preprocess images to fit model.
                        self.TestSet[train_index] = self.get_image_and_vectorize(os.path.join(cur_folder, filename))
                        self.TestSetY[train_index] = (i)
                        train_index += 1
                    else:
                        break
            # transdorm data to fit model, and save.
            self.X = self.X.T
            self.TestSet = self.TestSet.T
            np.save('x_set', self.X)
            np.save('test_set', self.TestSet)
            np.save('y', self.Y)
            np.save('test_setY', self.TestSetY)


    def get_image_and_vectorize(self, file_path):
        """
        open image and make it a 1d binary vector
        :param file_path:
        :return:
        """
        img = np.array(Image.open(file_path).convert('1'))
        img_vectorize = img.reshape(28 * 28)
        return img_vectorize


    def get_image_and_prep(self,file_path):
        """
               open image and make it a 1d binary matrix
               :param file_path:
               :return:
        """
        img = np.array(Image.open(file_path).convert('1'))
        img = img.reshape(28,28,1)
        return img