from calc_utils import calc_utils
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import  numpy as np
from PIL import Image
import os
import random
import tensorflow
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D, Convolution2D
class cnn(calc_utils):
    def __init__(self):
        calc_utils.__init__(self)

    def create_network(self):
        """
        build computaion graph for cnn.
        :return:
        """
        #create model
        myNet = Sequential()
        #first layer, convolution 32 3X3 filter with stride 1, will result in 32 output images with size 28X28
        myNet.add((Conv2D(32, 3, strides = 1, activation='relu', input_shape=(28,28,1),padding='same')))
        print("1" + str(myNet.output_shape))
        #max pooling will output 32 14X14 arrays.
        myNet.add(MaxPooling2D((2,2)))
        print("2" + str(myNet.output_shape))
        #second conv layer, will output 32 14x14 arrays.
        myNet.add((Conv2D(32, 3,  strides = 1, activation='relu',padding='same')))
        print("3" + str(myNet.output_shape))
        #maxpool will output 32 7x7 arrays
        myNet.add(MaxPooling2D((2, 2)))
        print("3.5" + str(myNet.output_shape))
        #flatting the matrices before the fully connected layer
        myNet.add(Flatten())
        print("4" + str(myNet.output_shape))
        #fully connected layer to output 128 data points
        myNet.add(Dense(128, activation='relu'))
        print("5" + str(myNet.output_shape))
        #dropout to reduce overfit
        myNet.add(Dropout(0.5))
        print("6" + str(myNet.output_shape))
        #final layer to produce score vector for 10 classes.
        myNet.add(Dense(10, activation='softmax'))
        print("7" + str(myNet.output_shape))
        #compiling and saving model.
        myNet.compile(optimizer='Adam',loss = 'mean_squared_error',metrics=["accuracy"])
        print(myNet.output_shape)
        self.model = myNet



    def get_data(self):
        # if data was already retrived and saved just load it.
        if os.path.isfile('C:/Users/USER/playground/test_set_cnn.npy'):
            self.X = np.load('x_set_cnn.npy')
            self.Y = np.load('y_cnn.npy')
            self.TestSet = np.load('test_set_cnn.npy')
            self.TestSetY = np.load('test_setY_cnn.npy')
        # open data sets train x,y and test x,y
        else:
            self.X = np.zeros((self.total_data, 28,28,1))
            self.Y = np.ones(self.total_data, dtype=np.int8)
            self.TestSet = np.zeros((self.test_data, 28,28,1))
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
                        self.X[x_index] = self.get_image_and_prep(os.path.join(cur_folder, filename))
                        self.Y[x_index] = (i)
                        x_index += 1
                    # for each digit get test_data/10 samples.
                    elif (counter < (self.test_data + self.total_data) / 10):
                        counter += 1
                        filename = os.fsdecode(file)
                        # preprocess images to fit model.
                        self.TestSet[train_index] = self.get_image_and_prep(os.path.join(cur_folder, filename))
                        self.TestSetY[train_index] = (i)
                        train_index += 1
                    else:
                        break
            # transdorm data to fit model, and save.
            self.X = self.X.T
            self.TestSet = self.TestSet.T
            np.save('x_set_cnn', self.X)
            np.save('test_set_cnn', self.TestSet)
            np.save('y_cnn', self.Y)
            np.save('test_setY_cnn', self.TestSetY)

    def preapre_data(self):
         """
         preping data to fit the cnn requirments
          will also show one sample of data for sanity check, and print dims of inputs.
         :return: null
         """
         #get data in overriden method for this class.
         self.get_data()
         #adjust the tag vectors to fit keras cnn API
         self.Y = to_categorical(self.Y, 10)
         self.TestSetY = to_categorical(self.TestSetY, 10)
         #reshape input data to fit keras API
         self.X = self.X.T.reshape(self.X.T.shape[0],28,28,1)
         self.TestSet = self.TestSet.T.reshape(self.TestSet.T.shape[0],28,28,1)
         #sanity checks.
         plt.imshow(self.X[1,:,:,0])
         plt.show()
         print(self.X.shape)
         print(self.Y.shape)
         print(self.TestSet.shape)
         print(self.TestSetY.shape)

    def train(self):
        """
        train the model and save it.
        :return:
        """
        self.model.fit(self.X, self.Y,
                  batch_size=32, nb_epoch=10, verbose=1)
        self.model.save('my_model.h5')

    def test(self):
        """
        test success rate of model on test data
        :return: the score of success
        """
        self.model = load_model('my_model.h5')
        score = self.model.evaluate(self.TestSet, self.TestSetY, verbose=0)
        print(score)
        return score

    def classify(self,file_name):
        """
        gets an image of digit and uses cnn to classify it.
        :param file_name: the path of the image to classify
        :return: an int, the digit recognized.
        """
        image_to_clasify = np.load(file_name)
        # load model.
        self.model = load_model('my_model.h5')
        # classify image using weights.
        res = self.model.predict(image_to_clasify.reshape(1,28,28,1))
        # return the classifacation.
        return np.argmax(res)
#c = cnn()
#c.create_network()
#c.preapre_data()
#c.test()
#c.train()