# import the necessary modulde
from calc_utils import  calc_utils
from sklearn.svm import LinearSVC
import  pickle
from sklearn.metrics import accuracy_score
from PIL import Image
import  numpy as np
import os
import random
class notMySVM(calc_utils):
    def __init__(self):
        calc_utils.__init__(self)
        # create an object of type LinearSVC
        self.svc_model = LinearSVC(random_state=0)

    def trainAndPredict(self):
        """
        use scilearn svm to train,test and save model.
        :return:
        """
        print("train")
        filename= 'finalized_model.sav'
        # train the algorithm on training data and predict using the testing data
        model = self.svc_model.fit(self.X.T, self.Y)
        pickle.dump(model, open(filename, 'wb'))
        #model = pickle.load(open(filename, 'rb'))
        pred1 =model.predict(self.TestSet.T)
        # print the accuracy score of the model
        print("LinearSVC accuracy : ", accuracy_score(self.TestSetY, pred1, normalize=True))

svm = notMySVM()
svm.get_data()
svm.trainAndPredict()