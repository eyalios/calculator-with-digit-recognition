from calc_utils import calc_utils
import  numpy as np
import random
class Learner(calc_utils):

   def __init__(self):
       calc_utils.__init__(self)
       #randomly start a weight matrix
       self.weights = np.random.rand(10,  self.size_of_vec)
       self.learned_weights = self.weights

   def hinge_lost(self,w,x,y,reg):
       """
        calcutlate hinge loss and optimiztion of loss function using sgd (one iteration).
       :param w: a matrix of weights 10X784
       :param x: a 3d array with N sample images 784xN
       :param y: vector of matching tags for each sample NX1
       :return:
       :arg total_loss - the summed loss from this w,x.
       :arg weight update - the matrix of updates calculated with sgd.
       """
       #compute the scores of all images with respect to W matrix
       score_array = np.dot(w,x)
       #create a vector of the results for each correct tag (Nx1)
       true_res_vec = score_array[y,np.arange(score_array.shape[1])]
       #create a matrix (10xN) where each column holds the result of its correct tag from true_res_vec
       true_res_matrix = np.zeros(score_array.shape)
       true_res_matrix[:,np.arange(true_res_vec.shape[0])] = true_res_vec[np.arange(true_res_vec.shape[0])]
       #calculating hinge loss
       loss_matrix =  np.maximum(0,score_array - true_res_matrix + 1)
       loss_matrix[y,np.arange(score_array.shape[1])] = 0
       #summing resualt and adding regularization
       total_loss = np.sum(loss_matrix)/x.shape[1] + 0.5 *reg * np.sum(w*w)

       #SGD
       #create matrix the size 10xN
       binary_loss_matrix = np.zeros(score_array.shape)
       #where loss function is larger then zero make entry = 1
       binary_loss_matrix[loss_matrix > 0] = 1
       #sum how many entrys are larger then 1 for each sample.
       sum_of_ones_vec = np.sum(binary_loss_matrix,axis=0)
       #in each correct y location put -sumofones, this will enable us to use dot product to get
       #the derivative of hinge loss.
       binary_loss_matrix[y,np.arange(loss_matrix.shape[1])] = - sum_of_ones_vec
       #dot product divided by N
       gradient = np.dot(binary_loss_matrix,x.T) / x.shape[1]
       #add regulazation to get update for weights.
       weight_update = gradient +  reg * w

       return (total_loss, weight_update)

   def test(self,W,X,Y):
       """
       test the perfomnce of the weights on given input
       :param W: weights matrix 10xD(28*28)
       :param X: data to check  DxN
       :param Y: Nx1 tags of data.
       :return: precentege of correct tags.
       """
       #predict all data with W, using dot product.
       res_matrix = np.dot(W,X)
       #get max argument for each data sample.
       res_index_vec = np.argmax(res_matrix, axis=0)
       #compute how many correct tags.
       dif_vec = res_index_vec - Y
       dif_vec[dif_vec != 0] = -1
       dif_vec += 1
       #check success rate.
       success_rate = sum(dif_vec) / X.shape[1]
       return  success_rate


   def train(self,batch_size,reg,step_size):
       """
       train model.
       :param batch_size: how many samples to use in each SGD iteration.
       :param reg: reg coefficent.
       :param step_size: step size for weight update in sgd.
       :return:
       """
       loss = 10
       counter = 0
       min = 1000
       update = 10000
       while(loss >0.04 and counter <400000):
         #get random batch to send to SGD.
         samples = random.sample(range(self.total_data),batch_size)
         #send data to calculete hinge loss and get weight update.
         loss,update = self.hinge_lost(self.weights,self.X[:,samples],self.Y[samples],reg)
         #update weights.
         self.weights -= update * step_size
         counter += 1
         print(loss)
         print(counter)
         if(min > loss):
             min = loss
             min_iter = counter
       print(np.sum(update))
       print(min)
       print(min_iter)
       #save W.
       np.save('learned_W',self.weights)

   def classify(self,filename):
       """
       using learned weights to classify image.
       :param filename: path to np array to classify.
       :return:
       """
       #open sample to classify.
       image_to_clasify = np.load(filename).reshape(28*28)
       #load weights.
       try:
         kernel = np.load('learned_W.npy')
       except:
         self.get_data()
         self.train(128,0.01,0.01)
         kernel = np.load('learned_W.npy')

       #classify image using weights.
       res = np.dot(kernel,image_to_clasify.T)
       #return the classifacation.
       return np.argmax(res)

#L = Learner()
###L.classify('try.npy')
