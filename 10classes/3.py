import sys
import numpy as np
from abc import ABC, abstractmethod 
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

class Layer(ABC):
    def __init__(self):
        self.prevIn = []
        self.prevOut = []

    def setPrevIn(self ,dataIn): 
        self.prevIn = dataIn

    def setPrevOut( self , out ): 
        self.prevOut = out
        
    def getPrevIn(self): 
        return self.prevIn 

    def getPrevOut(self): 
        return self.prevOut

    def backward(self, gradIn): 
        pass

    @abstractmethod
    def forward(self ,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass
  


class InputLayer(Layer):
    def __init__(self, dataIn):
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, ddof=1, axis=0)
        self.stdX[self.stdX==0] = 1

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        zscored = (dataIn - self.meanX) / self.stdX
        self.setPrevOut(zscored)
        return zscored
    
    def gradient(self):
        pass


class LinearLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = dataIn
        self.setPrevOut(y)
        return y
    
    #tensor method
    # def gradient(self):
    #     i = np.identity(self.prevIn.shape[1])
    #     tensor = np.array([i] * self.prevIn.shape[0])
    #     return tensor

    #Hadamard product
    def gradient(self):
        i = self.getPrevOut()
        tensor = np.ones(i.shape)
        return tensor

    def backward(self,gradIn):
        linearGrad = np.multiply(gradIn, self.gradient())
        return linearGrad
    
class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = np.maximum(0,dataIn)
        self.setPrevOut(y)
        return y
    
    #Tensor method
    # def gradient(self):
    #     i = np.identity(self.prevIn.shape[1])
    #     tensor = np.array([i] * self.prevIn.shape[0])
    #     tensor[self.prevIn<0]=0
    #     return tensor

    #Hadamard Product    
    def gradient(self):
        I = self.getPrevOut()
        i = self.getPrevIn()
        tensor = np.ones(I.shape)
        tensor[i < 0] = 0
        return tensor

    def backward(self,gradIn):
        reluGrad = np.multiply(gradIn, self.gradient())
        return reluGrad

        
class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = 1/(1+np.exp(-dataIn))
        self.setPrevOut(y)
        return y
    
    #Tensor method
    # def gradient(self):
    #     tensor = np.ndarray(shape=(self.prevIn.shape[0],self.prevIn.shape[1],self.prevIn.shape[1]))
    #     tensor.fill(0.0)
    #     for r in range(self.prevIn.shape[0]):
    #         for c in range(self.prevIn.shape[1]):
    #             tensor[r][c][c] = self.prevOut[r][c]*(1-self.prevOut[r][c])+0.0000001
    #     return tensor

    #Hadamard Product
    def gradient(self):
        i = self.getPrevOut()
        return i*(1-i)
    
    def backward(self,gradIn):
        sigmoidGrad = np.multiply(gradIn, self.gradient())
        return sigmoidGrad


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = np.exp(dataIn-np.max(dataIn,axis=1,keepdims=True)) / np.sum(np.exp(dataIn-np.max(dataIn,axis=1,keepdims=True)), axis=1, keepdims=True)
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        y = self.prevOut
        n = y.shape[0]
        k = y.shape[1]
        tensor = np.ndarray(shape=(n,k,k))
        tensor.fill(0.0) 
        for l in range(n):
            for i in range(len(y[l])):
                for j in range(len(y[l])):
                    if i == j:
                        tensor[l][i][j] = y[l][i] * (1-y[l][i])
                    else:
                        tensor[l][i][j] = -y[l][i] * y[l][j]
        return tensor

    # def backward(self,gradIn):
    #     softmaxGrad = np.zeros((gradIn.shape[0],self.gradient().shape[1]))
    #     #for each observation computation. 
    #     for n in range(gradIn.shape[0]): 
    #         softmaxGrad[n,:] = gradIn[n,:]@self.gradient()[n,:,:]
    #     return softmaxGrad

    def backward(self,gradIn):
        #without for loop faster
        softmaxGrad = np.einsum('ij,ijk->ik',gradIn,self.gradient())
        return softmaxGrad



class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        self.setPrevOut(y)
        return y
    
    #Tensor method
    # def gradient(self):
    #     tensor = np.ndarray(shape=(self.prevIn.shape[0],self.prevIn.shape[1],self.prevIn.shape[1]))
    #     tensor.fill(0.0)
    #     for r in range(self.prevIn.shape[0]):
    #         for c in range(self.prevIn.shape[1]):
    #             tensor[r][c][c] = (1-self.prevOut[r][c]*self.prevOut[r][c])+0.0000001
    #     return tensor

    #Hadamard Product
    def gradient(self):
        i = self.getPrevOut()
        return (1 - i**2)

    def backward(self,gradIn):
        tanhGrad = np.multiply(gradIn, self.gradient())
        return tanhGrad



class FullyConnectedLayer(Layer):
    def __init__(self ,sizeIn ,sizeOut):
        #xavier initialization - multiple on the range can either be 0.0001 or 0.001 or 0.01 
        self.weight = np.random.uniform(-np.sqrt(6/(sizeIn+sizeOut))*0.00001, np.sqrt(6/(sizeIn+sizeOut))*0.00001, size=(sizeIn,sizeOut))
        self.bias = np.random.uniform(-np.sqrt(6/(sizeIn+sizeOut))*0.00001, np.sqrt(6/(sizeIn+sizeOut))*0.00001, size=(1,sizeOut))
        self.sw = 0
        self.rw = 0
        self.sb = 0
        self.rb = 0

    def getWeights(self):
        return self.weight

    def setWeights(self , weights):
        self.weight = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias): 
        self.bias = bias

    def forward(self, dataIn): 
        self.setPrevIn(dataIn)
        y = np.dot(dataIn, self.weight) + self.bias
        self.setPrevOut(y)
        return y

    def gradient(self): 
        tensor = self.getWeights()
        return tensor.T

    def backward(self, gradIn): 
        fclGrad = np.dot(gradIn, self.gradient())
        return fclGrad

    def updateWeights(self, gradIn, eta, epochCnt, d1=0.9, d2=0.999, ns=10e-8):
        dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
        dJdw = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
        self.sw = d1*self.sw + (1-d1)*dJdw
        self.rw = d2*self.rw + (1-d2)*np.square(dJdw)
        self.sb = d1*self.sb + (1-d1)*dJdb
        self.rb = d2*self.rb + (1-d2)*np.square(dJdb)

        self.setWeights(self.weight - eta*(self.sw/(1-d1**epochCnt))/(np.sqrt(self.rw/(1-d2**epochCnt))+ns))
        self.setBias(self.bias - eta*(self.sb/(1-d1**epochCnt))/(np.sqrt(self.rb/(1-d2**epochCnt))+ns))

class SquaredError():
    def eval(self ,Y, Yhat): 
        return np.sum(np.square(Y-Yhat))/Y.shape[0]

    def gradient(self ,Y, Yhat):     
        return -2*(Y-Yhat)

class LogLoss():
    def eval(self ,Y, Yhat): 
        return np.mean(-(np.multiply(Y,np.log(Yhat + 0.0000001)) + (np.multiply((1-Y),np.log(1-Yhat + 0.0000001)))))
    def gradient(self ,Y, Yhat):     
        return -((Y - Yhat)/(Yhat*(1-Yhat) + 0.0000001))


class CrossEntropy():
    def eval(self ,Y, Yhat): 
        return -np.sum(np.multiply(Y, (np.log(Yhat+0.0000001))))/ Y.shape[0]
    def gradient(self ,Y, Yhat):     
        return -(Y/(Yhat+0.0000001))



#________________________________________________________________________________________________________________________

def termination(JL):
    if len(JL)>2 and np.abs(JL[-1]-JL[-2])<10**-8:
        return True
    else:
        return False

#running different architectures
def archRun(nnList, g_eta, epochs, archlabel):
    JLtr = []
    JLte = []
    epochL = []
    for e in tqdm(range(1,epochs+1)):
        p = np.random.permutation(len(X_train))
        xTrain = X_train[p]
        yTrain = ohYtrain[p]

        h = xTrain
        for i in range(len(nnList)-1):
            h = nnList[i].forward(h)
        
        
        Jtr = nnList[-1].eval(yTrain, h)


        #----------------------------------------------------------------------------------------------------------------------

        # Backward pass 
        grad = nnList[-1].gradient(yTrain, h)
        for i in range(len(nnList)-2, -1, -1):
            newgrad = nnList[i].backward(grad)
            if isinstance(nnList[i], FullyConnectedLayer):
                nnList[i].updateWeights(grad, g_eta, e)
            grad = newgrad
            
        
        #print(total_mb)
        #----------------------------------------------------------------------------------------------------------------------
        #Forward pass through training set
        tr = X_train
        for i in range(len(nnList)-1):
            tr = nnList[i].forward(tr)
        Jtr = nnList[-1].eval(ohYtrain, tr)        


        JLtr.append(Jtr)


        #----------------------------------------------------------------------------------------------------------------------
        #VALIDATION SET:

        te = X_test
        for i in range(len(nnList)-1):
            te = nnList[i].forward(te)
        Jte = nnList[-1].eval(ohYtest, te)
        JLte.append(Jte)

        print(Jtr, Jte)

        epochL.append(e)
        if termination(JLtr):
            break
#----------------------------------------------------------------------------------------------------------------------

    #Accuracy - Training
    atr = 0
    predictiontr = np.argmax(tr, axis=1).reshape(tr .shape[0],1)


    for i in range(predictiontr.shape[0]):
        if predictiontr[i,:] == Y_train[i]:
            atr += 1
    atr = atr*100/predictiontr.shape[0]
    print("Training Accuracy: ", atr)

    #Accuracy - Validation
    ate = 0
    predictionte = np.argmax(te, axis=1).reshape(te.shape[0],1)
    for i in range(predictionte.shape[0]):
        if predictionte[i,:] == Y_test[i]:
            ate += 1
    ate = ate*100/predictionte.shape[0]
    print("Validation Accuracy: ", ate)


    #Plot J vs epochs for training and validation set
    plt.plot(epochL, JLtr, c='darkblue', label='Training J vs Epochs')
    plt.plot(epochL, JLte, c='lawngreen', label='Validation J vs Epochs')
    plt.xlabel("Epochs")

    plt.ylabel("J")
    plt.legend()
    plt.title(archlabel+f': Training Accuracy - {atr}, Validation Accuracy - {ate}')
    plt.show()




if __name__=="__main__":
    np.random.seed(0)

    #LOADING DATA

    trainingfile = "./mnist_train_100.csv"
    testfile = "./mnist_valid_10.csv"


    #Design decison - without using input layer, dividing by 255
    #X_train = np.loadtxt(trainingfile, delimiter=",")[:,1:]/255
    X_train = np.loadtxt(trainingfile, delimiter=",", dtype=np.float128)[:,1:]
    Y_train = np.loadtxt(trainingfile, delimiter=",", dtype=np.float128)[:,0]
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    #One-hot encoding
    ohYtrain = np.squeeze(np.eye(10)[Y_train.astype(int).reshape(-1)])
    # print(ohYtrain.shape)
    # print(ohYtrain[:3,:])
    #print(X_train)
    #print(Y_train)
    #print(X_train.shape)
    #print(Y_train.shape)

    X_test = np.loadtxt(testfile, delimiter=",")[:,1:]
    Y_test = np.loadtxt(testfile, delimiter=",")[:,0]
    Y_test = Y_test.reshape(Y_test.shape[0],1)
    ohYtest = np.squeeze(np.eye(10)[Y_test.astype(int).reshape(-1)])


    # Initializing the layers

    #Model-1 - uncomment two lines below to run and comment out all other models. 
    nn1 = [InputLayer(X_train), FullyConnectedLayer(X_train.shape[1],50), SigmoidLayer(), FullyConnectedLayer(50,10), SoftmaxLayer(), CrossEntropy()]
    archRun(nn1, 2.55, 50, "Architecture 1")
    
    #Model-2 - uncomment two lines below to run and comment out all other models.
    # nn2 = [InputLayer(X_train), FullyConnectedLayer(X_train.shape[1],30), SigmoidLayer(), FullyConnectedLayer(30,10), SigmoidLayer(), LogLoss()]
    # archRun(nn2, 0.0333, 100, "Architecture 2")

    #Model-3 - uncomment two lines below to run and comment out all other models.
    # nn3 = [InputLayer(X_train), FullyConnectedLayer(X_train.shape[1],30), ReLuLayer(), FullyConnectedLayer(30,60), TanhLayer(), FullyConnectedLayer(60,10), SoftmaxLayer(), CrossEntropy()]
    # archRun(nn3, 0.007, 100, "Architecture 3")


