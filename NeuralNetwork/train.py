import numpy as np
import argparse

def sigmoid(X):
    return 1/(1+np.exp(-X))

def row_wise_softmax(X):
    exp = np.exp(X)
    ans = np.array([ x/np.sum(x) for x in exp])
    #print(ans)
    return ans

class NeuralNet:
    
    def __init__(self):
        print("NNet formed")

    def transform_Y(self):
        no_labels = len(np.unique(self.Y))
        Y = np.zeros( (self.Y.shape[0],no_labels) )
        print(Y)
        i=0
        for y in self.Y:
            print(i,y[0])
            Y[i,y[0]] = 1
            i+=1
        ans = np.zeros(Y.shape)

        #to match calculations on web
        ans[:,0]=Y[:,1]
        ans[:,1] =Y[:,0]
        return ans

    def forward_pass(self):
        #print(self.X1)
        #print(self.W1)
        self.Z1 = self.X1.dot(self.W1)
        self.X2 = sigmoid(self.Z1)
        self.X2 = np.concatenate([np.ones((self.X2.shape[0],1)), self.X2],axis=1)
        self.Z2 = self.X2.dot(self.W2)

        return row_wise_softmax(self.Z2)

    def backward_pass(self):
        dE_dW2 = np.zeros(self.W2.shape,dtype='float')
        dE_dW1 = np.zeros(self.W1.shape,dtype='float')
        #print(self.Y,self.yHat)
        for i in range(self.X1.shape[0]):
            dE_dyHat = self.Y[[i]]/self.yHat[[i]]
            #print(dE_dyHat.shape)
            #print("de_dyHat",dE_dyHat)

            """dyHat_dZ2 = (self.yHat[[i]].T).dot(self.yHat[[i]])
            diag_mask = (np.eye(dyHat_dZ2.shape[0],dtype='int')==1)
            non_diag_mask = (np.eye(dyHat_dZ2.shape[0],dtype='int')==0)
            dyHat_dZ2[diag_mask] =np.sqrt(dyHat_dZ2[diag_mask])*(1-np.sqrt(dyHat_dZ2[diag_mask]))
            dyHat_dZ2[non_diag_mask] = -dyHat_dZ2[non_diag_mask]"""
            dE_dZ2 = self.yHat[[i]]-self.Y[[i]]
            #print("de_dZ2",dE_dZ2)
            
            #print("de_dw2",dE_dW2)

            #print("W2",self.W2)
            dE_dX2 = dE_dZ2.dot(self.W2.T)
            #print("DE_DX2",dE_dX2)

            dE_dZ1 = dE_dX2[[0],1:]*self.X2[[i],1:]*(1-self.X2[[0],1:])
            #print("dE_dZ1",dE_dZ1)

            dE_dW2 += self.X2[[i]].T.dot(dE_dZ2)
            dE_dW1 +=  self.X1[[i]].T.dot(dE_dZ1)
            #print("de_dW1",dE_dW1)

        #print("dE_dW2 ",self.X1.shape[0],dE_dW2/self.X1.shape[0])
        #print("dE_dW1 ",self.X1.shape[0],dE_dW1/self.X1.shape[0])
        self.W2 = self.W2 - 0.01*dE_dW2/self.X1.shape[0]
        self.W1 = self.W1 - 0.01*dE_dW1/self.X1.shape[0]

        
    def fit(self,X,Y):
        
        self.Y = Y
        self.Y = self.transform_Y()

        #layer 1
        nn_next = 2
        self.X1 = X
        bias = np.ones((self.X1.shape[0],1))
        self.X1 = np.concatenate([bias,self.X1],axis=1)
        self.W1 = np.zeros((self.X1.shape[1],nn_next),dtype='float')
        self.Z1 = np.zeros((self.X1.shape[0],self.W1.shape[1]))
       


        #layer2
        nn_next = 2
        
        self.X2 = np.zeros((self.Z1.shape[0],self.Z1.shape[1]),dtype='float')
        bias = np.ones((self.X2.shape[0],1))
        self.X2 = np.concatenate([bias,self.X2],axis=1)
        
        self.W2 = np.zeros((self.X2.shape[1],nn_next),dtype='float')
        self.Z2 = np.zeros((self.X2.shape[0],self.W2.shape[1]),dtype='float')

        #layer3
        self.X3 = np.zeros((self.Z2.shape[0],self.Z2.shape[1]),dtype='float')
        
        self.W1 = np.array([[-0.00469,  0.00797 ],
                            [-0.00256 , 0.00889 ],
                            [0.00146  , 0.00322 ],
                            [0.00816  , 0.00258 ],
                            [-0.00597 , -0.00876]])
        self.W2 = np.array([[-0.00588 , -0.00232],
                            [-0.00647 , 0.00540 ],
                            [0.00374  , -0.00005]])

        #print(self.X1.shape,self.W1.shape,self.Z1.shape)
        #print(self.X2.shape,self.W2.shape,self.Z2.shape)
        iterations = 1000
        for i in range(iterations):
            self.yHat = self.forward_pass()
            if( i == iterations-1):
                print("Yhat after iteration",i,":",self.yHat)
            y_logyHat = np.log(self.yHat)*(self.Y)
            print("loss ",np.sum(np.sum(y_logyHat,axis=1))/4)
            self.backward_pass()



def main(data):
    """data = np.array([
            [1,  252,  4  , 155,  175,  1],  
            [2,  175,  10 , 186,  200,  1],  
            [3,  82 ,  131, 230,  100,  0],  
            [4,  115,  138,  80,  88 ,  0]
            ])
    """
    X = data[:,1:-1]
    Y = data[:,[-1]]

    Y = Y.astype(np.int32)
    print(X.shape)
    print(Y.shape)

    nnet = NeuralNet()
    nnet.fit(X,Y)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train","--train_file",action="store")
    args = parser.parse_args()
    print(args.train)
    data = np.genfromtxt(args.train, delimiter=',')
    #print(data)
    main(data)
