import matplotlib.pyplot as plt
import numpy as np

class MLP:
    def __init__(self, train_data, train_label,dev_data,dev_label,hidden_depth,
                 hidden_size,learning_rate=0.0001,weight_decay=0.1,
                 early_stopping=True,stop_epoch=10,stop_min=0.0001):
        self.h_depth=hidden_depth
        self.h_size=hidden_size
        self.depth=hidden_depth+1
        self.W=[None]*(self.depth)
        self.b=[None]*(self.depth)
        self.Z=[None]*(self.depth)
        self.A=[None]*(self.depth)
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.early_stopping=early_stopping
        self.stop_epoch=stop_epoch
        self.stop_min=stop_min

        data_mean=np.mean(train_data,axis=0)
        data_var=np.var(train_data,axis=0)
        self.train_data=(train_data.copy()-data_mean)/np.sqrt(data_var)
        self.train_label=train_label

        data_mean=np.mean(dev_data,axis=0)
        data_var=np.var(dev_data,axis=0)
        self.dev_data=(dev_data.copy()-data_mean)/np.sqrt(data_var)
        self.dev_label=dev_label

    def softmax(self,Z):
        operand=np.sum(np.exp(Z-np.max(Z,axis=0)),axis=0,keepdims=True)
        return np.exp(Z-np.max(Z,axis=0))/operand
    
    def cross_entropy(self,A,label,batch_size):
        epsilon=1e-8
        Loss=np.sum(label*A,axis=0)
        Loss=(-1/batch_size)*np.sum(np.log(Loss+epsilon))
        return Loss

    def forward_prop(self,X,label,batch_size):
        self.Z[0]=np.matmul(self.W[0],X)+self.b[0]
        self.A[0]=np.where(self.Z[0]>0,self.Z[0],0)

        for layer in range(1,self.depth):
            self.Z[layer]=np.matmul(self.W[layer],self.A[layer-1])+self.b[layer]   #Z=WX+b
            if layer!=self.depth-1:
                self.A[layer]=np.where(self.Z[layer]>0,self.Z[layer],0)
            else:
                self.A[layer]=self.softmax(self.Z[layer])

        Loss=self.cross_entropy(self.A[self.depth-1],label,batch_size)
        eval=np.argmax(self.A[self.depth-1],axis=0)
        
        accuracy=0
        for i in range(batch_size):
            if label[eval[i],i]==1:
                accuracy+=1
        accuracy=accuracy/batch_size

        return accuracy,Loss
    
    def back_prop(self,data,label):
        dW,dZ,db=None,None,None

        for layer in range(self.depth-1,-1,-1):
            if layer!=self.depth-1:
                dZ=dA*np.where(self.Z[layer]>0,1,0)     #dA*relu'(Z)
            else:
                dZ=self.A[self.depth-1]-label     #dZ=Y-label

            if layer!=0:
                dW=np.matmul(dZ,self.A[layer-1].T)
            else:
                dW=np.matmul(dZ,data.T)

            db=np.reshape(np.sum(dZ,axis=1),(-1,1))
            self.W[layer]=(1-self.learning_rate*self.weight_decay)*self.W[layer]-self.learning_rate*dW
            self.b[layer]=self.b[layer]-self.learning_rate*db
            dA=np.matmul(self.W[layer].T,dZ)

    def fit(self,batch_size=128,epoch=100):
        for layer in range(self.depth):
            # He initialization
            if layer==0:
                self.W[layer]=np.random.normal(0,np.sqrt(2/np.size(self.train_data,axis=0)),(self.h_size[layer],np.size(self.train_data,axis=0)))
                self.b[layer]= np.random.normal(0,np.sqrt(2/np.size(self.train_data,axis=0)),(self.h_size[layer],1))
            elif layer==self.depth-1:
                self.W[layer]=np.random.normal(0,np.sqrt(2/self.h_size[layer-1]),(np.size(self.train_label,axis=0),self.h_size[layer-1]))
                self.b[layer]= np.random.normal(0,np.sqrt(2/self.h_size[layer-1]),(np.size(self.train_label,axis=0),1))
            else:
                self.W[layer]=np.random.normal(0,np.sqrt(2/self.h_size[layer-1]),(self.h_size[layer],self.h_size[layer-1]))
                self.b[layer]= np.random.normal(0,np.sqrt(2/self.h_size[layer-1]),(self.h_size[layer],1))

            self.W[layer]=self.W[layer].astype(np.float64)
            self.b[layer]=self.b[layer].astype(np.float64)

        train_Loss=[None]*epoch
        train_accuracy=[None]*epoch
        dev_Loss=[None]*epoch
        dev_accuracy=[None]*epoch
        count=0
        max_accuracy=0
        for iter in range(epoch):
            batch_indices=np.random.choice(np.size(self.train_data,axis=1),batch_size,replace=True)
            batch_data=self.train_data[:,batch_indices].copy()
            batch_label=self.train_label[:,batch_indices].copy()

            train_accuracy[iter],train_Loss[iter]=self.forward_prop(batch_data,batch_label,batch_size)
            dev_accuracy[iter],dev_Loss[iter]=self.evaluate(self.dev_data,self.dev_label)
            train_accuracy[iter],train_Loss[iter]=np.trunc(train_accuracy[iter]*(1e+5))/(1e+5),np.trunc(train_Loss[iter]*(1e+5))/(1e+5)
            dev_accuracy[iter],dev_Loss[iter]=np.trunc(dev_accuracy[iter]*(1e+5))/(1e+5),np.trunc(dev_Loss[iter]*(1e+5))/(1e+5)

            print(("epoch "+str(iter+1)).ljust(10)+"-> train accuracy: "+str(train_accuracy[iter]).ljust(10)+"train Loss: "+str(train_Loss[iter]).ljust(10),end=' ')
            print("dev accuracy: "+str(dev_accuracy[iter]).ljust(10)+"dev Loss: "+str(dev_Loss[iter]))

            self.back_prop(batch_data,batch_label)
            
            if self.early_stopping:
                count+=1
                if max_accuracy+self.stop_min <= dev_accuracy[iter]:
                    max_accuracy=dev_accuracy[iter]
                    count=0
            
                if count==self.stop_epoch:
                    print("training stopped since accuracy of dev set didn't increase over "+str(self.stop_min)+" for recent 10 epochs")
                    break

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(range(1,epoch+1),train_Loss,label='train set')
        plt.plot(range(1,epoch+1),dev_Loss,color='green',label="dev set")
        plt.legend()
        plt.subplot(1,2,2)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.plot(range(1,epoch+1),train_accuracy,color='orange',label='train set')
        plt.plot(range(1,epoch+1),dev_accuracy,color='red',label='dev set')
        plt.legend()
        plt.show()

    def evaluate(self,data,label):
        data_size=np.size(data,axis=1)

        data_mean=np.mean(data,axis=0)
        data_var=np.var(data,axis=0)
        data=(data.copy()-data_mean)/np.sqrt(data_var)

        Z=np.matmul(self.W[0],data)+self.b[0]
        A=np.where(Z>0,Z,0)

        for layer in range(1,self.depth):
            Z=np.matmul(self.W[layer],A)+self.b[layer]   #Z=WX+b
            
            if layer!=self.depth-1:
                A=np.where(Z>0,Z,0)
            else:
                A=self.softmax(Z)

        Loss=self.cross_entropy(A,label,data_size)
        eval=np.argmax(A,axis=0)
        
        accuracy=0
        for i in range(data_size):
            if label[eval[i],i]==1:
                accuracy+=1
        accuracy=accuracy/data_size

        return accuracy,Loss