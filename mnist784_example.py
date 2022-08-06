from mlp import MLP
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import pickle

if __name__=="__main__":
    mnist=fetch_openml('mnist_784',as_frame=False)
    encoder=OneHotEncoder()
    data=np.array(mnist.data)
    label=np.array(mnist.target)

    data=np.reshape(data,(70000,-1))
    label=np.reshape(label,(-1,1))
    encoder.fit(label)
    label=encoder.transform(label).toarray()
    data,label=data.T,label.T
    data,label=data.astype(np.float64),label.astype(np.float64)

    train_data,train_label=data[:,0:60000],label[:,0:60000]
    dev_data,dev_label=data[:,60000:65000],label[:,60000:65000]
    test_data,test_label=data[:,65000:],label[:,65000:]

    mlp=MLP(train_data,train_label,dev_data,dev_label,2,[256,256],weight_decay=0.1,learning_rate=0.0001)
    mlp.fit(128,200)
    
    accuracy,Loss=mlp.evaluate(test_data,test_label)
    print("test accuracy: "+str(accuracy))
    print("test Loss: "+str(Loss))

    with open("mnist784_classifier.h5","wb") as f:
        pickle.dump(mlp,f)
