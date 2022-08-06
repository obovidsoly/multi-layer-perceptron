## multi-layer-perceptron
this is a multi layer perceptron implemented in Numpy.  
it can be used only for classifying problems because it uses cross entropy as loss function.  
depth of hidden layer, number of node of hidden layer, learning rate, weight decay rate, whether executing of early stopping and hyper parameter about early stopping can be adjusted.

### MLP constructor
```
def __init__(self, train_data, train_label,dev_data,dev_label,hidden_depth,
                 hidden_size,learning_rate=0.0001,weight_decay=0.1,
                 early_stopping=True,stop_epoch=10,stop_min=0.0001)
```
when constructing an instance, train set, dev set, depth of hidden layer, number of node of hidden layer must be given by parameter.  
before passing data and label, they should be form of matrix which the data and label must be arrayed in column vector, and label should be a matrix of one-hot vector. 
when passing 'hidden_size' which is the number of node at each layer, value should be given as list or tuple.  
if you set 'hidden_depth' as 3, then the number of total layer of perceptron is 5 including input and output layer. 'wieght_deay' means weight decay rate. if 'early_stopping', 'stop_epoch', 'stop_min' are respectively True, 10, 0.0001 then training will be automatically stopped if accuracy of valid data doesn't increase over 0.0001 for recent 10 epochs.

### fit
```
def fit(self,batch_size=128,epoch=100)
```
'fit' executes traning model.  
'batch_size' means the number of sample reflected at updating weight matrix and bias vector.  
'epoch' means maximum number of allowed training epoch.  
it prints graph of loss and accuracy of traning and dev set at the end.  
```
epoch 1   -> train accuracy: 0.10156   train Loss: 2.92097    dev accuracy: 0.07879   dev Loss: 2.9492
epoch 2   -> train accuracy: 0.14843   train Loss: 2.82545    dev accuracy: 0.101     dev Loss: 2.73842
                                                    ⋮
epoch 173 -> train accuracy: 0.88281   train Loss: 0.38714    dev accuracy: 0.845     dev Loss: 0.51442
training stopped since accuracy of dev set didn't increase over 0.0001 for recent 10 epochs
```
![image](https://user-images.githubusercontent.com/44926279/182771405-f8dde7b8-1b2a-4408-a20a-c66c9240557b.png)


### evaluate
```
def evaluate(self,data,label)
```
'evaluate' just returns accuracy and loss of given data.
