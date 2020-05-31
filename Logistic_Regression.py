
# coding: utf-8

#importing basic libraries that required for model
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import scipy.optimize as opt  
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
#reading the data set
data = pd.read_csv('diabetes.csv')
data.tail()



# sigmoid function

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))


# cost function
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


def predict(theta, X):  
    probability = sigmoid(X* theta.T)
    # for x in probability:
    #     probs.append(x)
    return [1 if x > 0.27 else 0 for x in probability]


#gradient descent to minimize the cost
def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    #print parameters
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
#     error = cost(theta,X,y)

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = (np.sum(term)/len(X))
    
    return grad

data.insert(0, 'Ones', 1)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1] 
y = data.iloc[:,cols-1:cols]
X = np.array(X.values)  
y = np.array(y.values)  
theta = np.zeros(9)

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))  
#print cost(result[0], X, y)

#result = cost(gradient(theta,X,y),X,y)
#print result
#print b
# probs = []

theta_min = np.matrix(result[0])
test=[]
predictions = predict(theta_min, X)

i=[]
for x in range(len(predictions)):
    if (predictions[x]==1):
        i.append(predictions[x])
# print (len(i))

'''
# import matplotlib.pyplot as plt
# import numpy as np

# fpr = 0.05
# tpr = 0.95
# # This is the ROC curve
# fig, ax = plt.subplots(figsize=(12,8))
# ax.legend()
# ax.plot(fpr,tpr,  c='r', marker='x', label='DG')
# ax.set_xlim(0.0,1.0)
# ax.set_ylim(0.0,1.1)

# x=np.asarray(x)
# y=np.asarray(y)
# auc = np.trapz([y],[x],dx=1.0)

'''
# calculating accuracies
fpr, tpr, thresholds = metrics.roc_curve(y,predictions)
#training set accuracy
training_set_auccuracy = auc(fpr, tpr)
print(training_set_auccuracy)

plt.plot(fpr, tpr, color='darkorange', label='ROC curve ' % training_set_auccuracy)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.1, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



# reading lables for test
#labels = pd.read_csv('labels.csv',header=None)

#test_labels = labels.iloc[:,0]
#test_labels = np.asarray(test_labels)
#len(test_labels)



#fpr1, tpr1, thresholds1 = metrics.roc_curve(test_labels,test_predictions)

# test accuracy

#test_set_auccuracy = auc(fpr1, tpr1)
#print test_set_auccuracy

