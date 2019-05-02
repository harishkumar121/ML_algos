
# coding: utf-8

#importing basic libraries that required for model
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  




#reading the data set
data = pd.read_csv('diabetes.csv')
data.tail()


# In[68]:

#labeling the data

EB = data[data['Outcome'].isin([0])]
DG = data[data['Outcome'].isin([1])]



# sigmoid function

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

'''
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(EB['R_freq'], EB['Y_freq'], c='b', marker='o', label='EB')
# ax.scatter(DG['R_freq'], EB['Y_freq'],  c='r', marker='x', label='DG')
# ax.set_ylim(0.0000,0.00006)
# ax.set_xlim(-0.0004,0.0150)
# # ax.set_xticks(np.arange(-0.0004,0.0150,0.001))
# ax.legend()
'''

# cost function
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))



data.insert(0, 'Ones', 1)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1] 
y = data.iloc[:,cols-1:cols]
X = np.array(X.values)  
y = np.array(y.values)  
theta = np.zeros(9)


# In[75]:

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


import scipy.optimize as opt  
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))  
#print cost(result[0], X, y)

#result = cost(gradient(theta,X,y),X,y)
#print result
#print b
# probs = []

def predict(theta, X):  
    probability = sigmoid(X* theta.T)
    # for x in probability:
    #     probs.append(x)
    return [1 if x > 0.27 else 0 for x in probability]




theta_min = np.matrix(result[0])
test=[]
predictions = predict(theta_min, X)

print theta_min


i=[]
for x in range(len(predictions)):
    if (predictions[x]==1):
        i.append(predictions[x])
# print (len(i))




# reading test data set
#test_data = pd.read_csv('/home/neureol/Documents/EB_DG_work/Data/test_set.csv',header=None)
#test_data.insert(0, 'Ones', 1)
#cols = test_data.shape[1]  
#test = test_data.iloc[:,0:cols] 
#test = np.array(test.values)


# prediction on test dataset
#test_predictions = predict(theta_min, test) 





#i=[]
#for x in range(len(test_predictions)):
#    if (test_predictions[x]==1):
#        i.append(test_predictions[x])
# print len(i)


# In[110]:
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


# In[111]:


# x=np.asarray(x)
# y=np.asarray(y)
# auc = np.trapz([y],[x],dx=1.0)

'''

# calculating accuracies
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = metrics.roc_curve(y,predictions)


#training set accuracy
training_set_auccuracy = auc(fpr, tpr)

print training_set_auccuracy


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
#labels = pd.read_csv('/home/neureol/Documents/EB_DG_work/Data/labels.csv',header=None)

#test_labels = labels.iloc[:,0]
#test_labels = np.asarray(test_labels)
#len(test_labels)



#fpr1, tpr1, thresholds1 = metrics.roc_curve(test_labels,test_predictions)

# test accuracy

#test_set_auccuracy = auc(fpr1, tpr1)
#print test_set_auccuracy

