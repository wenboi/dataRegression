# coding: utf-8

# %load ../../standard_import.txt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:10,:])
    return(data)


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);


# ### 逻辑斯特回归
data = loaddata('data1.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

##定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))


##定义损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    #print('theta: \n', theta)
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
 
print('Cost: \n', cost)

# #### 最小化损失函数
res = minimize(costFunction, initial_theta, args=(X,y),  options=None)
#res

    
# #### 考试1得分35，考试2得分85的同学通过概率有多高
# #### P(y=1|x:theta)=h_theta(x)
# #### 即为求  h_theta(x)   
# #### 而 h_theta(x) =  sigmoid（  theta的T * x ）  
# #### 其中 theta的值为 res.x.T
   
#fig, ax = plt.subplots()
out = sigmoid(  res.x.T.dot( np.array([1, 35, 80])  ))
print('通过概率 \n', out)


plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
plt.scatter(35, 80, s=60, c='r', marker='v', label='(35, 80)')
#print('sigmoid(np.array([1, 35, 80]): \n', sigmoid(np.array([1,35, 80])))   #X 
#print('res.x.T-----------: \n', res.x.T)   #最终求得的theta值
plt.show()
# #### 画决策边界
#x1_min, x1_max = X[:,1].min(), X[:,1].max(),
#x2_min, x2_max = X[:,2].min(), X[:,2].max(),
#xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
#h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
#h = h.reshape(xx1.shape)
#plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');


