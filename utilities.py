import torch

dtype = torch.cuda.FloatTensor

def softmax(Z):
    shiftZL = Z - torch.max(Z)
    exps = torch.exp(shiftZL)
    A= exps / torch.sum(exps, dim=0,keepdim=True)
    return A,Z
    
def relu(Z):
    A = Z.clamp(min=0)
        
    cache = Z    
    return A,cache
    
def relu_backward(dA, cache):

    Z = cache
    dZ = dA.clone().detach()# just converting dz to a correct object.
    
     
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
   
def softmax_backward(cache,Y):

    Z = cache
    
    A,ZL = softmax(Z)
    dZ = A-Y
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def one_hot_encoding(Y,Classes):
    print(len(Y[0]))
    Y1=np.zeros((Classes,len(Y[0])))
    for j in range(len(Y[0])):
        Y1[Y[0][j]][j]=1
    Y1=np.array(Y1)
    return Y1.T


def linear_forward(A,W,b):
   
    Z=torch.mm(W,A) + b
    
    cache = (A, W, b)
    
    return Z,cache


def cost_function(AL,Y,parameters,lambd):
    m=Y.shape[0]
    
    summa=0 
    L=len(parameters)//2
    for p in range(1,L):
        summa+=torch.sum((parameters["W" + str(p)])**2)
    
    loss = -torch.sum(torch.log(AL)*Y + torch.log(AL)*(1-Y), dim=0)

    cost = torch.mean(loss) + (lambd/(2*m)) * summa
    return cost,AL


def Dropout(a,keep_prop):
    drop_array = torch.rand(*a.shape) < keep_prop
    a = torch.mul(a,drop_array.type(dtype))
    a /=keep_prop
    return a