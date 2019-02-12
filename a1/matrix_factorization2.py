import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix, random
import time
import matplotlib.pyplot as plt

print(time.time())

num_factors = 10
num_iter = 75
reg = 0.05
lrate = 0.005

randomseed = 9294

def ratings_matrix(x):
    return coo_matrix((x[:,2], (x[:,0]-1, x[:,1]-1)), shape=(max(ratings[:,0]),max(ratings[:,1]))).toarray()

def weights_matrices(x):
    """
    Create matrix with values only nonzero at rows/columns with user/movie from training set
    Values are drawn from normal distribution specified by mu and sigma
    """
    np.random.seed(randomseed)
    mu = np.sqrt(np.mean(x[:,2])/num_factors)
    sigma = 1./(num_factors)
    u = sigma*np.random.randn(max(ratings[:,0]), num_factors) + mu
    m = sigma*np.random.randn(num_factors, max(ratings[:,1])) + mu
    return u,m

def squeeze(x, minimum, maximum):
    x = x/(np.max(x)-np.min(x))*(maximum-minimum)
    x = x-(np.min(x)-minimum)
    return x

def plot_error(fold, rmse_train, rmse_test, mae_train, mae_test):
    plt.figure()
    plt.plot(range(len(rmse_train[fold,:])), rmse_train[fold,:], label='rmse train')
    plt.plot(range(len(rmse_train[fold,:])), rmse_test[fold,:], label='rmse test')
    plt.plot(range(len(rmse_train[fold,:])), mae_train[fold,:], label='mae train')
    plt.plot(range(len(rmse_train[fold,:])), mae_test[fold,:], label='mae test')
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.legend()
    plt.savefig('error_%s.png'%fold)
    plt.close()

def plot_hist(x, name):
    plt.figure()
    plt.hist(x)
    plt.savefig('%s.png'%name)

#load data
#ratings=read_data("ratings.dat")
ratings=[]
f = open("datasets/ratings.dat", 'r')
for line in f:
    data = line.split('::')
    ratings.append([int(z) for z in data[:3]])
f.close()
ratings=np.array(ratings)

#split data into 5 train and test folds
nfolds=5

#allocate memory for results:
rmse_train=np.zeros((nfolds, num_iter))
rmse_test=np.zeros((nfolds, num_iter))
mae_train=np.zeros((nfolds, num_iter))
mae_test=np.zeros((nfolds, num_iter))

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(randomseed)

seqs=[x%nfolds for x in range(len(ratings))] #number between 0 and 4 to indicate to which nfold an item in ratings belongs
np.random.shuffle(seqs)

ts = time.time()
print("Start time for loop: " + str(ts))
#for each fold:
for fold in range(nfolds):
    print("Fold ", fold)
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train=ratings[train_sel]
    test=ratings[test_sel]
    
    U, M = weights_matrices(train)
   
    X_train = ratings_matrix(train)
    X_test = ratings_matrix(test)
    
    print("Training..")
    for it in range(num_iter):
        print(it)
        for i in range(len(train[:,0])):
            err = train[i,2] - np.dot(U[train[i,0]-1,:],M[:,train[i,1]-1])
            U[train[i,0]-1,:] += lrate*(2.*err*M[:,train[i,1]-1]-reg*U[train[i,0]-1,:])               
            M[:,train[i,1]-1] += lrate*(2.*err*U[train[i,0]-1,:]-reg*M[:,train[i,1]-1])        
        
        X_pred = np.clip(np.dot(U,M), 1., 5.)
        X_pred_flat = np.array([X_pred[train[i,0]-1,train[i,1]-1] for i in range(len(train[:,0]))])
        print("nr of x_pred elements outside limits:", len(X_pred[np.where(X_pred>5.)]), len(X_pred[np.where(X_pred<0.)]))
        print("nr of x_pred elements with existing rating outside limits:", len(X_pred_flat[np.where(X_pred_flat>5.)]), len(X_pred_flat[np.where(X_pred_flat<0.)]))
        err_train = X_train - X_pred
        err_test = X_test - X_pred
        
        err_train_flat = np.array([err_train[train[i,0]-1,train[i,1]-1] for i in range(len(train[:,0]))])
        err_test_flat = np.array([err_test[test[i,0]-1,test[i,1]-1] for i in range(len(test[:,0]))])
        
        rmse_train[fold,it] = np.sqrt(np.mean(err_train_flat**2))
        rmse_test[fold,it] = np.sqrt(np.mean(err_test_flat**2))
        mae_train[fold,it] = np.mean(np.absolute(err_train_flat))
        mae_test[fold,it] = np.mean(np.absolute(err_test_flat))
        
        
        print("rmse train ", rmse_train[fold,it], "rmse test ", rmse_test[fold,it], "mae train ", mae_train[fold,it], "mae test ", mae_test[fold,it])
        plot_error(fold, rmse_train, rmse_test, mae_train, mae_test)
        
    #print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(rmse_train[fold,-1]) + "; RMSE_test=" + str(rmse_test[fold,-1]) + "; Time = " + str(time.time()-ts))
    print("Fold " + str(fold) + ": MAE_train=" + str(mae_train[fold,-1]) + "; MAE_test=" + str(mae_test[fold,-1]) + "; Time = " + str(time.time()-ts))

np.save('rmse_train', rmse_train)
np.save('rmse_test', rmse_test)

np.save('mae_train', mae_train)
np.save('mae_test', mae_test)

#print the final conclusion:
print("\n")
print("RMSE on TRAIN: " + str(np.mean(rmse_train[:,-1])))
print("RMSE on  TEST: " + str(np.mean(rmse_test[:,-1])))
print("MAE on TRAIN: " + str(np.mean(mae_train[:,-1])))
print("MAE on  TEST: " + str(np.mean(mae_test[:,-1])))

