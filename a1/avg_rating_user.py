import numpy as np
import time

print(time.time())

#load data
#ratings=read_data("ratings.dat")
ratings=[]
f = open("datasets/ratings.dat", 'r')
for line in f:
    data = line.split('::')
    ratings.append([int(z) for z in data[:3]])
f.close()
ratings=np.array(ratings)

"""
Alternatively, instead of reading data file line by line you could use the Numpy
genfromtxt() function. For example:

ratings = np.genfromtxt("ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')

will create an array with 3 columns.

Additionally, you may now save the rating matrix into a binary file 
and later reload it very quickly: study the np.save and np.load functions.
"""


#split data into 5 train and test folds
nfolds=5

#allocate memory for results:
err_train=np.zeros(nfolds)
err_test=np.zeros(nfolds)
err_train_MAE=np.zeros(nfolds)
err_test_MAE=np.zeros(nfolds)

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(9294)

seqs=[x%nfolds for x in range(len(ratings))] #number between 0 and 4 to indicate to which nfold an item in ratings belongs
np.random.shuffle(seqs)

ts = time.time()
print("Start time for loop: " + str(ts))
#for each fold:
for fold in range(nfolds):
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train=ratings[train_sel]
    test=ratings[test_sel]
    """
    gmr_user = np.zeros(np.max(ratings[:,0]))
    for i in range(np.max(ratings[:,0])):
        gmr_user[i] = np.mean(train[train[:,0]==(i+1)][:,2])
    """
    gmr_user = np.array([np.mean(train[train[:,0]==(i+1)][:,2]) for i in range(np.max(ratings[:,0]))])
	#calculate model parameters: mean rating over the training set:
    gmr_user[np.isnan(gmr_user)] = np.mean(train[:,2])
    
    """
    diff = np.zeros(np.max(train[:,2]))
    for j in range(len(train[:,2])):
        diff[j] = train[j,2] - gmr_movie[train[j,1]
    """

    #diff = [(train[j,2] - gmr_movie[train[j,1]-1]) for j in range(len(train[:,2]))]
    #apply the model to the train set:
    err_train[fold]=np.sqrt(np.mean((np.array([(train[j,2] - gmr_user[train[j,0]-1]) for j in range(len(train[:,2]))]))**2))
    err_train_MAE[fold]=np.mean(np.absolute(np.array([(train[j,2] - gmr_user[train[j,0]-1]) for j in range(len(train[:,2]))])))

    #apply the model to the test set:
    err_test[fold]=np.sqrt(np.mean((np.array([(test[j,2] - gmr_user[test[j,0]-1]) for j in range(len(test[:,2]))]))**2))
    err_test_MAE[fold]=np.mean(np.absolute(np.array([(test[j,2] - gmr_user[test[j,0]-1]) for j in range(len(test[:,2]))])))
    
    #print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]) + "; Time = " + str(time.time()-ts))
    print("Fold " + str(fold) + ": MAE_train=" + str(err_train_MAE[fold]) + "; MAE_test=" + str(err_test_MAE[fold]) + "; Time = " + str(time.time()-ts))

#print the final conclusion:
print("\n")
print("RMSE on TRAIN: " + str(np.mean(err_train)))
print("RMSE on  TEST: " + str(np.mean(err_test)))
print("MAE on TRAIN: " + str(np.mean(err_train_MAE)))
print("MAE on  TEST: " + str(np.mean(err_test_MAE)))


