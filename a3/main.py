import numpy as np
from argparse import ArgumentParser
import scipy as sp
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import time
import itertools

parser = ArgumentParser()
parser.add_argument("seed", help = "The random seed to use", default = 17, type = int)
parser.add_argument("filename", help = "Location of movie_users", default='/disks/strw9/vanweenk/AiDM/Assignment3/user_movie.npy', type = str)
args = parser.parse_args()

def jaccard_similarity(a, b):
    """
    Determine the Jaccard Similarity of binary array a and array b.
    Input
        a - binary array
        b - binary array
    Return 
        len(intersection(a,b)) / len(union(a,b))
    """
    return np.sum(a.multiply(b))/np.sum(a+b)

def corresponding_signatures(a,b):
    """
    Determine the percentage of corresponding signatures
    Input
        a - int array (list of signatures)
        b - int array (list of signatures)
    Return 
        len(intersection(a,b)) / len(union(a,b))
    """
    return len(np.where(a==b)[0])/len(a)

def load_data(filename):
    """
    Load the data consisting of user-movie combinations.
    Create a binary scipy sparse matrix of size (movies, users).
    where element (movie_i, user_i) is True if the combination exists, else False.
    Input
        filename - location + name of user-movie file
    Return
        user_movie  - user_movie input array
        nr_users    - total number of users
        nr_movies   - total number of movies
        csr         - binary sparse (row) matrix of user-movie combinations
        csc         - binary sparse (column) matrix of user-movie combinations
    """
    print("Loading data..")
    #load user_movie numpy array
    user_movie = np.load(filename)
    
    #determine number of users and movies in file
    nr_users = np.max(user_movie[:,0])+1
    nr_movies = np.max(user_movie[:,1])+1
    
    #convert to user_movie array to scipy sparse array 
    #binary matrix (movies, users) consisting of 1s where the user-movie combination exists, and 0s where the user-movie combination does not exist
    ones = np.ones(len(user_movie[:,0]), dtype = bool)
    coo = coo_matrix((ones, (user_movie[:,1], user_movie[:,0])), dtype = bool)
    csr = csr_matrix(coo)
    csc = csc_matrix(csr)
    
    return user_movie, nr_users, nr_movies, csr, csc

def minhashing(nr_users, nr_movies, csr, it=150):
    """
    Implement minhashing to create signatures.
    Create 'it' random permutations of the number of movies.
    Use the list of movies of a given user as a mask for the random permutations.
    Take the minimum of each masked permutation, giving the signature for this user.
    Input
        nr_users    - total number of users
        nr_movies   - total number of movies
        csr         - binary sparse matrix of user-movie combinations
        it          - (optional, default=150) length of user signatures (also, number of permutations)
    Return
        sigs        - return int array of size (it, nr_users) containing signature for each user
    """
    print("Minhashing: Creating signatures..")
    sigs = np.zeros((it,nr_users), dtype=int)
    perm = np.array([np.random.permutation(nr_movies) for i in range(it)])
    for index, u in enumerate(csr.T):
        sigs[:,index] = perm[:,u.nonzero()[1]].min(axis=1).T
    return sigs

def lsh(sigs, rows = 6):
    """
    Implement Locality Sensitive Hashing (LSH) to find candidate pairs of users.
    User pairs are candidates if they have at least one similar signature band.
    Similar signatures bands are indicated by the same value in the np.unique inverse array.
    Split the array into multiplicates of users that have the same band.
    Input
        sigs        - signatures array of size (it, nr_users)
        rows        - (optional, default=6) number of rows that one band consists of
    Return
        cands       - array with candidate pairs
    """
    print("LSH..")
    cands = [] #array with candidate pairs
    for i in range(np.int(np.ceil(sigs.shape[0]/rows))):
        #Take the ith band from the signature matrix
        band = sigs[(i*rows):(i+1)*rows,:]

        #ids is the inverse array of np.unique
        #users have the same id if their signature bands are similar
        unique, ids = np.unique(sigs[(i*rows):(i+1)*rows,:], axis=1, return_inverse=True)
        
        #Split the array into multiplicates of users that have the same band        
        sidx = ids.argsort()
        sorted_ids = ids[sidx]
        buckets = np.array(np.split(sidx, np.nonzero(sorted_ids[1:] > sorted_ids[:-1])[0] + 1))
        
        #Append the duplicates immediately to the candidates array
        #For multiplicates, find the number of combinations
        for b in buckets:
            if len(b) == 2:
                cands.append(np.sort(b))
            if len(b) > 2:
                combs_b = list(map(list, itertools.combinations(b,2)))
                for j in combs_b:
                    cands.append(np.sort(np.array(j)))
    
    return cands
    
def calc_similarity(cands, sigs, csc, thres_sig = .46, thres = .5):   
    """
    Make a preselection on the candidates by comparing their signatures.
    Users are a new candidate if their signatures are 'thres_sig'*100% similar.
    Of the new candidates, calculate the Jaccard similarity and save the results in 'results.txt'
    Input
        cands       - array with candidate pairs
        sigs        - signatures array of size (it, nr_users)
        csc         - binary sparse (column) matrix of user-movie combinations
        thres_sig   - (optional, default=.46)
        thres       - (optional, default=.5)
    """
    print("Calculating similarities..")
    new_candidates = [] #
    similar = [] #array of similar users
    for count, c in enumerate(cands):
        if corresponding_signatures(sigs[:,c[0]], sigs[:,c[1]]) > thres_sig:
            new_candidates.append(c)
            movies_a = csc.getcol(c[0])
            movies_b = csc.getcol(c[1])
            sim = jaccard_similarity(movies_a, movies_b)
            if sim >= .5:
                similar.append([c[0], c[1]])
                if type(similar[0]) == list:
                    similar = np.unique(np.array(similar).T, axis=1).T.tolist()
                np.savetxt('results.txt', similar, delimiter=',', fmt='%i')
    print(len(cands), " candidates ", len(new_candidates), " new candidates ", len(similar), " similar users")

if __name__ == '__main__':
    begin = time.time()
    np.random.seed(seed=args.seed)
    userMovie, nrUsers, nrMovies, CSR, CSC = load_data(args.filename)
    print("Time: ", time.time()-begin)
    signatures = minhashing(nrUsers, nrMovies, CSR)
    print("Time: ", time.time()-begin)
    candidates = lsh(signatures)
    print("Time: ", time.time()-begin)
    calc_similarity(candidates, signatures, CSC)
    print("Total time elapsed: ", time.time()-begin)

            









