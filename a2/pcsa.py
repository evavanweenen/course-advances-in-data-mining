import random
import numpy as np
import time

groups = [5, 10, 15, 20, 25, 30, 35, 40]
phi = 0.77352

def create_data(seed, m):
    random.seed(seed)    
    return [random.getrandbits(32) for i in range(int(m))]

def trailing_zeroes(num):
    """Counts the number of trailing 0 bits in num."""
    if num == 0:
        return 32 # Assumes 32 bit integer inputs!
    p = 0
    while (num >> p) & 1 == 0:
        p += 1
    return p

def estimate_cardinality(data):
    """Estimates the number of unique elements in the input set values
    Arguments:
        values: An iterator of hashable elements to estimate the cardinality of.
        
    """
    max_zeroes = max([trailing_zeroes(d) for d in data])
    return 2 ** float(max_zeroes)

def stochastic_averaging(est, num_groups, group_length):
    group_avg = [np.mean(est[g*group_length:(g+1)*group_length]) for g in range(num_groups)]
    return np.median(group_avg)

def rae(true,est):
    return np.absolute(true-est)/true

exp_value = np.array([1e2,1e3,1e4,1e5,1e6,1e7])#,1e7,1e8]) #m number of distinct elements

save_err = np.zeros((len(exp_value), len(groups)))

for idxg, g in enumerate(groups):
    print("num_groups", g)    
    for idxm, m in enumerate(exp_value):
        ts = time.time()    
        seeds = g * int(2*np.log2(m))    
        estimate = phi*stochastic_averaging(np.array([estimate_cardinality(np.array(create_data(s, m))) for s in range(seeds)]), g, int(2*np.log2(m)))
        error = rae(m,estimate)
        dtime = time.time()-ts
        print("Expected value (m): " + str(int(m)))
        print("Error: " + str(error))
        print("Time = " + str(dtime))
        print("")
        save_err[idxm, idxg] = round(error,4)

    np.save('save_err', save_err)
np.save('groups', groups)
np.save('exp_value', exp_value)

