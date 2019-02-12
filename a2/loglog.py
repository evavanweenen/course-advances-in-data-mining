import random
import numpy as np
import time

def trailing_zeroes(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

def estimate_cardinality(values, k):
  """Estimates the number of unique elements in the input set values.

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
    k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
  """
  num_buckets = 2 ** k
  max_zeroes = [0] * num_buckets
  for value in values:
    bucket = value & (num_buckets - 1) # Mask out the k least significant bits as bucket ID
    bucket_hash = value >> k
    max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(bucket_hash))
  return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402

def rae(true,est):
    return np.absolute(true-est)/true

exp_value = np.array([1e3,1e4,1e5,1e6,1e7,1e8])
bucket = np.arange(2,17,1)

save_err = np.zeros((len(exp_value),len(bucket)))
save_time = np.zeros((len(exp_value),len(bucket)))

random.seed(9294)
for idxk, k in enumerate(exp_value):
    for idxl, l in enumerate(bucket):
        ts = time.time()
        estimate = np.array([estimate_cardinality([random.getrandbits(32) for i in range(int(k))], int(l)) for j in np.arange(25)])
        error = rae(k,estimate)
        meanerr = np.mean(error)
        dtime = time.time()-ts
        print("Expected value: " + str(int(k)) + ", buckets: " + str(2**l))
        print("Error: " + str(meanerr))
        print("Time = " + str(dtime))
        print("")
        save_err[idxk,idxl] = round(meanerr,4)
        save_time[idxk,idxl] = round(dtime,4)

np.save('save_err2', save_err)
np.save('save_time2', save_time)
np.save('exp_value2', exp_value)
np.save('bucket2', bucket)



