import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
import sobol
from concurrent.futures import ProcessPoolExecutor
from Minimization import minimization
np.set_printoptions(suppress = True)

obs_series = pd.read_csv('data.csv', delimiter=',')
obs_series = np.array(obs_series.iloc[:,1:]).T

datadir = os.path.dirname(os.getcwd()) + '/sobol_sequence/'
outputdir = os.path.dirname(os.getcwd()) + '/opt/'

block = 28

with open(datadir + 'sobol_'+str(block)+'.pkl','rb') as f:
    sobol_sequence = pickle.load(f)

n_points = sobol_sequence.shape[0]
start = np.array_split(sobol_sequence, n_points)

start_time = time.time()
if __name__ == '__main__':
    with ProcessPoolExecutor() as pool:
        results = pool.map(minimization, start)
    results = [r for r in results]

run_time = time.time() - start_time
print(block)
print(run_time)
with open(outputdir + 'output_'+str(block)+'.pkl', 'wb') as f:
       pickle.dump(results, f)