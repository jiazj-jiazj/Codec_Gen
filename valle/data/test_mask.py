import numpy as np
def poisson_sampling(lam=1, size=1):  
    return np.random.poisson(lam, size)

for i in range(20):
    print(poisson_sampling(lam=1, size=10))