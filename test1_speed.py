import numpy as np
import torch
from gmm import GMM_torch
from sklearn import mixture
import time
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    # Generate random sample, two components
    n_samples = 20000
    n_init = 3
    max_iter = 30
    n_components = 30
    np.random.seed(0)
    # C = np.array([[0.0, -0.1], [1.7, 0.4]])
    # X_numpy = np.r_[
    #     np.dot(np.random.randn(n_samples, 2), C),
    #     0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
    # ]           
    X_numpy = np.random.rand(n_samples,2)
    X_tensor_cpu = torch.from_numpy(X_numpy)
    X_tensor_gpu = torch.from_numpy(X_numpy).to('cuda')

    # training
    sk_start_time = time.time()
    gmm_sklearn = mixture.GaussianMixture(n_components=n_components,n_init=n_init,max_iter=max_iter, covariance_type="full",tol=1e-20).fit(X_numpy)
    sk_end_time = time.time()

    torch_cpu_start_time = time.time()
    gmm_torch_cpu = GMM_torch(n_components=n_components, total_iter=max_iter,kmeans_iter=n_init)
    gmm_torch_cpu.fit(X_tensor_cpu)
    torch_cpu_end_time = time.time()

    torch_gpu_start_time = time.time()
    gmm_torch_gpu = GMM_torch(n_components=n_components, total_iter=max_iter,kmeans_iter=n_init)
    gmm_torch_gpu.fit(X_tensor_gpu)
    torch_gpu_end_time = time.time()

    print(f'scikit learn training time = \t{sk_end_time - sk_start_time} s')
    print(f'torch cpu learn training time = \t{torch_cpu_end_time - torch_cpu_start_time} s')
    print(f'torch gpu learn training time = \t{torch_gpu_end_time - torch_gpu_start_time} s')