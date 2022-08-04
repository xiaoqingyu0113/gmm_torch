# gmm_torch
use PyTorch batch feature to speed up GMM training

# install
scikit-learn
pytorch
numpy
matplotlib

# compare with scikit-learn

python3 test1_speed.py

loop = 99

scikit learn training time = 0.6259 +- 0.0150 s
torch cpu learn training time = 0.0990 +- 0.0079 s
torch gpu learn training time = 0.0388 +- 0.0562 s