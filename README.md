# gmm_torch
use PyTorch batch feature to speed up GMM training

# install
scikit-learn <br />
pytorch <br />
numpy <br />
matplotlib <br />

# compare with scikit-learn

python3 test1_speed.py <br />

loop = 99 <br />

scikit learn training time = 0.6259 +- 0.0150 s <br />
torch cpu learn training time = 0.0990 +- 0.0079 s <br />
torch gpu learn training time = 0.0388 +- 0.0562 s <br />