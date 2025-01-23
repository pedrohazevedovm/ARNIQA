import pickle
import torch
import sk2torch

pkl_file = 'experiments/'

scikit_regressor = pickle.load(open(pkl_file, "rb"))
torch_regressor = sk2torch.wrap(scikit_regressor).float()
torch.jit.script(torch_regressor).save("experiments/")
