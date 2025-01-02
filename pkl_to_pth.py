import pickle
import torch
import sk2torch

pkl_file = 'experiments/ARNIQA_PHAVM_PC_GAMER_20241128_screen/regressors/spaq_srocc_0.9041_plcc_0.9082.pkl'

scikit_regressor = pickle.load(open(pkl_file, "rb"))
torch_regressor = sk2torch.wrap(scikit_regressor).float()
torch.jit.script(torch_regressor).save("experiments/ARNIQA_PHAVM_PC_GAMER_20241128_screen/regressors/regressor_spaq.pth")
