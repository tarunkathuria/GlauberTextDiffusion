import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
# sys.path.insert(0, '../')
from config.eval.cifar10 import get_config as get_eval_config
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils

# %matplotlib inline

eval_cfg = get_eval_config()
print(eval_cfg)
input()
train_cfg = bookkeeping.load_ml_collections(Path(eval_cfg.train_config_path))

for item in eval_cfg.train_config_overrides:
    utils.set_in_nested_dict(train_cfg, item[0], item[1])

S = train_cfg.data.S
device = torch.device(eval_cfg.device)

model = model_utils.create_model(train_cfg, device)

loaded_state = torch.load(Path(eval_cfg.checkpoint_path),
    map_location=device)

modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
model.load_state_dict(modified_model_state)

model.eval()

def imgtrans(x):
    x = np.transpose(x, (1,2,0))
    return x

num_samples = 1
sampler = sampling_utils.get_sampler(eval_cfg)
samples, x_hist, x0_hist = sampler.sample(model, 1, 10)

samples = samples.reshape(num_samples, 3, 32, 32)
x_hist = x_hist.reshape(10, num_samples, 3, 32, 32)
x0_hist = x0_hist.reshape(10, num_samples, 3, 32, 32)

idx = 0
# plt.imsave("sample.png", imgtrans(samples[idx, ...]))
plt.imsave("sample.png", imgtrans(samples[idx, ...]).astype(np.uint8))
# plt.show()