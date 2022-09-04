import torch
import torch.nn
from torchsummary import summary
from importlib import import_module

import backbone
from io_utils import parse_args
from methods.ournet import OurNet

feature_model = backbone.Conv4NP
loss_type = 'mse'
params = parse_args('train')

train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OurNet(feature_model, loss_type=loss_type, **train_few_shot_params)

model = model.to(device)
summary(model, input_size=(3, 84, 84))