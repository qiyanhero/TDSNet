import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager

from io_utils import parse_args, get_best_file

def save_features(model, data_loader, outfile ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats1,_= model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats1.size()[1:]) , dtype='f')
        all_feats[count:count+feats1.size(0)] = feats1.data.cpu().numpy()
        #all_feats[count+feats1.size(0):count+feats1.size(0)+feats1.size(0)] = feats2.data.cpu().numpy()
        all_labels[count:count+feats1.size(0)] = y.cpu().numpy()
        #all_labels[count+feats1.size(0):count+feats1.size(0)+feats1.size(0)] = y.cpu().numpy()

        count = count + feats1.size(0)
        #print(feats1.size())
        #print(x[0].type())
        # plt.imshow(feats1[0])
        # heatmap = np.mean(feats1[0].data.cpu().numpy(), axis=-1)
        # heatmap = np.maximum(heatmap, 0)  # heatmap与0比较，取其大者
        # heatmap /= np.max(heatmap)
        # heatmap = cv2.resize(heatmap,(84,84))
        # heatmap = np.uint8(255*heatmap)
        # heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        # # x[0] = x[0].transpose(0, 1)
        # img = heatmap * 0.4 + x[0]
        # cv2.imshow('img',img)
        # vutils.save_image(img, '4.png')
        # vutils.save_image(x_var[0].data.cpu(), '3.png')

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    params = parse_args('save_features')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)

    image_size = 84 


    split = params.split
    
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)

    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)



    modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints", "features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5") 

    datamgr         = SimpleDataManager(image_size, batch_size = 64)
    data_loader      = datamgr.get_data_loader(loadfile, aug = False)

    model = backbone.Conv4NP()




    model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    model.load_state_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile)
