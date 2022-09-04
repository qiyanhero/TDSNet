import torch
import torch.nn as nn
import itertools
import numpy as np
import torch.nn.functional as F

class TripleLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripleLoss, self).__init__()
        self.margin = margin  # 阈值

    def forward(self, po_dis,ne_dis,DEVICE):
        part=po_dis.shape[1]
        loss=torch.max(ne_dis-po_dis+self.margin,torch.tensor(0.0).to(DEVICE))
        loss=torch.mean(loss)
        return loss

def combination(iterable, r):
    pool = list(iterable)
    n = len(pool)

    for indices in itertools.permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield list(pool[i] for i in indices)


def get_triplets(labels):
    labels = labels.cuda().cpu().data.numpy()
    triplets = []
    #print(labels.size())
    for label in set(labels):
        label_mask = (labels == label)
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        anchor_positives = list(combination(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))


def construct_triplets(image_embedding, image_labels):
    triplets = get_triplets(image_labels)
    if triplets.shape[0]==0:
        return None,None,None
    else:
        anch = triplets[:, 0]
        posi = triplets[:, 1]
        nega = triplets[:, 2]
        anch = image_embedding[anch, :, :]
        posi = image_embedding[posi, :, :]
        nega = image_embedding[nega, :, :]
        return anch, posi, nega


