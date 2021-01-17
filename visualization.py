import os
import pickle 

import cv2
import numpy as np

import networkx as nx
from matplotlib import pyplot as plt

from src.preproc import Preproc
from src.learning import train_image, sparsify
from src.inference import LoopyBPInference


clean_image_array = cv2.resize(cv2.imread("data/clean.bmp", 0), (112, 112))
clean_image = np.pad(clean_image_array, pad_width=tuple([(p, p) for p in (44, 44)]),
                    mode='constant', constant_values=0)

occluded_image_array = cv2.imread("data/occluded.png", 0)
occluded_image = np.pad(occluded_image_array, pad_width=tuple([(p, p) for p in (44, 44)]),
                    mode='constant', constant_values=0)

model_factors = train_image(clean_image)

frcs, graph = model_factors[0], model_factors[2]

points = [(r, c) for (f, r, c) in frcs]

preproc_layer = Preproc()
edge_map = preproc_layer.fwd_infer(clean_image)
edge_map = edge_map.max(0)
edge_map = 255 * (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())

frcs_map = np.zeros_like(edge_map)
for (_, r, c) in frcs:
    frcs_map[r][c] = 1

frcs_map = 255 * (frcs_map - frcs_map.min()) / (frcs_map.max() - frcs_map.min())

bu_msg = preproc_layer.fwd_infer(occluded_image)

lbp = LoopyBPInference(bu_msg, frcs, model_factors[1], (25, 25), preproc_layer, n_iters=1000)
_, bt = lbp.bwd_pass()

bt = 255 * (bt - bt.min()) / (bt.max() - bt.min())

cv2.imwrite("images/edge_map.png", edge_map)
cv2.imwrite("images/frcs_map.png", frcs_map)
cv2.imwrite("images/backtrace.png", bt)

fig, ax = plt.subplots()
nx.draw(graph, pos=points)
plt.show()