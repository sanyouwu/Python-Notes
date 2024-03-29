from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import scipy.linalg as la
from skimage import draw

from scipy.interpolate import griddata
import torch
import torch.nn as nn

##----- -----------------save lineplots 
plt.figure(figsize = (16,4))
for i in range(len(num_ins_list)):
    num_ins = num_ins_list[i]
    ax = plt.subplot(1,len(num_ins_list),i+1)
    
    sns.lineplot(x,eval("att_"+str(num_ins))[:6,0], label = "attention",linestyle ="dashdot",marker = "<")
    sns.lineplot(x,eval("gated_att_"+str(num_ins))[:6,0], label = "gated_attention",linestyle ="dashdot",marker = ">")
    sns.lineplot(x,eval("mean_"+str(num_ins))[:6,0], label = "mean-pooling",linestyle ="dashdot",marker = "*")
    sns.lineplot(x,eval("IRD_"+str(num_ins))[:,0], label = "IRD",linestyle ="dashdot", marker = "o")#,color = "blue")
    plt.xlabel("Number of training bags")
    plt.xticks(x,rotation=45) 
    # ax.set_xticklabels(x, rotation=45);
    plt.legend()
    plt.title("Auc: " + str(num_ins) + " instances per bag")
save_dir = r"C:\\Users\\sanyo\\OneDrive - connect.hku.hk\\Codes\\IndividualizedRegionSelectionMIL\\res_data"
plt.savefig(Path(save_dir,"MNIST_ss.pdf"), dpi = 300, bbox_inches = "tight", pad_inches = 0)

## ----------- another example for plot grid 
fig,ax = plt.subplots(figsize = (8,4))
ax = plt.subplot(1,2,1)
plt.imshow(ori_test[k],"gray");
ax = plt.subplot(1,2,2)
plt.imshow(np.array(g_test[k]),"gray");
grid_size = (d1,d2) 
ax.xaxis.set_major_locator(ticker.IndexLocator(offset=0, base=d1));
ax.yaxis.set_major_locator(ticker.IndexLocator(offset=0, base=d2));
ax.grid(which='major', axis='both', linewidth=0.7, linestyle='-.', color='w');
ax.tick_params(bottom=False, top=False, left=False, right=False);
ax.set_xticklabels([]);
ax.set_yticklabels([]);
