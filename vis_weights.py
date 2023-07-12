#%%
import matplotlib.pyplot as plt
import torch

#%%
dataset = 'davis'
cp = torch.load(f'models/model_GNN_{dataset}_t2_msa_50E.model', map_location='cpu') # loading t2 model
plt.matshow(cp['pro_conv1.lin.weight'])
plt.title('msa pro_conv1.lin.weight')
plt.show()

cp = torch.load(f'models/model_GNN_{dataset}_t2_nomsa_50E.model', map_location='cpu') # loading t2 model
plt.matshow(cp['pro_conv1.lin.weight'])
plt.title('nomsa pro_conv1.lin.weight')
plt.show()

# %%
