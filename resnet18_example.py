import torch as ch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

# Setup a typical dummy problem, and remove the last fully 
# connected layer of the model to return the representation
N_EX=128
N_CLASSES=10
X,y = ch.randn(N_EX,3,32,32), ch.randint(0,N_CLASSES,(N_EX,))
model = models.resnet18(num_classes=10)
model.fc = nn.Sequential()

# Get a  normalized representations and make an indexed dataloader
# Can use IndexedTensorDataset or add an index to an existing
# dataset with InexedDataset, or add an index to an existing
# dataloader with add_index_to_dataloader
z = model(X)
z = (z - z.mean(0))/z.std(0)
indexed_train_ds = IndexedTensorDataset(z,y)

# You should use a real train/validation/test split here
test_ds = TensorDataset(z,y)

indexed_train_loader = DataLoader(indexed_train_ds, batch_size=32, shuffle=True)
test_loader = val_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Make linear model and zero initialize
linear = nn.Linear(512,10)
linear.weight.data.zero_()
linear.bias.data.zero_()

# Solver parameters
STEP_SIZE=0.1
NITERS=500
ALPHA=0.99
EPSILON=1e-1

# Solve the GLM path
output = glm_saga(linear, indexed_train_loader, STEP_SIZE, NITERS, ALPHA, 
				  val_loader=val_loader, test_loader=test_loader)
path = output["path"]
best_params = output["best"]
print(best_params)

