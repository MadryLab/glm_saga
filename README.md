# SAGA-based GPU solver for elastic net problems
*A package for fitting sparse linear models at deep learning scales. Created by [Eric Wong](https://riceric22.github.io/). This work was initially created and described in our paper, ["Leveraging Sparse Linear Layers for Debuggable Deep Networks"](TODO) with [Shibani Santurkar](https://people.csail.mit.edu/shibani/) and [Aleksander Madry](https://people.csail.mit.edu/madry/).*

This package implements a SAGA-based solver in PyTorch for fitting sparse linear models with elastic net regularization. It combines the path algorithm used by [`glmnet`](https://glmnet.stanford.edu/) with a [minibatch variant of the SAGA algorithm](https://arxiv.org/abs/1902.00071), which allows solving the elastic net at ImageNet scales, which coordinate descent-based elastic net solvers struggle with. 

### Citation
If you find this solver to be useful in your work, consider citing our paper: 
```
TODO
```

## Installation
This package is on PyPI. Install it with `pip install glm_saga`. The only requirement is PyTorch. 

## Usage and documentation
The following function is the main interface which can be used to fit a sequence of sparse linear models. A barebones example which fits a sparse linear model on top of a ResNet18 can be found in `resnet18_example.py`. 

```
def glm_saga(linear, loader, max_lr, nepochs, alpha, 
             table_device=None, precompute=None, group=False, 
             verbose=None, state=None, n_ex=None, n_classes=None, 
             tol=1e-4, epsilon=0.001, k=100, checkpoint=None, 
             solver='saga', do_zero=True, lr_decay_factor=1, metadata=None, 
             val_loader=None, test_loader=None, lookbehind=None, 
             family='multinomial'):
```
### Required arguments
+ `linear`: a PyTorch `nn.Linear` module which the solver initializes from (initialize this to zero)
+ `loader`: a dataloader which returns examples in the form `(X,y,i)` where `X` is a batch of features, `y` is a batch of labels, and `i` is a batch of indices which uniquely identify each example. *Important: the features must be normalized (zero mean and unit variance) and the index is necessary for the solver*. Optionally, the dataloader can also return `(X,y,i,w)` where `w` is the sample weight. 
+ `max_lr`: the starting learning rate to use for the SAGA solver at the starting regularization
+ `nepochs`: the maximum number of epochs to run the SAGA solver for each step of regularization
+ `alpha`: a hyperparameter for elastic net regularization which controls the tradeoff between L1 and L2 regularization (typically taken to be 0.8 or 0.99). `alpha=1` corresponds to only L1 regularization, whereas `alpha=0` corresponds to only L2 regularization. 

### Optional arguments
+ `table_device=None`: if specified, manually stores the SAGA gradient table on the specified device (otherwise, defaults to the device fo the given model). Useful for reducing memory usage. 
+ `precompute=None`: If specified, passes each example from the loader through `precompute`, which is assumed to be a PyTorch `nn.Module`. This can be used to normalize the data if the dataloaders are not already normalized. 
+ `group=False`: If true, use the grouped LASSO where groups are all parameters for a given feature. If false, use standard LASSO. 
+ `verbose=None`: If set to an integer, print the status of the inner GLM solver every `verbose` iterations. 
+ `state=None`: If specified, a previous state of the SAGA solver to continue from (gradient table and averages). Otherwise, the state will be initialized at zero
+ `n_ex=None`: The total number of examples in the dataloader. If not specified, a single pass will be made over the dataloader to count the total number of examples. 
+ `n_classes=None`: the total number of classes in the dataloader. If not specified, a single pass will be made over the dataloader to count the total number of classes. 
+ `tol=1e-4`: The tolerance level for the stopping criteria of the SAGA solver
+ `epsilon=0.001`: The regularization path will be calculated at log-spaced intervals between `max_lambda` and `max_lambda*epsilon`, where `max_lambda` is calculated to be the smallest regularization which results in the all zero solution. The elastic-net paper recommends `epsilon=0.001` 
+ `k=100`: The number of steps to take along the regularization path
+ `checkpoint=None`: If specified, save the weights and solver logs for each point of the regularization path within the directory `checkpoint` (makes the directory if it does not exist)
+ `solve='saga'`: A string which specifies a particular solver to use (stochastic proximal gradient via `solver=spg` is experimental and not recommended)
+ `do_zero=True`: If true, at the end of the regularization path calculate one more solution corresponding to zero regularization (i.e. fully dense linear model)
+ `lr_decay_factor=50`: The learning rate of solver will be decayed from `max_lr` to `max_lr/lr_decay_factor`. Adjust this value to be smaller if progress stalls before reaching an optimal solution, or adjust this value to be larger if the solution path is unstable. 
+ `metadata=None`: a dictionary which contains metadata about the representation which can be used instead of `n_ex` and `n_classes`. See below for the assumed structure. 
+ `val_loader=None`: If specified, will calculate statistics (loss and accuracy) and perform model selection based on the given validation set 
+ `test_loader=None`: If specified, will calculate statistics (loss and accuracy) on the given test set
+ `lookbehind`: The stopping criterion strategy. If `None`, the solver will stop when progress within an interation is less than `tol`. If specified as an integer, the solver will stop when `tol` progress has not been made for more than `lookbehind` steps. The second is more accurate, but will typically take longer. 
+ `family='multinomial'`: The distribution family for the GLM. Supported familes are `multinomial` and `gaussian`

For extremely large datasets like ImageNet, it may be desired to avoid multiple initial passes over the dataloader which only calculates dataset statistics. This can be done by supplying the `metadata` variable, which is of the following form: 

```
metadata = {
    'X' : {
        'mean': 0, 
        'std': 1
    },
    'y' : {
        'mean': 0, 
        'std': 1
    }, 
    'max_reg': {
        'grouped': 0.5, 
        'ungrouped': 0.5
    }
}
```
Any metadata supplied through this variable will not be recomputed. Not all variables need to be specified (i.e. it is possible to supply only the mean and standard deviation, and still perform one pass to calculate the maximum regularization). 

## Additional helper functions
The package also implements several additional functions which are helpful in order to adapt datasets to the format required by the solver, such as adding example indices and normalizers for the data. 

### Adding indices to datasets and dataloaders
```
IndexedTensorDataset(TensorDataset): 
    def __init__(self, *tensors): 
```
+ A subclass of the PyTorch `TensorDataset` which returns the tensor indices in addition

```
class IndexedDataset(Dataset): 
    def __init__(self, ds, sample_weight=None): 
```
+ A `Dataset` wrapper which takes a PyTorch `Dataset` which returns the dataset indices in addition
+ `sample_weight=None` can be specified to weight each example differently (e.g. for passing to LIME)

```
add_index_to_dataloader(loader, sample_weight=None): 
```
+ A function which takes a dataloader and returns a new dataloader which returns the dataloader indices in addition
+ `sample_weight=None` can be specified to weight each example differently

### Normalizing datasets
Sometimes a PyTorch dataset or dataloader is not easy to normalize directly. In this case, we can construct a normalizing PyTorch module and pass this into the solver via the `preprocess` argument. 
```
class NormalizedRepresentation(nn.Module): 
    def __init__(self, loader, model=None, do_tqdm=True, mean=None, std=None, metadata=None, device='cuda'): 
```
+ A module which normalizes inputs by the mean and standard deviation of the given dataloader
+ `model=None` If specified, examples will be passed through the given `model` before calculating the mean and standard deviation
+ `do_tqdm=True`: If true, use `tqdm` progress bars
+ `mean=None`: If specified, uses this as the mean instead of calculating the mean from the dataloader
+ `std=None`: If specified, uses this as the standard deviation instead of calculating the standard deviation from the dataloader
+ `metadata=None`: If specified, uses dataset statistics from the given dictionary
+ `device='cuda'`: The device to store the mean and standard deviation on

## Tests
A number of tests are in `tests.py`, which match the output of the implemented solve to solvers that exist in `scikit-learn`. 