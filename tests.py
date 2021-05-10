from glm_saga.elasticnet import * 


from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import r2_score

def toy_example(verbose=False, tol=1e-5): 
    NITERS = 2000
    NUM_CLASSES=3
    ALPHA = 0.5
    STEP_SIZE = 1
    DEVICE = 'cpu'

    # dataset parameter
    NUM_FEATURES = 10

    X, y = make_classification(n_features=NUM_FEATURES, random_state=0, n_classes=NUM_CLASSES,
        n_informative=3)

    # Standardize input
    X = (X-X.mean(0))/X.std(0)

    # calculate maximum lam, use half of maximum for testing purposes
    LAMBDA = maximum_reg(ch.from_numpy(X).float(),ch.from_numpy(y).long(), group=False) / max(0.001, ALPHA)
    LAMBDA = LAMBDA / 2

    # Convert to sklearn parameters
    C = 1/(LAMBDA*X.shape[0])

    regr = LogisticRegression(random_state=0, penalty='elasticnet',
                              multi_class='multinomial',
                              l1_ratio=ALPHA, C=C, 
                              solver='saga', max_iter=NITERS, tol=1e-6) 

    regr.fit(X, y)
    if verbose: 
        print(regr.coef_)
        print(regr.intercept_)
        print(regr.predict(np.zeros((1,10))))

    X,y = ch.from_numpy(X).float(), ch.from_numpy(y).long()

    linear = nn.Linear(X.size(1),NUM_CLASSES)
    weight = linear.weight
    bias = linear.bias

    # Print sklearn loss
    weight.data = ch.from_numpy(regr.coef_).float()
    bias.data = ch.from_numpy(regr.intercept_).float()
    loss_sklearn = elastic_loss(linear, X, y, LAMBDA, ALPHA).item()
    print(f"sklearn loss: {loss_sklearn}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_sklearn = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")


    # note: sklearn doesn't do group elasticnet
    train(linear, X, y, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=False)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,NUM_FEATURES)).max(1)[1])
    loss_pg = elastic_loss(linear, X, y, LAMBDA, ALPHA).item()
    print(f"proximal gradient loss: {loss_pg}")
    assert abs(loss_pg - loss_sklearn) < tol

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_pg = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")
    assert abs(nnz_pg - nnz_sklearn) < tol

    dataset = IndexedTensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    LAMBDA_loader = maximum_reg_loader(loader, group=False) / max(0.001, ALPHA)
    print("lam check", LAMBDA, LAMBDA_loader/2)
    assert abs(LAMBDA - LAMBDA_loader/2) < tol

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    train_spg(linear, loader, 0.1*STEP_SIZE, 2*NITERS, LAMBDA, ALPHA, group=False, verbose=None)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,NUM_FEATURES)).max(1)[1])
    loss_spg = elastic_loss(linear, X, y, LAMBDA, ALPHA).item()
    print(f"spg loss: {loss_spg}")
    assert abs(loss_spg - loss_sklearn) < tol

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    train_saga(linear, loader, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=False, verbose=None)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,NUM_FEATURES)).max(1)[1])
    loss_saga = elastic_loss(linear, X, y, LAMBDA, ALPHA).item()
    print(f"saga loss: {loss_saga}")
    assert abs(loss_saga - loss_sklearn) < tol
    
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_saga = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")
    assert abs(nnz_saga - nnz_sklearn) < tol

def toy_dataloader_example(verbose=False, tol=1e-5): 
    NITERS = 2000
    NUM_CLASSES=3
    ALPHA = 0.5
    STEP_SIZE = 1
    DEVICE = 'cpu'

    # dataset parameter
    NUM_FEATURES = 10

    X, y = make_classification(n_features=NUM_FEATURES, random_state=0, n_classes=NUM_CLASSES,
        n_informative=3)

    X,y = ch.from_numpy(X).float(), ch.from_numpy(y).long()

    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(NUM_FEATURES, 10), 
        nn.ReLU(), 
        nn.Linear(10,5)
    )
    
    ds = IndexedTensorDataset(X,y)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    preprocess = NormalizedRepresentation(loader, model)

    # Standardize input
    with ch.no_grad(): 
        Z = model(X)
        Z = (Z - Z.mean(0))/Z.std(0)

    # calculate maximum lam
    LAMBDA = maximum_reg(Z,y, group=False) / max(0.001, ALPHA)
    # for testing purposes use half of maximum lambda
    LAMBDA = LAMBDA / 2

    # Convert to sklearn parameters
    C = 1/(LAMBDA*X.shape[0])

    regr = LogisticRegression(random_state=0, penalty='elasticnet',
                              multi_class='multinomial',
                              l1_ratio=ALPHA, C=C, 
                              solver='saga', max_iter=NITERS, tol=1e-6) 

    regr.fit(Z.numpy(), y.numpy())
    if verbose: 
        print(regr.coef_)
        print(regr.intercept_)
        print(regr.predict(np.zeros((1,5))))

    linear = nn.Linear(Z.size(1),NUM_CLASSES)
    weight = linear.weight
    bias = linear.bias

    # Print sklearn loss
    weight.data = ch.from_numpy(regr.coef_).float()
    bias.data = ch.from_numpy(regr.intercept_).float()
    loss_sklearn = elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()
    print(f"sklearn loss: {loss_sklearn}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_sklearn = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    # sklearn doesn't do group elasticnet
    train(linear, Z, y, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=False)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,5)).max(1)[1])
    loss_pg = elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()
    print(f"proximal gradient loss: {loss_pg}")
    assert abs(loss_pg - loss_sklearn) < tol


    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_pg = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")
    assert abs(nnz_pg - nnz_sklearn) < tol

    dataset = IndexedTensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    LAMBDA_loader = maximum_reg_loader(loader, group=False, preprocess=preprocess) / max(0.001, ALPHA)
    print("lam check", LAMBDA, LAMBDA_loader / 2)
    assert abs(LAMBDA - LAMBDA_loader/2) < tol

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    train_spg(linear, loader, 0.1*STEP_SIZE, 2*NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=None, preprocess=preprocess)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,5)).max(1)[1])
    loss_spg = elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()
    print(f"spg loss: {loss_spg}")
    assert abs(loss_spg - loss_sklearn) < tol

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    alt_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True))
    train_saga(linear, alt_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=None, preprocess=preprocess)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,5)).max(1)[1])
    loss_saga = elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()
    print(f"saga loss: {loss_saga}")
    assert abs(loss_saga - loss_sklearn) < tol

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_saga = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")
    assert abs(nnz_saga - nnz_sklearn) < tol

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()


    print("==> testing with sample weights")
    # sample_weight = np.arange(Z.size(0))/Z.size(0)
    sample_weight = np.ones((Z.size(0),))
    sample_weight[::2] = 0.5
    regr.fit(Z.numpy(), y.numpy(), sample_weight = sample_weight)
    if verbose: 
        print(regr.coef_)
        print(regr.intercept_)
        print(regr.predict(np.zeros((1,5))))

    linear = nn.Linear(Z.size(1),NUM_CLASSES)
    weight = linear.weight
    bias = linear.bias

    # Print sklearn loss
    weight.data = ch.from_numpy(regr.coef_).float()
    bias.data = ch.from_numpy(regr.intercept_).float()
    sample_weight_pth = ch.from_numpy(sample_weight).float()
    wloss_sklearn = elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='multinomial', sample_weight=sample_weight_pth).item()
    print(f"sklearn weighted loss: {wloss_sklearn}")

    for p in [weight,bias]: 
        p.data.zero_()

    weighted_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True), sample_weight=sample_weight_pth)
    train_saga(linear, weighted_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=None, preprocess=preprocess, family='multinomial', tol=1e-7, lookbehind=100)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,5)).max(1)[1])
    wloss_saga = elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='multinomial', sample_weight=sample_weight_pth).item()
    print(f"saga loss: {wloss_saga}")
    assert abs(wloss_saga - wloss_sklearn) < tol
    

    for p in [weight,bias]: 
        p.data.zero_()

    alt_weighted_loader = DataLoader(IndexedTensorDataset(X,y,sample_weight_pth), batch_size=20, shuffle=True)
    train_saga(linear, alt_weighted_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=None, preprocess=preprocess, family='multinomial', tol=1e-7, lookbehind=100)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,5)).max(1)[1])
    wloss_saga = elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='multinomial', sample_weight=sample_weight_pth).item()
    print(f"alt saga loss: {wloss_saga}")
    assert abs(wloss_saga - wloss_sklearn) < tol

def toy_regression_example(verbose=False, tol=1e-2): 
    # sklearn needs a lot of iterations to converge
    NITERS = 100000
    NUM_TARGETS=3
    ALPHA = 0.9
    STEP_SIZE = 0.1
    DEVICE = 'cpu'

    # dataset parameter
    NUM_FEATURES = 10

    X, y = make_regression(n_features=NUM_FEATURES, random_state=0, n_targets=NUM_TARGETS,
        n_informative=2)

    X,y = ch.from_numpy(X).float(), ch.from_numpy(y).float()

    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(NUM_FEATURES, 10), 
        nn.ReLU(), 
        nn.Linear(10,5)
    )
    
    ds = IndexedTensorDataset(X,y)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    preprocess = NormalizedRepresentation(loader, model)

    # Standardize input
    with ch.no_grad(): 
        Z = model(X)
        Z = (Z - Z.mean(0))/Z.std(0)

    # calculate maximum lam
    LAMBDA = maximum_reg(Z,y, group=False, family='gaussian') / max(0.001, ALPHA)
    # for testing purposes use half of maximum lambda
    LAMBDA = LAMBDA / 2

    # Convert to sklearn parameters
    C = 1/(LAMBDA*X.shape[0])

    regr = ElasticNet(alpha=LAMBDA, l1_ratio=ALPHA, tol=1e-8, max_iter=NITERS, selection='random')

    regr.fit(Z.numpy(), y.numpy())
    if verbose: 
        print(regr.coef_)
        print(regr.intercept_)
        print(regr.predict(np.zeros((1,5))))

    linear = nn.Linear(Z.size(1),NUM_TARGETS)
    weight = linear.weight
    bias = linear.bias

    # Print sklearn loss
    weight.data = ch.from_numpy(regr.coef_).float()
    bias.data = ch.from_numpy(regr.intercept_).float()
    loss_sklearn = elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian').item()
    print(f"sklearn loss: {loss_sklearn}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_sklearn = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    for p in [weight,bias]: 
        p.data.zero_()

    dataset = IndexedTensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    LAMBDA_loader = maximum_reg_loader(loader, group=False, family='gaussian', preprocess=preprocess) / max(0.001, ALPHA)
    print("lam check", LAMBDA, LAMBDA_loader/2)
    assert abs(LAMBDA - LAMBDA_loader/2) < tol

    alt_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True))
    train_saga(linear, alt_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, preprocess=preprocess, family='gaussian', tol=1e-9, lookbehind=100)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,5)).max(1)[1])
    loss_saga = elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian').item()
    print(f"saga loss: {loss_saga}")
    assert abs(loss_saga - loss_sklearn) < tol
    
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    nnz_saga = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")
    assert abs(nnz_saga - nnz_sklearn) < tol

    print("==> testing with sample weights")
    # sample_weight = np.arange(Z.size(0))/Z.size(0)
    sample_weight = np.ones((Z.size(0),))
    sample_weight[::2] = 0.5
    # Elasticnet in sklearn does a thing where sample weights 
    # are automatically rescaled to sum to n_samples. See: 
    # https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/linear_model/_coordinate_descent.py#L791
    # 
    # To make it consistent, we do the same thing here so that 
    # SAGA is seeing the same weights
    sample_weight *= Z.size(0)/(sample_weight.sum())

    regr.fit(Z.numpy(), y.numpy(), sample_weight = sample_weight)
    if verbose: 
        print(regr.coef_)
        print(regr.intercept_)
        print(regr.predict(np.zeros((1,5))))

    linear = nn.Linear(Z.size(1),NUM_TARGETS)
    weight = linear.weight
    bias = linear.bias

    # Print sklearn loss
    weight.data = ch.from_numpy(regr.coef_).float()
    bias.data = ch.from_numpy(regr.intercept_).float()
    sample_weight_pth = ch.from_numpy(sample_weight).float()
    wloss_sklearn = elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian', sample_weight=sample_weight_pth).item()
    print(f"sklearn weighted loss: {wloss_sklearn}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    wnnz_sklearn = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    for p in [weight,bias]: 
        p.data.zero_()

    weighted_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True), sample_weight=sample_weight_pth)
    train_saga(linear, weighted_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, preprocess=preprocess, family='gaussian', tol=1e-7, lookbehind=100)
    if verbose: 
        print(weight)
        print(bias)
        print(linear(ch.zeros(1,5)).max(1)[1])
    wloss_saga = elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian', sample_weight=sample_weight_pth).item()
    print(f"saga loss: {wloss_saga}")
    assert abs(wloss_saga - wloss_sklearn) < tol
    
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    wnnz_saga = nnz/total
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")
    assert abs(wnnz_saga - wnnz_sklearn) < tol

if __name__ == "__main__": 
    print("==> toy example tests")
    toy_example()
    print("==> dataloader tests")
    toy_dataloader_example()
    print("==> regression tests")
    toy_regression_example()