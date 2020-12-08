from glm_saga.elasticnet import * 


from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import r2_score

def toy_example(): 
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

    # calculate maximum lam
    LAMBDA = maximum_reg(ch.from_numpy(X).float(),ch.from_numpy(y).long(), group=False) / max(0.001, ALPHA)
    LAMBDA = LAMBDA / 2

    # Convert to sklearn parameters
    C = 1/(LAMBDA*X.shape[0])

    regr = LogisticRegression(random_state=0, penalty='elasticnet',
                              multi_class='multinomial',
                              l1_ratio=ALPHA, C=C, 
                              solver='saga', max_iter=NITERS, tol=1e-6) 
    # regr = LogisticRegression(random_state=0, penalty='l2',
    #                           multi_class='multinomial', 
    #                           C=C,
    #                           solver='saga', max_iter=NITERS, tol=1e-6)
    # regr = LogisticRegression(random_state=0, penalty='none',
    #                           multi_class='multinomial', 
    #                           C=1/X.shape[0],
    #                           max_iter=NITERS, tol=1e-6)
    regr.fit(X, y)
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
    print(f"sklearn loss: {elastic_loss(linear, X, y, LAMBDA, ALPHA).item()}")


    # sklearn doesn't do group elasticnet
    train(linear, X, y, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=False)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,NUM_FEATURES)).max(1)[1])
    print(f"proximal gradient loss: {elastic_loss(linear, X, y, LAMBDA, ALPHA).item()}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    dataset = IndexedTensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    LAMBDA_loader = maximum_reg_loader(loader, group=False) / max(0.001, ALPHA)
    print("lam check", LAMBDA, LAMBDA_loader)

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    train_spg(linear, loader, 0.1*STEP_SIZE, 2*NITERS, LAMBDA, ALPHA, group=False, verbose=NITERS//10)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,NUM_FEATURES)).max(1)[1])
    print(f"spg loss: {elastic_loss(linear, X, y, LAMBDA, ALPHA).item()}")

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    train_saga(linear, loader, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=False, verbose=NITERS//10)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,NUM_FEATURES)).max(1)[1])
    print(f"saga loss: {elastic_loss(linear, X, y, LAMBDA, ALPHA).item()}")
    
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    # print("Running GLM path")
    # glm_saga(linear, loader, STEP_SIZE, NITERS, ALPHA, group=False, verbose=NITERS//10, tol=1e-4)

def imagenet_example(): 
    # Prepare dataset
    X_tr, y_tr = [],[]
    X_te, y_te = [],[]

    for eps in [0,0.25,1,3,5]: 
        X_tr.append(ch.load(f"data/train_features_{eps}.pth"))
        y_tr.append(ch.load(f"data/train_labels_{eps}.pth"))
        X_te.append(ch.load(f"data/val_features_{eps}.pth"))
        y_te.append(ch.load(f"data/val_labels_{eps}.pth"))

    # Standardize the variables
    X = ch.cat(X_tr,dim=1)
    u = X.mean(1).unsqueeze(1)
    s = X.std(1).unsqueeze(1)
    X = (X - u)/s
    y = y_tr[0]

    NITERS = 50
    NUM_CLASSES=10
    ALPHA = 0.99
    STEP_SIZE = 0.001
    DEVICE = 'cuda'

    # regr = LogisticRegression(random_state=0, penalty='elasticnet',
    #                           multi_class='multinomial',
    #                           alpha=ALPHA, C=1/LAMBDA, 
    #                           solver='saga', max_iter=NITERS, tol=1e-6) 
    # print("sklearn fitting...")
    # regr.fit(X[:1000,:2048].cpu(), y[:1000])
    # print(regr.coef_)
    # print(regr.intercept_)
    # print(regr.predict(np.zeros((1,10))))
    # assert False

    X,y = X.cuda(), y.cuda()
    print(X.size(0))

    LAMBDA = maximum_reg(X,y, group=True) / max(0.001, ALPHA)
    print(f"lam {LAMBDA}")

    linear = nn.Linear(X.size(1),NUM_CLASSES)
    weight = linear.weight
    bias = linear.bias
    # if DEVICE == 'cuda': 
    #     linear = nn.DataParallel(linear)
    linear = linear.to(DEVICE)

    # weight.data.zero_()
    # bias.data.zero_()

    # train(linear, X, y, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=True, verbose=NITERS//10)
    # print(f"proximal gradient loss: {elastic_loss(linear, X, y, LAMBDA, ALPHA).item()}")

    weight.data.zero_()
    bias.data.zero_()

    dataset = IndexedTensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=250, shuffle=True)
    start_time = time.time()
    train_spg(linear, loader, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=True, verbose=NITERS//10)
    print(f"minibatch proximal gradient loss: {elastic_loss(linear, X, y, LAMBDA, ALPHA).item()} | time {time.time()-start_time}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f}) max {weight.max()}")

    weight.data.zero_()
    bias.data.zero_()

    start_time = time.time()
    train_saga(linear, loader, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=True, verbose=NITERS//10, 
               n_ex = X.size(0), n_classes = y.max().item()+1)
    print(f"saga loss: {elastic_loss(linear, X, y, LAMBDA, ALPHA).item()} | time {time.time()-start_time}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f}) max {weight.max()}")

def toy_dataloader_example(): 
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
    encoder = NormalizedRepresentation(loader, model)

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
    # regr = LogisticRegression(random_state=0, penalty='l2',
    #                           multi_class='multinomial', 
    #                           C=C,
    #                           solver='saga', max_iter=NITERS, tol=1e-6)
    # regr = LogisticRegression(random_state=0, penalty='none',
    #                           multi_class='multinomial', 
    #                           C=1/X.shape[0],
    #                           max_iter=NITERS, tol=1e-6)
    regr.fit(Z.numpy(), y.numpy())
    print(regr.coef_)
    print(regr.intercept_)
    print(regr.predict(np.zeros((1,5))))

    linear = nn.Linear(Z.size(1),NUM_CLASSES)
    weight = linear.weight
    bias = linear.bias

    # Print sklearn loss
    weight.data = ch.from_numpy(regr.coef_).float()
    bias.data = ch.from_numpy(regr.intercept_).float()
    print(f"sklearn loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()}")


    # sklearn doesn't do group elasticnet
    train(linear, Z, y, STEP_SIZE, NITERS, LAMBDA, ALPHA, group=False)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,5)).max(1)[1])
    print(f"proximal gradient loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()}")

    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    dataset = IndexedTensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    LAMBDA_loader = maximum_reg_loader(loader, group=False, encoder=encoder) / max(0.001, ALPHA)
    print("lam check", LAMBDA, LAMBDA_loader / 2)

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    train_spg(linear, loader, 0.1*STEP_SIZE, 2*NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, encoder=encoder)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,5)).max(1)[1])
    print(f"spg loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()}")

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()

    alt_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True))
    train_saga(linear, alt_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, encoder=encoder)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,5)).max(1)[1])
    print(f"saga loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA).item()}")
    
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    for p in [weight,bias]: 
        p.data.zero_()
        p.grad.zero_()


    print("sample weighted test")
    # sample_weight = np.arange(Z.size(0))/Z.size(0)
    sample_weight = np.ones((Z.size(0),))
    sample_weight[::2] = 0.5
    regr.fit(Z.numpy(), y.numpy(), sample_weight = sample_weight)
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
    print(f"sklearn weighted loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='multinomial', sample_weight=sample_weight_pth).item()}")

    for p in [weight,bias]: 
        p.data.zero_()

    weighted_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True), sample_weight=sample_weight_pth)
    train_saga(linear, weighted_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, encoder=encoder, family='multinomial', tol=1e-7, lookbehind=100)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,5)).max(1)[1])
    print(f"saga loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='multinomial', sample_weight=sample_weight_pth).item()}")
    

    for p in [weight,bias]: 
        p.data.zero_()

    alt_weighted_loader = DataLoader(IndexedTensorDataset(X,y,sample_weight_pth), batch_size=20, shuffle=True)
    train_saga(linear, alt_weighted_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, encoder=encoder, family='multinomial', tol=1e-7, lookbehind=100)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,5)).max(1)[1])
    print(f"alt saga loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='multinomial', sample_weight=sample_weight_pth).item()}")
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

def toy_regression_example(): 
    NITERS = 2000
    NUM_TARGETS=3
    ALPHA = 0.5
    STEP_SIZE = 0.1
    DEVICE = 'cpu'

    # dataset parameter
    NUM_FEATURES = 10

    X, y = make_regression(n_features=NUM_FEATURES, random_state=0, n_targets=NUM_TARGETS,
        n_informative=3)

    X,y = ch.from_numpy(X).float(), ch.from_numpy(y).float()

    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(NUM_FEATURES, 10), 
        nn.ReLU(), 
        nn.Linear(10,5)
    )
    
    ds = IndexedTensorDataset(X,y)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    encoder = NormalizedRepresentation(loader, model)

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

    regr = ElasticNet(alpha=LAMBDA, l1_ratio=ALPHA, tol=1e-8, max_iter=NITERS)
    # regr = ElasticNet(random_state=0, penalty='elasticnet',
    #                           multi_class='multinomial',
    #                           l1_ratio=ALPHA, C=C, 
    #                           solver='saga', max_iter=NITERS, tol=1e-6) 
    # regr = LogisticRegression(random_state=0, penalty='l2',
    #                           multi_class='multinomial', 
    #                           C=C,
    #                           solver='saga', max_iter=NITERS, tol=1e-6)
    # regr = LogisticRegression(random_state=0, penalty='none',
    #                           multi_class='multinomial', 
    #                           C=1/X.shape[0],
    #                           max_iter=NITERS, tol=1e-6)
    regr.fit(Z.numpy(), y.numpy())
    print(regr.coef_)
    print(regr.intercept_)
    print(regr.predict(np.zeros((1,5))))

    linear = nn.Linear(Z.size(1),NUM_TARGETS)
    weight = linear.weight
    bias = linear.bias

    # Print sklearn loss
    weight.data = ch.from_numpy(regr.coef_).float()
    bias.data = ch.from_numpy(regr.intercept_).float()
    print(f"sklearn loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian').item()}")

    for p in [weight,bias]: 
        p.data.zero_()

    dataset = IndexedTensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    LAMBDA_loader = maximum_reg_loader(loader, group=False, family='gaussian', encoder=encoder) / max(0.001, ALPHA)
    print("lam check", LAMBDA, LAMBDA_loader)

    alt_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True))
    train_saga(linear, alt_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, encoder=encoder, family='gaussian', tol=1e-7, lookbehind=100)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,5)).max(1)[1])
    print(f"saga loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian').item()}")
    
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")

    print("sample weighted test")
    # sample_weight = np.arange(Z.size(0))/Z.size(0)
    sample_weight = np.ones((Z.size(0),))
    sample_weight[::2] = 0.5
    regr.fit(Z.numpy(), y.numpy(), sample_weight = sample_weight)
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
    print(f"sklearn weighted loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian', sample_weight=sample_weight_pth).item()}")

    for p in [weight,bias]: 
        p.data.zero_()

    weighted_loader = add_index_to_dataloader(DataLoader(TensorDataset(X,y), batch_size=20, shuffle=True), sample_weight=sample_weight_pth)
    train_saga(linear, weighted_loader, STEP_SIZE, NITERS, ch.ones(1)*LAMBDA, ALPHA, group=False, verbose=NITERS//10, encoder=encoder, family='gaussian', tol=1e-7, lookbehind=100)
    print(weight)
    print(bias)
    print(linear(ch.zeros(1,5)).max(1)[1])
    print(f"saga loss: {elastic_loss(linear, Z, y, LAMBDA, ALPHA, family='gaussian', sample_weight=sample_weight_pth).item()}")
    
    nnz = (weight.abs() > 1e-5).sum().item()
    total = weight.numel()
    print(f"weight nnz {nnz}/{total} ({nnz/total:.4f})")


    # glm_saga(linear, alt_loader, STEP_SIZE, NITERS, ALPHA, 
    #          table_device=None, encoder=encoder, group=False, 
    #          verbose=None, state=None, n_ex=None, n_classes=None, 
    #          tol=1e-4, epsilon=0.0001, k=100, checkpoint='tmp', 
    #          solver='saga', do_zero=True, lr_decay_factor=50, metadata=None, 
    #          val_loader=loader, test_loader=loader, lookbehind=None, 
    #          family='gaussian') 

if __name__ == "__main__": 
    # toy_example()
    toy_dataloader_example()
    toy_regression_example()
    # imagenet_example()
