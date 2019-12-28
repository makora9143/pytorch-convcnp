import sklearn.gaussian_process as gp


def oracle_gp(xc, yc, xt):
    kernel = gp.kernels.ConstantKernel() * gp.kernels.RBF()
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=1e-5)
    model.fit(xc[0].cpu().numpy(), yc[0].cpu().numpy())
    y_pred, std = model.predict(xt[0].cpu().numpy(), return_std=True)
    return y_pred, std
