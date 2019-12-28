import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fastprogress import master_bar, progress_bar

from convcnp import ConvCNP1d
from gp import oracle_gp
from dataset import Synthetic1D
from kernels import eq_kernel, matern_kernel, periodic_kernel
from visualize import plot_all, convert_tfboard

torch.set_default_dtype(torch.float64)


def train(model, dataloader, optimizer):
    model.train()
    avg_loss = 0

    for i, (xc, yc, xt, yt) in enumerate(progress_bar(dataloader, parent=args.mb)):
        xc, yc, xt, yt = xc.to(args.device), yc.to(args.device), xt.to(args.device), yt.to(args.device)

        optimizer.zero_grad()

        pred_dist = model(xc, yc, xt)

        loss = - pred_dist.log_prob(yt).sum(-1).mean()

        loss.backward()
        optimizer.step()

        avg_loss -= loss.item() * xc.size(0)

    return avg_loss / len(dataloader.dataset)


def validate(model, dataloader):
    model.eval()

    xc, yc, xt, yt = iter(dataloader).next()
    out_range_xt = torch.linspace(-4, 4, 401).reshape(1, -1, 1).to(args.device)
    xc, yc, xt, yt = xc.to(args.device), yc.to(args.device), xt.to(args.device), yt.to(args.device)

    gp_pred = oracle_gp(xc, yc, xt)
    out_range_gp_pred = oracle_gp(xc, yc, out_range_xt)

    with torch.no_grad():
        pred_dist = model(xc, yc, xt)
        out_range_pred_dist = model(xc, yc, out_range_xt)

    loss = - pred_dist.log_prob(yt).sum(-1).mean()
    rmse = (pred_dist.mean - yt).pow(2).sum(-1).mean()

    image = plot_all(xc, yc, xt, yt, xt, gp_pred, pred_dist)
    image = convert_tfboard(image)
    out_range_image = plot_all(xc, yc,
                               xt, yt,
                               out_range_xt, out_range_gp_pred, out_range_pred_dist,
                               support=True)
    out_range_image = convert_tfboard(out_range_image)
    return loss, rmse, image, out_range_image


def main():
    trainset = Synthetic1D(args.kernel, train=True)
    testset = Synthetic1D(args.kernel, train=False, num_total_max=15)
    trainset.set_length(args.batch_size)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # cnp = ConditionalNeuralProcess(1, 1, 128, data_noise=args.data_noise)
    cnp = ConvCNP1d()
    cnp = cnp.to(args.device)

    optimizer = optim.Adam(cnp.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    args.mb = master_bar(range(1, args.epochs + 1))

    for epoch in args.mb:
        avg_train_loss = train(cnp, trainloader, optimizer)
        valid_loss, rmse, image, out_range_image = validate(cnp, testloader)

        args.writer.add_scalar('params/lengthscale', cnp.psi.base_kernel.lengthscale.item(), epoch)
        args.writer.add_scalar('params/rho_lengthscale', cnp.psi_rho.base_kernel.lengthscale.item(), epoch)
        args.writer.add_scalar('params/outputscale', cnp.psi.outputscale.item(), epoch)
        args.writer.add_scalar('params/rho_outputscale', cnp.psi_rho.outputscale.item(), epoch)
        args.writer.add_scalar('train/likelihood', avg_train_loss, epoch)
        args.writer.add_scalar('validate/likelihood', valid_loss, epoch)
        args.writer.add_scalar('validate/rmse', rmse, epoch)
        args.writer.add_image('validate/image', image, epoch)
        args.writer.add_image('validate/out_range_image', out_range_image, epoch)
    torch.save(cnp.state_dict(), filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-B', type=int, default=16)
    parser.add_argument('--learning-rate', '-LR', type=float, default=1e-3)
    parser.add_argument('--epochs', '-E', type=int, default=200)
    parser.add_argument('--kernel', '-K', type=str, default='eq', choices=['eq', 'matern', 'periodic'])
    parser.add_argument('--data-noise', default=False, action='store_true')
    parser.add_argument('--logging', default=False, action='store_true')

    args = parser.parse_args()

    filename = 'convcnp_{}.pth.gz'.format(args.kernel)
    if args.kernel == 'eq':
        args.kernel = eq_kernel
    elif args.kernel == 'matern':
        args.kernel = matern_kernel
    elif args.kernel == 'periodic':
        args.kernel = periodic_kernel
    else:
        raise NotImplementedError()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    args.writer = SummaryWriter()
    main()
    args.writer.close()