import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as tf
from torchvision.datasets import MNIST, CIFAR10

from fastprogress import master_bar, progress_bar

from convcnp import ConvCNP2d, channel_last
from visualize import plot_all_2d, convert_tfboard


def train(model, dataloader, optimizer):
    model.train()
    avg_loss = 0

    for index, (I, _) in enumerate(progress_bar(dataloader, parent=args.mb)):
        I = I.to(args.device)

        optimizer.zero_grad()

        pred_dist = model(I)

        loss = - pred_dist.log_prob(channel_last((I))).sum(-1).mean()

        loss.backward()
        optimizer.step()

        avg_loss -= loss.item() * I.size(0)
        if index % 10 == 0:
            args.mb.child.comment = 'loss={:.3f}'.format(loss.item())

    return avg_loss / len(dataloader.dataset)


def validate(model, dataloader):
    model.eval()

    I, _ = iter(dataloader).next()
    I = I.to(args.device)

    with torch.no_grad():
        Mc, f, dist = model.complete(I)

    likelihood = dist.log_prob(channel_last(I)).sum(-1).mean()
    rmse = (I - f).pow(2).mean()

    image = plot_all_2d(I, Mc, f)
    image = convert_tfboard(image)
    return likelihood, rmse, image


def main():
    if args.dataset == 'mnist':
        trainset = MNIST('~/data/mnist', train=True, transform=tf.ToTensor())
        testset = MNIST('~/data/mnist', train=False, transform=tf.ToTensor())
        cnp = ConvCNP2d(channel=1)
    elif args.dataset == 'cifar10':
        trainset = CIFAR10('~/data/cifar10', train=True, transform=tf.ToTensor())
        testset = CIFAR10('~/data/cifar10', train=False, transform=tf.ToTensor())
        cnp = ConvCNP2d(channel=3)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=16, shuffle=True)

    cnp = cnp.to(args.device)

    optimizer = optim.Adam(cnp.parameters(), lr=args.learning_rate)

    args.mb = master_bar(range(1, args.epochs + 1))

    for epoch in args.mb:
        avg_train_loss = train(cnp, trainloader, optimizer)
        valid_ll, rmse, image = validate(cnp, testloader)

        args.writer.add_scalar('train/likelihood', avg_train_loss, epoch)
        args.writer.add_scalar('validate/likelihood', valid_ll, epoch)
        args.writer.add_scalar('validate/rmse', rmse, epoch)
        args.writer.add_image('validate/image', image, epoch)
    torch.save(cnp.state_dict(), filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-B', type=int, default=16)
    parser.add_argument('--learning-rate', '-LR', type=float, default=5e-4)
    parser.add_argument('--epochs', '-E', type=int, default=100)
    parser.add_argument('--dataset', '-D', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--logging', default=False, action='store_true')

    args = parser.parse_args()

    filename = 'convcnp2d_{}.pth.gz'.format(args.dataset)

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    args.writer = SummaryWriter()
    main()
    args.writer.close()