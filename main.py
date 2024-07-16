import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy
import torch
import tqdm
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = '../Dataset/SOD'


def lr(args):
    return 1E-3 * args.batch_size / 64


def train(args):
    # Model
    model = nn.CTDNet()
    model = util.load_weight(model, ckpt='./weights/model.pt')

    # Fuse bn into conv
    model.eval()
    for m in model.modules():
        if type(m) is nn.Conv:
            modules_to_fuse = ["conv", "norm"]
            if type(m.relu) is torch.nn.ReLU:
                modules_to_fuse.append("relu")
            torch.ao.quantization.fuse_modules_qat(m, modules_to_fuse, inplace=True)

    model.train()

    model = nn.QAT(model)
    model.qconfig = torch.ao.quantization.get_default_qconfig()
    torch.ao.quantization.prepare_qat(model, inplace=True)

    model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(util.set_params(model, lr(args)), lr(args),
                                momentum=0.9, weight_decay=0, nesterov=True)

    filenames = []
    with open(f'{data_dir}/DUTS-TR.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split()[0]
            filenames.append(f'{data_dir}/' + filename)

    sampler = None
    dataset = Dataset(filenames, args.input_size, train=True)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    best = 0
    criterion = util.ComputeLoss()
    scheduler = util.CosineLR(args, optimizer)

    with open('weights/step.csv', 'w') as log:
        logger = csv.DictWriter(log, fieldnames=['epoch', 'loss', 'MAE', 'F-beta'])
        logger.writeheader()
        for epoch in range(args.epochs):

            model.train()

            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
            p_bar = tqdm.tqdm(loader, total=len(loader))  # progress bar

            optimizer.zero_grad()
            avg_loss = util.AverageMeter()

            for samples, targets in p_bar:
                samples = samples.cuda()
                targets = targets.cuda()

                samples = samples.float()
                targets = targets.float()

                # Forward
                outputs = model(samples)
                loss = criterion(outputs, targets)

                avg_loss.update(loss.item(), samples.size(0))

                # Backward
                loss.backward()

                # Optimize
                optimizer.step()
                optimizer.zero_grad()

                # Log
                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, avg_loss.avg)
                p_bar.set_description(s)

            scheduler.step(epoch, optimizer)

            save = copy.deepcopy(model)
            save.eval()
            save.cpu()

            torch.ao.quantization.convert(save, inplace=True)

            last = test(args, save)

            logger.writerow({'epoch': str(epoch + 1).zfill(3),
                             'loss': str(f'{avg_loss.avg:.3f}'),
                             'MAE': str(f'{last[0]:.3f}'),
                             'F-beta': str(f'{last[1]:.3f}')})
            log.flush()

            # Update best F-beta
            if last[1] > best:
                best = last[1]

            # Save last, best and delete
            save = torch.jit.trace(save.cpu(), samples.cpu())
            save.save('./weights/last.qat')
            if best == last[1]:
                save.save('./weights/best.qat')
            del save

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    filenames = []
    with open(f'{data_dir}/DUTS-TE.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split()[0]
            filenames.append(f'{data_dir}/' + filename)

    dataset = Dataset(filenames, args.input_size, train=False)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

    if model is None:
        model = torch.jit.load('./weights/best.qat')

    model.eval()

    num = 50
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('', 'F-beta', 'MAE'))

    avg_mae = 0
    avg_recall = torch.zeros(num)
    avg_precision = torch.zeros(num)
    for samples, targets in p_bar:
        samples = samples.float()
        targets = targets.float()
        # Inference
        outputs = model(samples)
        outputs = outputs.sigmoid()
        outputs = outputs.squeeze(1)
        # Metrics
        precision, recall, mae = util.compute_metric(outputs, targets, num)
        avg_precision += precision
        avg_recall += recall
        avg_mae += mae
    # Compute metrics
    avg_precision = avg_precision / len(dataset)
    avg_recall = avg_recall / len(dataset)

    f_beta = (1 + 0.3) * avg_precision * avg_recall / (0.3 * avg_precision + avg_recall + numpy.finfo(float).eps)
    avg_mae = avg_mae / len(dataset)
    if isinstance(avg_mae, torch.Tensor):
        avg_mae = avg_mae.item()
    if isinstance(f_beta, torch.Tensor):
        f_beta = f_beta.max().item()
    # Print results
    print(('%10s' + '%10.3g' * 2) % ('', f_beta, avg_mae))
    model.float()  # for training
    torch.cuda.empty_cache()
    return avg_mae, f_beta


@torch.no_grad()
def demo(args):
    filenames = []
    with open(f'{data_dir}/DUTS-TE.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split()[0]
            filenames.append(f'{data_dir}/' + filename)

    model = torch.jit.load('./weights/best.qat')
    model.eval()
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    for filename in tqdm.tqdm(filenames):
        image = cv2.imread(filename)
        # resize and normalize the image
        x = cv2.resize(image, dsize=(args.input_size, args.input_size))
        x = x.astype('float32') / 255
        x -= mean
        x /= std

        cv2.cvtColor(x, cv2.COLOR_BGR2RGB, x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)

        # Inference
        output = model(x)
        output = output.sigmoid()
        output = torch.nn.functional.interpolate(output,
                                                 image.shape[:2],
                                                 mode='bilinear', align_corners=False)
        output = (output * 255).numpy()[0, 0].astype('uint8')

        cv2.imwrite('./results/' + os.path.basename(filename), output)


def profile(args):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.CTDNet()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    print(f'Number of parameters: {num_params}')
    print(f'Number of FLOPs: {flops}')
    if args.benchmark:
        # Latency
        model = nn.CTDNet().fuse()
        model.eval()

        x = torch.zeros(shape)
        for i in range(10):
            model(x)

        total = 0
        import time
        for i in range(1_000):
            start = time.perf_counter()
            with torch.no_grad():
                model(x)
            total += time.perf_counter() - start

        print(f"Latency: {total / 1_000 * 1_000:.3f} ms")


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    if not os.path.exists('weights'):
        os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
