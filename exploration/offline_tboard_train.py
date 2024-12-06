import datetime
import os
import math
import random
from pathlib import Path
import warnings
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train2epoch", "-e", type=int)
parser.add_argument("--ckptfolder", "-c", default="ckpt")

parser.add_argument("--bands", default=8, type=int)

parser.add_argument("--resampevery", default=10, type=int)
parser.add_argument("--saveevery", default=50, type=int)
parser.add_argument("--testevery", "-t", default=1, type=int)
parser.add_argument("--valevery", "-v", default=5, type=int)

parser.add_argument("--nens", "-n", default=350)
parser.add_argument("--batch", "-b", default=256)

parser.add_argument("--enslr", default=1.0, type=float)
parser.add_argument("--enssched", choices=["const", "lin", "exp", "cos"], default="cos")
parser.add_argument("--ensschedtau", default=3000, type=float)
parser.add_argument("--ensschedmin", default=0.01, type=float)

parser.add_argument("--gradlr", default=0.1, type=float)
parser.add_argument(
    "--gradsched", choices=["const", "lin", "exp", "cos"], default="cos"
)
parser.add_argument("--gradschedpshift", default=-math.pi/4, type=float)
parser.add_argument("--gradschedcycles", default=0.625, type=float)
parser.add_argument("--gradschedtau", default=3000, type=float)
parser.add_argument("--gradschedmin", default=0.01, type=float)

parser.add_argument("--gradwrperiod", default=1500, type=float)
parser.add_argument("--enswrperiod", default=1500, type=float)

parser.add_argument("--gradwrdecay", default=0.25, type=float)
parser.add_argument("--enswrdecay", default=0.25, type=float)

parser.add_argument("--wstd", default=0.1, type=float)
parser.add_argument("--prec", default=0.05, type=float)

parser.add_argument("--aug", "-a", default=False, action="store_true")

args = parser.parse_args()

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import functional as F
from data.torchfsdd.dataset import TorchFSDDGenerator
from data.augmentations.randphase import AllPassFilter
from torch_audiomentations import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

pltr = lambda x: x.cpu().detach().numpy()

from optim.ens_optim import HybridEnsGradOptimizer
from optim.schedules.sched import *
from optim.schedules.noise_sched import UncertAnnealingNS

from models.rc import FSDDPipelineV8, N_CLASSES
from models.pqmf import PQMF

# this lib spits some annoying bs so I ignore it
with warnings.catch_warnings(action="ignore"):
    # Initialize augmentation callable
    audio_augmentation = Compose(
        transforms=[
            AllPassFilter(sample_rate=8000, p=0.5, output_type="dict"),
            PitchShift(
                max_transpose_semitones=2, p=0.1, sample_rate=8000, output_type="dict"
            ),
            AddColoredNoise(
                p=0.5,
                min_snr_in_db=2,
                max_snr_in_db=15,
                sample_rate=8000,
                output_type="dict",
            ),
            Shift(0.05, 0.33, p=0.1, sample_rate=8000, output_type="dict"),
        ],
        output_type="dict",
    )
pqmf = PQMF(100, args.bands)
transform = lambda x: pqmf(audio_augmentation(x)["samples"])

# Initialize a generator for a local version of FSDD
fsdd = TorchFSDDGenerator(
    version="local",
    path="/home/theloni/audio-rc-rtl/exploration/data/torchfsdd/recordings",
    train_transforms=transform,
    val_transforms=pqmf,
)

train_set, val_set, test_set = fsdd.train_val_test_split(
    test_size=0.05, val_size=0.2
)

NOISE_PROB = 1 / N_CLASSES
# SILENCE_PROB  = 0.0833
# SILENCE_EPS = 1e-4
b_rms = lambda x: x.pow(2).mean(dim=-1).mean(dim=-1).sqrt()


def collate_fn(batch):
    """Collects together sequences into a single batch, arranged in descending length order."""
    batch_size = len(batch)

    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int))

    # Create list of sequences, and tensors for lengths and labels
    sequences, labels = [], torch.zeros(batch_size, dtype=torch.long)
    for i, (sequence, label) in enumerate(batch):
        labels[i] = label
        seq = sequence
        dice = random.random()
        if dice < NOISE_PROB:
            labels[i] = 10
            seq = pqmf(
                b_rms(seq).unsqueeze(-1).unsqueeze(-1)
                * torch.randn(seq.shape[0], 1, pqmf.n_band * seq.shape[-1])
            )
        # if dice > NOISE_PROB and dice < NOISE_PROB + SILENCE_PROB:
        #     labels[i] = 11
        #     seq = pqmf(SILENCE_EPS * torch.randn(seq.shape[0], 1, pqmf.n_band * seq.shape[-1]))
        sequences.append(seq)

    # Combine sequences into a padded matrix
    stacked_sequences = torch.cat(
        sequences, dim=0
    )  # torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Shape: (B x T_max x D)

    return stacked_sequences, labels


def save_model(m, epochs_reached):
    outdir = Path(args.ckptfolder) / Path(f"epoch{epochs_reached}")
    os.makedirs(outdir, exist_ok=True)
    for name, ckpt in m.named_parameters():
        torch.save(ckpt, outdir / Path(f"{name}.tensor"))


def save_exit(opt, epochs_reached, writer):
    m = opt.forward_model
    test_X, test_y = next(iter(test_gen))
    test_X, test_y = test_X.cuda(), test_y.cuda()
    tce = torch.nn.functional.cross_entropy(m(test_X), test_y)
    val_X, val_y = next(iter(val_gen))
    val_X, val_y = val_X.cuda(), val_y.cuda()
    yhat = m(val_X)
    acc = (yhat.argmax(-1) == val_y).sum() / val_y.numel()
    writer.add_hparams(vars(args), {"test/crossentropy": tce, "val/accuracy": acc})

    save_model(m, epochs_reached)
    linebreak()
    print("save completed - spinning down")
    opt.join()
    del opt
    linebreak()
    print("neatly exited")

    exit(0)


def load_model(m, folder):
    ckpt_path = Path(folder)
    fnames = [f.name for f in ckpt_path.iterdir() if f.is_file()]
    fpnames = [n.split("_epoch")[0] for n in fnames]
    epoch = int(fnames[0].split("_epoch")[-1].split(".")[0])
    for pname, p in m.named_parameters():
        assert (
            pname in fpnames
        ), f"Unable to locate parameter {pname} in given checkpoint dir"
        p.data = torch.load(ckpt_path / fnames[fpnames.index(pname)])
    return epoch


def log_dist(y, yhat, writer, step):
    f, ax = plt.subplots()

    ax.set_title("Digit Model Distribution histogram (10 := Noise)")
    ax.hist(pltr(y), bins=np.arange(0, N_CLASSES + 1, 1), label="digit counts")
    ax.hist(
        pltr(yhat), bins=np.arange(0, N_CLASSES + 1, 1), label="predicted digit counts"
    )
    ax.set_xticks(list(map(float, range(N_CLASSES))))
    ax.legend()
    writer.add_figure("val/dist", f, global_step=step)


def log_confusion(y, yhat, writer, step):
    pred = pltr(yhat)
    # Compute confusion matrix
    cm = confusion_matrix(pltr(y), pred)

    # Plot confusion matrix
    f, ax = plt.subplots()
    ax.set_title("Digit Model Confusion (10 := Noise)")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=np.unique(pltr(y)),
        yticklabels=np.unique(pltr(y)),
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    writer.add_figure("val/confusion", f, global_step=step)


RUNNING_SIZE = 2


def testval_log_save(
    ij, mean_train_l, uncert, saturation, prec, n_free, writer, start_epoch, model, lrs
):
    grad_lr, ens_lr = lrs
    i, j = ij
    step = i * (len(train_set) // args.batch) + j
    if (step + 1) % RUNNING_SIZE == 0:
        writer.add_scalar("train/mean_crossentropy", mean_train_l, global_step=step)
        writer.add_scalar("train/params/mean_uncert", uncert[1], global_step=step)
        writer.add_scalar("train/params/ens_saturation", saturation, global_step=step)
        writer.add_scalar("train/params/precision", prec, global_step=step)
        writer.add_scalar("train/n_free_params", n_free, global_step=step)
        writer.add_scalar("train/hypers/grad_lr", grad_lr, global_step=step)
        writer.add_scalar("train/hypers/ens_lr", ens_lr, global_step=step)

    if j == 0:
        model.eval()
        if (i + 1) % args.saveevery == 0:
            save_model(model, i)

        if (i + 1) % args.testevery == 0 or i == start_epoch:
            test_X, test_y = next(iter(test_gen))
            test_X, test_y = test_X.cuda(), test_y.cuda()
            tce = torch.nn.functional.cross_entropy(uut(test_X), test_y)
            writer.add_scalar("test/crossentropy", tce, global_step=i)

        if (i + 1) % args.valevery == 0 or i == start_epoch:
            val_X, val_y = next(iter(val_gen))
            val_X, val_y = val_X.cuda(), val_y.cuda()
            yhat = uut(val_X)
            acc = (yhat.argmax(-1) == val_y).sum() / val_y.numel()
            writer.add_scalar("val/accuracy", acc, global_step=i)
            log_dist(val_y.flatten(), yhat.argmax(-1).flatten(), writer, i)
            log_confusion(val_y.flatten(), yhat.argmax(-1).flatten(), writer, i)


epochs_ran = 0


def train(start_epoch, opt, esched, gsched, writer):
    opt.forward_model.train()
    train_l, u, prec = [], None, None

    for i in range(start_epoch, args.train2epoch):
        for j, (X, y) in enumerate(train_gen):
            X, y = X.cuda(), y.cuda()
            Y = F.one_hot(y, N_CLASSES)

            loss_closure = lambda: F.cross_entropy(opt.forward_model(X), y)

            l, u, prec = opt.ens_step(X, Y.float(), loss_closure)
            if len(train_l) > RUNNING_SIZE:
                train_l.pop(0)
            train_l.append(l)

            testval_log_save(
                (i, j),
                sum(train_l) / len(train_l),
                u,
                opt.n_saturated_params,
                prec,
                opt.n_active_weights,
                writer,
                start_epoch,
                opt.forward_model,
                (opt.grad_lr, opt.ens_lr),
            )

            opt.ens_lr = esched.step()
            opt.grad_lr = gsched.step()

        opt.noise_sched.step(u[1])

        global epochs_ran
        epochs_ran = epochs_ran + 1


EPS = 1e-7


def linebreak():
    print("\n")
    print("".join("-" for _ in range(10)))
    print("\n")


# goal here is to prevent posterior collapse(i.e. just output one label for everything)
H_REG = 0.1


def entropy(x):
    x = F.softmax(x, dim=-1).mean(dim=0)
    b = F.softmax(x, dim=0) * F.log_softmax(x, dim=0)
    b = b.sum()  # mul by -1 to minimize entropy
    return b


def grad_loss(yhat, y):
    H_yhaty = F.cross_entropy(yhat, y.argmax(-1), label_smoothing=0.1)
    # H_batchyhat = entropy(yhat)
    return H_yhaty  # - H_REG * H_batchyhat


if __name__ == "__main__":
    linebreak()
    train_gen = data_utils.DataLoader(
        train_set,
        collate_fn=collate_fn,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_gen = data_utils.DataLoader(
        val_set, collate_fn=collate_fn, batch_size=len(val_set)
    )
    test_gen = data_utils.DataLoader(
        test_set, collate_fn=collate_fn, batch_size=len(test_set)
    )

    model_params = {
        "n_bands": args.bands,
        "n_bits": 4,
        "spec_rad": -1,
        "rc_multiplex_degree": 2,
        "pl_version": 8.3,
    }
    # TODO: replace fsdd with wake word detection
    uut = FSDDPipelineV8(**model_params).cuda()  # V5

    with torch.no_grad():
        start_epoch = (
            load_model(uut, args.ckptfolder) if args.ckptfolder != "ckpt" else 0
        )
    assert start_epoch < args.train2epoch, "target epoch is less than starting epoch"

    print(
        f"loaded model with {sum(p.numel() for p in uut.parameters())} params => spinning up optimizer"
    )
    linebreak()

    ens_sched = None
    if args.enssched == "const":
        ens_sched = ConstSched(args.enslr)
    elif args.enssched == "exp":
        ens_sched = ExpAnnealingSched(
            args.enslr, args.ensschedtau, min_val=args.ensschedmin
        )
    elif args.enssched == "cos":
        ens_sched = CosAnnealingSched(
            args.enslr,
            args.ensschedtau,
            args.gradschedpshift,
            cycles=args.gradschedcycles,
            min_val=args.ensschedmin,
        )
    elif args.enssched == "lin":
        ens_sched = LinSched(args.enslr, args.ensschedmin, args.ensschedtau)
    else:
        assert False
    if args.enswrperiod > 0:
        ens_sched = WarmRestarter(args.enswrperiod, args.enswrdecay, ens_sched)

    grad_sched = None
    if args.gradsched == "const":
        grad_sched = ConstSched(args.gradlr)
    elif args.gradsched == "exp":
        grad_sched = ExpAnnealingSched(
            args.gradlr, args.gradschedtau, min_val=args.gradschedmin
        )
    elif args.gradsched == "cos":
        grad_sched = CosAnnealingSched(
            args.gradlr,
            args.gradschedtau,
            args.gradschedpshift,
            cycles=args.gradschedcycles,
            min_val=args.gradschedmin,
        )
    elif args.gradsched == "lin":
        grad_sched = LinSched(args.gradlr, args.gradschedmin, args.gradschedtau)
    else:
        assert False
    if args.gradwrperiod > 0:
        grad_sched = WarmRestarter(args.gradwrperiod, args.gradwrdecay, grad_sched)

    ensopt = HybridEnsGradOptimizer(
        uut,
        args.nens,
        args.batch,
        N_CLASSES,
        host_device="cuda:0",
        gpus=[f"cuda:{i}" for i in range(1, 8)],
        ens_lr=args.enslr,
        grad_lr=args.gradlr,
        init_weight_std=args.wstd,
        noise_sched=UncertAnnealingNS(args.prec, 1e-4, 0.9975),
        grad_criterion=grad_loss,
        collapse_policy= "fixing", #'resampling_with_inflation', #
        ens_weight_decay=None,
        grad_weight_decay=0.0001,
        resampevery=args.resampevery,
        mask_fixed=False,
    )
    ensopt.modelget_timeout = 30

    dt = datetime.datetime.now()
    writer = SummaryWriter(f"logs/pll_mlp_rc/{dt.month}_{dt.day}_{dt.hour}_{dt.minute}")

    with open(Path(writer.log_dir) / "model_params.json", "w") as f:
        f.write(str(model_params))

    print("optimizer instantiated => starting training")

    linebreak()

    try:
        train(start_epoch, ensopt, ens_sched, grad_sched, writer)
    except (InterruptedError, KeyboardInterrupt) as e:
        print("interrupted: saving")
        save_exit(ensopt, start_epoch + epochs_ran, writer)
        linebreak()
        raise e

    print("training complete => saving model")

    save_exit(ensopt, args.train2epoch, writer)
