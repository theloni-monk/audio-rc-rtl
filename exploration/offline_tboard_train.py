import datetime
import os
from pathlib import Path
import math

import signal


import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--train2epoch", "-e", type=int)
parser.add_argument("--ckptfolder", "-c", default="ckpt")

parser.add_argument("--saveevery", default=250, type=int)
parser.add_argument("--testevery", "-t", default=50, type=int)
parser.add_argument("--valevery", "-v", default=50, type=int)

parser.add_argument("--nens", "-n", default=400)
parser.add_argument("--batch", "-b", default=64)

parser.add_argument("--enslr",  default=0.005, type=float)
parser.add_argument("--enssched", choices = ['const', 'exp', 'cos'], default='cos')
parser.add_argument("--ensschedtau", default=200, type=float)
parser.add_argument("--ensschedmin", default=1e-8, type=float)

parser.add_argument("--gradlr",  default=0.0025, type=float)
parser.add_argument("--gradsched", choices = ['const', 'exp', 'cos'], default='cos')
parser.add_argument("--gradschedpshift", default = 0.01-5*math.pi/6)
parser.add_argument("--gradschedtau", default=150, type=float)
parser.add_argument("--gradschedmin", default=1e-8, type=float)

parser.add_argument("--gradwrperiod", default=150, type=float)
parser.add_argument("--enswrperiod", default=-1, type=float)

parser.add_argument("--wrdecay", default=0.995, type=float)

parser.add_argument("--wstd", default = 1.0, type=float)
parser.add_argument("--prec", default = 0.1, type=float)

parser.add_argument("--phase_aug", "-a", default=False, action='store_true')

args = parser.parse_args()

import torch
from torch import utils
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from data.torchfsdd.dataset import TorchFSDDGenerator
from data.augmentations.randphase import RandomAllPass

from optim.ens_optim import HybridEnsGradOptimizer
from optim.schedules.sched import *
from optim.schedules.noise_sched import UncertAnnealingNS

from models.rc import FSDDPipelineV3
from models.pqmf import PQMF

# Initialize a generator for a local version of FSDD
fsdd = TorchFSDDGenerator(version='local', 
                        path='/home/theloni/audio-rc-rtl/exploration/data/torchfsdd/recordings',
                        transforms=RandomAllPass if args.phase_aug else None)

# Create two Torch datasets for a train-test split from the generator
train_set, val_set, test_set = fsdd.train_val_test_split(test_size=0.05, val_size=0.2)

SNR = 1e-4
PADLEN = 10112
def collate_fn(batch):
    """Collects together sequences into a single batch, arranged in descending length order."""
    batch_size = len(batch)

    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int))

    # Create list of sequences, and tensors for lengths and labels
    sequences, lengths, labels = [], torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)
    for i, (sequence, label) in enumerate(batch):
        lengths[i], labels[i] = len(sequence), label
        sequences.append(sequence)
        # print(sequence.shape)
    avg_rms = sum(seq.pow(2).mean().sqrt() for seq in sequences) / len(sequences)
    

    # Combine sequences into a padded matrix
    padded_sequences = torch.stack([torch.nn.functional.pad(seq,(0, PADLEN - seq.shape[-1]), "constant", 0.0)+SNR*torch.randn(1, PADLEN)/avg_rms for seq in sequences])#torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Shape: (B x T_max x D)

    return padded_sequences, lengths, labels
    # Shapes: (B x T_max x D), (B,), (B,)

def save_model(m, epochs_reached):
    outdir = Path(args.ckptfolder) / Path(f"epoch{epochs_reached}")
    os.makedirs(outdir, exist_ok=True)
    for name, ckpt in m.named_parameters():
        torch.save(ckpt, outdir /Path(f"{name}.tensor"))
    

def save_exit(opt, epochs_reached, writer, pqmf):
    m = opt.forward_model
    test_X, _, test_y = next(iter(test_gen))
    test_X, test_y = pqmf(test_X.cuda()), test_y.cuda()
    tce = torch.nn.functional.cross_entropy(m(test_X), test_y)
    val_X, _, val_y = next(iter(val_gen))
    val_X, val_y = pqmf(val_X.cuda()), val_y.cuda()
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
    for (pname, p) in m.named_parameters():
        assert pname in fpnames, f"Unable to locate parameter {pname} in given checkpoint dir"
        p.data = torch.load(ckpt_path / fnames[fpnames.index(pname)])
    return epoch

def testval_log_save(i, mean_train_l, uncert, saturation, prec, writer, start_epoch, model, pqmf):
    writer.add_scalar("train/mean_crossentropy", mean_train_l, global_step=i)
    writer.add_scalar("train/params/mean_uncert", uncert[1], global_step=i)
    writer.add_scalar("train/params/ens_saturation", saturation, global_step=i)
    writer.add_scalar("train/params/precision", prec, global_step=i)

    if (i + 1) % args.saveevery == 0:
        save_model(model, i)
    
    if (i + 1) % args.testevery == 0 or i == start_epoch:
        test_X, _, test_y = next(iter(test_gen))
        test_X, test_y = pqmf(test_X.cuda()), test_y.cuda()
        tce = torch.nn.functional.cross_entropy(uut(test_X), test_y)
        writer.add_scalar("test/crossentropy", tce, global_step=i)

    if (i+1) % args.valevery == 0 or i == start_epoch:
        val_X, _, val_y = next(iter(val_gen))
        val_X, val_y = pqmf(val_X.cuda()), val_y.cuda()
        yhat = uut(val_X) 
        acc = (yhat.argmax(-1) == val_y).sum() / val_y.numel()
        writer.add_scalar("val/accuracy", acc, global_step=i)

epochs_ran = 0
def train(start_epoch, opt, esched, gsched, writer, pqmf):
    for i in range(start_epoch, args.train2epoch):
        train_l, u, prec = [], None, None
        for j, (X, _, y) in enumerate(train_gen):
                X, y = pqmf(X.cuda()), y.cuda()

                if X.shape[0] != args.batch:
                    continue
                Y = F.one_hot(y, 10)

                loss_closure = lambda: F.cross_entropy(opt.forward_model(X), y)
                
                l, u, prec = opt.ens_step(X, Y.float(), loss_closure)
                train_l.append(l)
        
        global epochs_ran 
        epochs_ran = epochs_ran + 1
        testval_log_save(i, sum(train_l) / len(train_l), u, opt.n_saturated_params, prec, writer, start_epoch, opt.forward_model, pqmf)

        opt.ens_lr = esched.step()
        opt.grad_lr = gsched.step()

def linebreak():
    print("\n")
    print("".join('-' for _ in range(10)))
    print("\n")

if __name__ == "__main__":
    linebreak()
    train_gen = utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=args.batch, shuffle=True, pin_memory=True)
    val_gen = utils.data.DataLoader(val_set, collate_fn=collate_fn, batch_size=len(val_set))
    test_gen = utils.data.DataLoader(test_set, collate_fn=collate_fn, batch_size=len(test_set))

    model_params = {
        "n_bands": 32,
        "n_feats": 64,
        "n_bits": 4,
        "spec_rad": -1,
        # "mix_degree": -1,
        "rc_multiplex_degree": 4,
        "feedback_nl": True
    }
    uut = FSDDPipelineV3(**model_params).cuda()
    print(uut.rc.feedback_nl)
    pqmf = PQMF(100, model_params["n_bands"])

    with torch.no_grad():
        start_epoch = load_model(uut, args.ckptfolder) if args.ckptfolder != "ckpt" else 0
    assert start_epoch < args.train2epoch, "target epoch is less than starting epoch" 

    print(f"loaded model with {sum(p.numel() for p in uut.parameters())} params => spinning up optimizer")
    
    linebreak()
    
    ensopt = HybridEnsGradOptimizer(uut, args.nens, args.batch, 10, 
                             gpus=[f"cuda:{i}" for i in range(torch.cuda.device_count())], # kenneth is pinning gpu 2
                             ens_lr = args.enslr,
                             grad_lr = args.gradlr,
                             init_weight_std = args.wstd,
                             noise_sched = UncertAnnealingNS(args.prec, 0.01, 0.995),
                             criterion = F.cross_entropy
    )
    ensopt.modelget_timeout = 30     

    ens_sched = None
    if args.enssched == 'const':
        ens_sched = ConstSched(args.enslr)
    elif args.enssched == 'exp':
        ens_sched = ExpAnnealingSched(args.enslr, args.ensschedtau, min_val=args.ensschedmin)
    elif args.enssched == 'cos':
        ens_sched = CosAnnealingSched(args.enslr, args.ensschedtau, cycles=0.5, min_val=args.ensschedmin)
    else:
        assert False
    if args.enswrperiod > 0:
        ens_sched = WarmRestarter(ens_sched, args.enswrperiod, args.wrdecay)
    
    grad_sched = None
    if args.gradsched == 'const':
        grad_sched = ConstSched(args.gradlr)
    elif args.enssched == 'exp':
        grad_sched = ExpAnnealingSched(args.gradlr, args.gradschedtau, min_val=args.gradschedmin)
    elif args.gradsched == 'cos':
        grad_sched = CosAnnealingSched(args.gradlr, args.gradschedtau, args.gradschedpshift, min_val=args.gradschedmin)
    else:
        assert False
    if args.gradwrperiod > 0:
        grad_sched = WarmRestarter(args.gradwrperiod, args.wrdecay, grad_sched)
    
    
    dt = datetime.datetime.now()
    writer = SummaryWriter(f"logs/pll_mlp_rc/{dt.month}_{dt.day}_{dt.hour}_{dt.minute}")
    
    with open(Path(writer.log_dir) / "model_params.json", "w") as f:
        f.write(str(model_params))

    print("optimizer instantiated => starting training")

    linebreak()

    try:
        train(start_epoch, ensopt, ens_sched, grad_sched, writer, pqmf)
    except (InterruptedError, KeyboardInterrupt) as e:
        print("interrupted: saving")
        save_exit(ensopt, start_epoch+epochs_ran, writer, pqmf)
        linebreak()
        raise e
        

    print("training complete => saving model")

    save_exit(ensopt, args.train2epoch, writer, pqmf)