import datetime
import os
from pathlib import Path

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--train2epoch", "-e", type=int)
parser.add_argument("--ckptfolder", "-c", default="ckpt")

parser.add_argument("--bands", default=32, type=int)

parser.add_argument("--saveevery", default=250, type=int)
parser.add_argument("--testevery", "-t", default=50, type=int)
parser.add_argument("--valevery", "-v", default=50, type=int)

parser.add_argument("--nens", "-n", default=400)
parser.add_argument("--batch", "-b", default=64)

parser.add_argument("--enslr",  default=0.01, type=float)
parser.add_argument("--enssched", choices = ['const', 'exp', 'cos'], default='cos')
parser.add_argument("--ensschedtau", default=400, type=float)
parser.add_argument("--ensschedmin", default=1e-7, type=float)

parser.add_argument("--gradlr",  default=0.025, type=float)
parser.add_argument("--gradsched", choices = ['const', 'exp', 'cos'], default='cos')
parser.add_argument("--gradschedpshift", default = 0)
parser.add_argument("--gradschedtau", default=400, type=float)
parser.add_argument("--gradschedmin", default=1e-6, type=float)

parser.add_argument("--gradwrperiod", default=300, type=float)
parser.add_argument("--enswrperiod", default=300, type=float)

parser.add_argument("--wrdecay", default=0.5, type=float)

parser.add_argument("--wstd", default = 2, type=float)
parser.add_argument("--prec", default = 0.25, type=float)

parser.add_argument("--aug", "-a", default=False, action='store_true')

args = parser.parse_args()

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from data.torchfsdd.dataset import TorchFSDDGenerator
from data.augmentations.randphase import AllPassFilter
from torch_audiomentations import *

from optim.ens_optim import HybridEnsGradOptimizer
from optim.schedules.sched import *
from optim.schedules.noise_sched import UncertAnnealingNS

from models.rc import FSDDPipelineV3
from models.pqmf import PQMF

# Initialize augmentation callable
audio_augmentation = Compose(
    p=0.5,
    transforms=[
        Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.1,
            sample_rate = 8000,
            output_type='dict'
        ),
        PitchShift(p=0.05, p_mode='per_example',
            sample_rate = 8000,
            output_type='dict'),
        AddColoredNoise(p=1,
            min_snr_in_db = 3,
            sample_rate = 8000),
        Shift(-0.05, 0.2, p=0.1,
            sample_rate = 8000,
            output_type='dict'),
        PolarityInversion(p=0.1,
            sample_rate = 8000,
            output_type='dict'),
        AllPassFilter(sample_rate=8000,
            output_type='dict')
    ], output_type='dict'
)
pqmf = PQMF(100, args.bands)
transform = lambda x: pqmf(audio_augmentation(x)['samples'])

# Initialize a generator for a local version of FSDD
fsdd = TorchFSDDGenerator(version='local', 
                        path='/home/theloni/audio-rc-rtl/exploration/data/torchfsdd/recordings',
                        transforms=transform,
                        dont_transform_val=False)

train_set, test_set, val_set  = fsdd.train_val_test_split(test_size=0.05, val_size=0.1)

#TODO: add extra labels for noise and silence, and add randomly in the dataloader
def collate_fn(batch):
    """Collects together sequences into a single batch, arranged in descending length order."""
    batch_size = len(batch)

    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int))

    # Create list of sequences, and tensors for lengths and labels
    sequences,  labels = [], torch.zeros(batch_size, dtype=torch.long)
    for i, (sequence, label) in enumerate(batch):
        labels[i] = label
        sequences.append(sequence)

    # Combine sequences into a padded matrix
    stacked_sequences = torch.cat(sequences, dim=0)#torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Shape: (B x T_max x D)

    return stacked_sequences, labels
    # Shapes: (B x T_max x D), (B,), (B,)


def save_model(m, epochs_reached):
    outdir = Path(args.ckptfolder) / Path(f"epoch{epochs_reached}")
    os.makedirs(outdir, exist_ok=True)
    for name, ckpt in m.named_parameters():
        torch.save(ckpt, outdir /Path(f"{name}.tensor"))


def save_exit(opt, epochs_reached, writer):
    m = opt.forward_model
    test_X,  test_y = next(iter(test_gen))
    test_X, test_y = test_X.cuda(), test_y.cuda()
    tce = torch.nn.functional.cross_entropy(m(test_X), test_y)
    val_X,  val_y = next(iter(val_gen))
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
    for (pname, p) in m.named_parameters():
        assert pname in fpnames, f"Unable to locate parameter {pname} in given checkpoint dir"
        p.data = torch.load(ckpt_path / fnames[fpnames.index(pname)])
    return epoch

def testval_log_save(i, mean_train_l, uncert, saturation, prec, writer, start_epoch, model):
    writer.add_scalar("train/mean_crossentropy", mean_train_l, global_step=i)
    writer.add_scalar("train/params/mean_uncert", uncert[1], global_step=i)
    writer.add_scalar("train/params/ens_saturation", saturation, global_step=i)
    writer.add_scalar("train/params/precision", prec, global_step=i)

    if (i + 1) % args.saveevery == 0:
        save_model(model, i)
    
    if (i + 1) % args.testevery == 0 or i == start_epoch:
        test_X,  test_y = next(iter(test_gen))
        test_X, test_y = test_X.cuda(), test_y.cuda()
        tce = torch.nn.functional.cross_entropy(uut(test_X), test_y)
        writer.add_scalar("test/crossentropy", tce, global_step=i)

    if (i+1) % args.valevery == 0 or i == start_epoch:
        val_X,  val_y = next(iter(val_gen))
        val_X, val_y = val_X.cuda(), val_y.cuda()
        yhat = uut(val_X) 
        acc = (yhat.argmax(-1) == val_y).sum() / val_y.numel()
        writer.add_scalar("val/accuracy", acc, global_step=i)

epochs_ran = 0
def train(start_epoch, opt, esched, gsched, writer):
    for i in range(start_epoch, args.train2epoch):
        train_l, u, prec = [], None, None
        for j, (X,  y) in enumerate(train_gen):
                X, y = X.cuda(), y.cuda()
                if X.shape[0] != args.batch:
                    continue
                Y = F.one_hot(y, 10)

                loss_closure = lambda: F.cross_entropy(opt.forward_model(X), y)
                
                l, u, prec = opt.ens_step(X, Y.float(), loss_closure)
                train_l.append(l)
        
        global epochs_ran 
        epochs_ran = epochs_ran + 1
        testval_log_save(i, sum(train_l) / len(train_l), u, opt.n_saturated_params, prec, writer, start_epoch, opt.forward_model)

        opt.ens_lr = esched.step()
        opt.grad_lr = gsched.step()

def linebreak():
    print("\n")
    print("".join('-' for _ in range(10)))
    print("\n")

if __name__ == "__main__":
    linebreak()
    train_gen = data_utils.DataLoader(train_set, collate_fn=collate_fn, batch_size=args.batch, shuffle=True, pin_memory=True)
    val_gen = data_utils.DataLoader(val_set, collate_fn=collate_fn, batch_size=len(val_set))
    test_gen = data_utils.DataLoader(test_set, collate_fn=collate_fn, batch_size=len(test_set))

    model_params = {
        "n_bands": 32,
        "n_feats": 256,
        "n_bits": 4,
        "spec_rad": -1,
        "rc_multiplex_degree": 4,
        "feedback_nl": True,
        "pl_version": 3
    }
    uut = FSDDPipelineV3(**model_params).cuda() # V3

    with torch.no_grad():
        start_epoch = load_model(uut, args.ckptfolder) if args.ckptfolder != "ckpt" else 0
    assert start_epoch < args.train2epoch, "target epoch is less than starting epoch" 

    print(f"loaded model with {sum(p.numel() for p in uut.parameters())} params => spinning up optimizer")
    
    linebreak()
    
    ensopt = HybridEnsGradOptimizer(uut, args.nens, args.batch, 10, 
                             gpus=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
                             ens_lr = args.enslr,
                             grad_lr = args.gradlr,
                             init_weight_std = args.wstd,
                             noise_sched = UncertAnnealingNS(args.prec, 0.01, 0.995),
                             obsv_operator = lambda x: x / x.abs().max(dim=1)[0].unsqueeze(1),
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
        ens_sched = WarmRestarter(args.enswrperiod, args.wrdecay, ens_sched)

    grad_sched = None
    if args.gradsched == 'const':
        grad_sched = ConstSched(args.gradlr)
    elif args.enssched == 'exp':
        grad_sched = ExpAnnealingSched(args.gradlr, args.gradschedtau, min_val=args.gradschedmin)
    elif args.gradsched == 'cos':
        grad_sched = CosAnnealingSched(args.gradlr, args.gradschedtau, args.gradschedpshift, cycles=0.625,  min_val=args.gradschedmin)
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
        train(start_epoch, ensopt, ens_sched, grad_sched, writer)
    except (InterruptedError, KeyboardInterrupt) as e:
        print("interrupted: saving")
        save_exit(ensopt, start_epoch+epochs_ran, writer)
        linebreak()
        raise e

    print("training complete => saving model")

    save_exit(ensopt, args.train2epoch, writer)