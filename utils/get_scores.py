import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as sk
import copy
import os
import sys

from torch.autograd import Variable
from sklearn.linear_model import LogisticRegressionCV
from utils.mahalanobis_lib import get_Mahalanobis_score, tune_mahalanobis_hyperparams
from args import args
from utils.get_metrics import metrics
from utils.net_utils import save_checkpoint, get_threshold
from utils.logging import AverageMeter, ProgressMeter


class measures(object):
    def __init__(self, ckpt_base_dir):
        self.distance_best = 1.0
        self.energy_best = 1.0
        self.msp_best = 1.0
        self.odin_best = 1.0
        self.mahalanobis_best = 1.0
        self.ntom_best = 1.0
        self.ckpt_base_dir = ckpt_base_dir
        self.save = False

    def save_best(self, measure, labels, examples, id_mean, ood_mean, model, epoch, recall_level=0.95):
        auroc, aupr, fpr = metrics(labels, examples, recall_level=recall_level)
        print(f"ID {measure:<12}: {id_mean:>10.6f} \tOOD {measure:<12}: {ood_mean:>10.6f} \tauroc: {auroc*100:>4.2f} \taupr: {aupr*100:>4.2f} \tfpr95: {fpr*100:>4.2f}")
        if not self.save:
            return auroc, aupr, fpr
        best = getattr(self, measure + "_best")
        if fpr < best:
            setattr(self, measure + "_best", fpr)
            save_dir = self.ckpt_base_dir / (measure + ".pth")
            print(f"==> {measure}: New best, saving at {save_dir}")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "examples": examples,
                    "auroc": auroc,
                    "aupr": aupr,
                    "fpr": fpr,
                    "recall_level": recall_level
                },
                is_best=False,
                filename=save_dir,
                save=True,
            )
            
        return auroc, aupr, fpr

    def get_distance(self, dataloader, ood_dataloader, model, init_model, recall_level=0.95, epoch=-1):
        model.eval()
        init_model.eval()
        kl_loss = nn.KLDivLoss(reduction="none", log_target=True)
        logsoftmax = nn.LogSoftmax(dim=1)
        num_batches = len(dataloader)
        ood_num_batches = len(ood_dataloader)
        batches = min(num_batches, ood_num_batches)

        dist, ood_dist = 0., 0.
        labels = []
        examples = []

        if args.prune_type == "model_wise":
            get_threshold(model, args.prune_rate)

        with torch.no_grad():
            for X, _ in dataloader:
                X = X.cuda()
                pred = logsoftmax(model(X))
                init_pred = logsoftmax(init_model(X))
                distance = kl_loss(pred, init_pred).sum(dim=1).cpu().numpy()
                dist += distance.mean()
                examples += distance.tolist()
                labels += np.ones_like(distance).tolist()
            for ood_X, _ in ood_dataloader:
                ood_X = ood_X.cuda()
                ood_pred = logsoftmax(model(ood_X))
                ood_init_pred = logsoftmax(init_model(ood_X))
                ood_distance = kl_loss(ood_pred, ood_init_pred).sum(dim=1).cpu().numpy()
                examples += ood_distance.tolist()
                labels += np.zeros_like(ood_distance).tolist()
                ood_dist += ood_distance.mean()

        dist /= num_batches
        ood_dist /= num_batches

        labels = np.array(labels)
        examples = np.array(examples)

        auroc, aupr, fpr = self.save_best("distance", labels, examples, dist, ood_dist, model, epoch, recall_level=recall_level)

        return dist, ood_dist, auroc, aupr, fpr

    def get_msp(self, dataloader, ood_dataloader, model, recall_level=0.95, epoch=-1):
        model.eval()
        num_batches = len(dataloader)
        ood_num_batches = len(ood_dataloader)

        id_pred, ood_pred = 0., 0.
        labels = []
        examples = []

        if args.prune_type == "model_wise":
            get_threshold(model, args.prune_rate)

        with torch.no_grad():
            for X, _ in dataloader:
                X = X.cuda()
                pred = model(X)
                pred = np.max(F.softmax(pred, dim=1).detach().cpu().numpy(), axis=1)
                examples += pred.tolist()
                labels += np.ones_like(pred).tolist()
                id_pred += pred.mean()
            for X, _ in ood_dataloader:
                X = X.cuda()
                ood_predi = model(X)
                ood_predi = np.max(F.softmax(ood_predi, dim=1).detach().cpu().numpy(), axis=1)
                examples += ood_predi.tolist()
                labels += np.zeros_like(ood_predi).tolist()
                ood_pred += ood_predi.mean()
        id_pred /= num_batches
        ood_pred /= ood_num_batches

        labels = np.array(labels)
        examples = np.array(examples)
        auroc, aupr, fpr = self.save_best("msp", labels, examples, id_pred, ood_pred, model, epoch, recall_level=recall_level)
        return id_pred, ood_pred, auroc, aupr, fpr

    def get_energy(self, dataloader, ood_dataloader, model, pre_model=None, temperature=1.0, recall_level=0.95, epoch=-1):
        model.eval()
        num_batches = len(dataloader)
        ood_num_batches = len(ood_dataloader)

        energy, ood_energy = 0., 0.
        labels = []
        examples = []

        if args.prune_type == "model_wise":
            get_threshold(model, args.prune_rate)

        with torch.no_grad():
            for X, _ in dataloader:
                X = X.cuda()
                pred = model(X)
                e = temperature * torch.logsumexp(pred / temperature, 1).cpu().numpy()
                if pre_model is not None:
                    pre_pred = pre_model(X)
                    pre_e = (temperature * torch.logsumexp(pre_model(X) / temperature, 1)).cpu().numpy()
                    e -= pre_e
                labels += np.ones_like(e).tolist()
                examples += e.tolist()
                energy += e.mean()
            for X, _ in ood_dataloader:
                X = X.cuda()
                ood_pred = model(X)
                e = temperature * torch.logsumexp(ood_pred / temperature, 1).cpu().numpy()
                if pre_model is not None:
                    pre_pred = pre_model(X)
                    pre_e = (temperature * torch.logsumexp(pre_model(X) / temperature, 1)).cpu().numpy()
                    e -= pre_e
                labels += np.zeros_like(e).tolist()
                examples += e.tolist()
                ood_energy += e.mean()
        energy /= num_batches
        ood_energy /= ood_num_batches

        labels = np.array(labels)
        examples = np.array(examples)
        
        auroc, aupr, fpr = self.save_best("energy", labels, examples, energy, ood_energy, model, epoch, recall_level=recall_level)
        return energy, ood_energy, auroc, aupr, fpr

    def get_ntom(self, dataloader, ood_dataloader, model, recall_level=0.95, epoch=-1):
        model.eval()
        num_batches = len(dataloader)
        ood_num_batches = len(ood_dataloader)

        id_pred, ood_pred = 0., 0.
        labels = []
        examples = []

        with torch.no_grad():
            for X, _ in dataloader:
                X = X.cuda()
                pred = model(X)
                pred = np.max(F.softmax(pred, dim=1).detach().cpu().numpy(), axis=1)
                examples += pred.tolist()
                labels += np.ones_like(pred).tolist()
                id_pred += pred.mean()
            for X, _ in ood_dataloader:
                X = X.cuda()
                ood_predi = model(X)
                ood_predi = np.max(F.softmax(ood_predi, dim=1).detach().cpu().numpy(), axis=1)
                examples += ood_predi.tolist()
                labels += np.zeros_like(ood_predi).tolist()
                ood_pred += ood_predi.mean()
        id_pred /= num_batches
        ood_pred /= ood_num_batches

        labels = np.array(labels)
        examples = np.array(examples)
        auroc, aupr, fpr = self.save_best("ntom", labels, examples, id_pred, ood_pred, model, epoch, recall_level=recall_level)
        return id_pred, ood_pred, auroc, aupr, fpr

    def get_odin(self, dataloader, ood_dataloader, model, epsilon=0.0014, temper=1000.0, recall_level=0.95, epoch=-1):
        model.eval()
        num_batches = len(dataloader)
        ood_num_batches = len(ood_dataloader)
        criterion = torch.nn.CrossEntropyLoss().cuda()

        id_odin, ood_odin = 0., 0.
        labels = []
        examples = []
        temper = Variable(torch.tensor(temper).cuda(), requires_grad=True)

        if args.prune_type == "model_wise":
            get_threshold(model, args.prune_rate)

        for X, _ in dataloader:
            X = Variable(X.cuda(), requires_grad=True)
            # X = X.cuda()
            outputs = model(X)
            maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
            outputs /= temper
            
            tmp_labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
            loss = criterion(outputs, tmp_labels)
            loss.requires_grad_(True)
            loss.backward()
            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(X.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            tempInputs = torch.add(X.data, gradient, alpha=-epsilon)
            outputs = model(Variable(tempInputs))
            outputs = outputs / temper
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu().numpy()
            nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

            odin = np.max(nnOutputs, axis=1)
            examples += odin.tolist()
            labels += np.ones_like(odin).tolist()
            id_odin += odin.mean()

        for X, _ in ood_dataloader:
            X = Variable(X.cuda(), requires_grad=True)
            outputs = model(X)

            maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
            outputs = outputs / temper

            tmp_labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
            loss = criterion(outputs, tmp_labels)
            loss.requires_grad_(True)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(X.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            tempInputs = torch.add(X.data, gradient, alpha=-epsilon)
            outputs = model(Variable(tempInputs))
            outputs = outputs / temper
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu().numpy()
            nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

            odin = np.max(nnOutputs, axis=1)
            examples += odin.tolist()
            labels += np.zeros_like(odin).tolist()
            ood_odin += odin.mean()
        id_odin /= num_batches
        ood_odin /= ood_num_batches

        labels = np.array(labels)
        examples = np.array(examples)
        auroc, aupr, fpr = self.save_best("odin", labels, examples, id_odin, ood_odin, model, epoch, recall_level=recall_level)
        return id_odin, ood_odin, auroc, aupr, fpr

    def get_mahalanobis(self, dataloader, ood_dataloader, model, trainloader, recall_level=0.95, epoch=-1):
        model.eval()
        num_batches = len(dataloader)
        ood_num_batches = len(ood_dataloader)
        sample_mean, precision, best_regressor, best_magnitude, num_output = tune_mahalanobis_hyperparams(model, args.num_classes, trainloader, dataloader)
        num_classes = args.num_classes

        id_pred, ood_pred = 0., 0.
        labels = []
        examples = []

        if args.prune_type == "model_wise":
            get_threshold(model, args.prune_rate)

        for X, _ in dataloader:
            X = X.cuda()
            Mahalanobis_scores = get_Mahalanobis_score(X, model, num_classes, sample_mean, precision, num_output, best_magnitude)
            scores = -best_regressor.predict_proba(Mahalanobis_scores)[:, 1].flatten()
            examples += scores.tolist()
            labels += np.ones_like(scores).tolist()
            id_pred += scores.mean()
        for X, _ in ood_dataloader:
            X = X.cuda()
            Mahalanobis_scores = get_Mahalanobis_score(X, model, num_classes, sample_mean, precision, num_output, best_magnitude)
            scores = -best_regressor.predict_proba(Mahalanobis_scores)[:, 1].flatten()
            examples += scores.tolist()
            labels += np.zeros_like(scores).tolist()
            ood_pred += scores.mean()
        id_pred /= num_batches
        ood_pred /= ood_num_batches

        labels = np.array(labels)
        examples = np.array(examples)
        auroc, aupr, fpr = self.save_best("mahalanobis", labels, examples, id_pred, ood_pred, model, epoch, recall_level=recall_level)
        return id_pred, ood_pred, auroc, aupr, fpr

# https://github.com/wetliu/energy_ood/blob/77f3c09b788bb5a7bfde6fd3671228320ea0949c/utils/display_results.py
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


class ood_measure(object):
    def __init__(self, id_loader, ood_loaders, msp=False, energy=False, odin=False, mahalanobis=False):
        self.id_loader = id_loader
        self.ood_loaders = ood_loaders
        self.msp = msp
        self.energy = energy
        self.odin = odin
        self.mahalanobis = mahalanobis

    def ood_metrics(self, model, epoch, trainloader=None):
        ood_num = len(args.ood_set)
        if self.energy or self.msp:
            id_energy, id_msp = self.get_scores(self.id_loader, model)
            energy_dict = {}
            msp_dict = {}
            for i, ood_loader in enumerate(self.ood_loaders):
                ood_energy, ood_msp = self.get_scores(ood_loader, model)
                energy_dict[args.ood_set[i]] = ood_energy
                msp_dict[args.ood_set[i]] = ood_msp
            if self.energy:
                ood_avg, auroc_avg, aupr_avg, fpr_avg = 0., 0., 0., 0.
                labels, scores = torch.tensor([]).cuda(), torch.tensor([]).cuda()
                for k, v in energy_dict.items():
                    id_mean, ood_mean, auroc, aupr, fpr, labels, scores= self.compute_metrics(id_energy, v)
                    print(f"energy\t{args.set:>9}\t{id_mean:<10.6f}\t{k:>12}\t{ood_mean:<10.6f}\tauroc\t{auroc:<4.2f}\taupr\t{aupr:<4.2f}\tfpr95\t{fpr:<4.2f}")
                    ood_avg += ood_mean
                    auroc_avg += auroc
                    aupr_avg += aupr
                    fpr_avg += fpr
                ood_avg /= ood_num
                auroc_avg /= ood_num
                aupr_avg /= ood_num
                fpr_avg /= ood_num
                print(f"energy\tID\t{id_mean:<10.6f}\tOOD\t{ood_avg:<10.6f}\tauroc\t{auroc_avg:<4.2f}\taupr\t{aupr_avg:<4.2f}\tfpr95\t{fpr_avg:<4.2f}")
                if epoch % args.save_every == 0:
                    self.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'labels': labels,
                        'scores': scores
                    }, epoch, "energy")
            if self.msp:
                ood_avg, auroc_avg, aupr_avg, fpr_avg = 0., 0., 0., 0.
                for k, v in msp_dict.items():
                    id_mean, ood_mean, auroc, aupr, fpr, labels, scores= self.compute_metrics(id_msp, v)
                    print(f"msp\t{args.set:>9}\t{id_mean:<10.6f}\t{k:>12}\t{ood_mean:<10.6f}\tauroc\t{auroc:<4.2f}\taupr\t{aupr:<4.2f}\tfpr95\t{fpr:<4.2f}")
                    ood_avg += ood_mean
                    auroc_avg += auroc
                    aupr_avg += aupr
                    fpr_avg += fpr
                ood_avg /= ood_num
                auroc_avg /= ood_num
                aupr_avg /= ood_num
                fpr_avg /= ood_num
                print(f"msp\tID\t{id_mean:<10.6f}\tOOD\t{ood_avg:<10.6f}\tauroc\t{auroc_avg:<4.2f}\taupr\t{aupr_avg:<4.2f}\tfpr95\t{fpr_avg:<4.2f}")
                if epoch % args.save_every == 0:
                    self.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'labels': labels,
                        'scores': scores
                    }, epoch, "msp")
        if self.odin:
            id_odin = self.get_odin(self.id_loader, model)
            ood_avg, auroc_avg, aupr_avg, fpr_avg = 0., 0., 0., 0.
            for i, ood_loader in enumerate(self.ood_loaders):
                ood_odin = self.get_odin(ood_loader, model)
                id_mean, ood_mean, auroc, aupr, fpr, labels, scores= self.compute_metrics(id_odin, ood_odin)
                print(f"odin\t{args.set:>9}\t{id_mean:<10.6f}\t{args.ood_set[i]:>12}\t{ood_mean:<10.6f}\tauroc\t{auroc:<4.2f}\taupr\t{aupr:<4.2f}\tfpr95\t{fpr:<4.2f}")
                ood_avg += ood_mean
                auroc_avg += auroc
                aupr_avg += aupr
                fpr_avg += fpr
            ood_avg /= ood_num
            auroc_avg /= ood_num
            aupr_avg /= ood_num
            fpr_avg /= ood_num
            print(f"odin\tID\t{id_mean:<10.6f}\tOOD\t{ood_avg:<10.6f}\tauroc\t{auroc_avg:<4.2f}\taupr\t{aupr_avg:<4.2f}\tfpr95\t{fpr_avg:<4.2f}")
            if epoch % args.save_every == 0:
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'labels': labels,
                    'scores': scores
                }, epoch, "odin")
        if epoch == args.epochs and self.mahalanobis:
            sample_mean, precision, best_regressor, best_magnitude, num_output = tune_mahalanobis_hyperparams(model, args.num_classes, trainloader, self.id_loader)
            id_mahalanobis = self.get_mahalanobis(self.id_loader, model, sample_mean, precision, best_regressor, best_magnitude, num_output)
            ood_avg, auroc_avg, aupr_avg, fpr_avg = 0., 0., 0., 0.
            for i, ood_loader in enumerate(self.ood_loaders):
                ood_mahalanobis = self.get_mahalanobis(ood_loader, model, sample_mean, precision, best_regressor, best_magnitude, num_output)
                id_mean, ood_mean, auroc, aupr, fpr, labels, scores= self.compute_metrics(id_mahalanobis, ood_mahalanobis)
                print(f"mahalanobis\t{args.set:>9}\t{id_mean:<10.6f}\t{args.ood_set[i]:>12}\t{ood_mean:<10.6f}\tauroc\t{auroc:<4.2f}\taupr\t{aupr:<4.2f}\tfpr95\t{fpr:<4.2f}")
                ood_avg += ood_mean
                auroc_avg += auroc
                aupr_avg += aupr
                fpr_avg += fpr
            ood_avg /= ood_num
            auroc_avg /= ood_num
            aupr_avg /= ood_num
            fpr_avg /= ood_num
            print(f"mahalanobis\tID\t{id_mean:<10.6f}\tOOD\t{ood_avg:<10.6f}\tauroc\t{auroc_avg:<4.2f}\taupr\t{aupr_avg:<4.2f}\tfpr95\t{fpr_avg:<4.2f}")
            if epoch % args.save_every == 0:
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'labels': labels,
                    'scores': scores
                }, epoch, "mahalanobis")

    def save_checkpoint(self, state, epoch, measure):
        directory = f"{args.ckpt_base_dir}/{measure}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
        torch.save(state, filename)

    def compute_metrics(self, id_scores, ood_scores, recall_level=0.95):
        id_mean = id_scores.mean()
        ood_mean = ood_scores.mean()
        scores = torch.cat((id_scores, ood_scores), 0).cpu().numpy()
        labels = torch.cat((torch.ones_like(id_scores), torch.zeros_like(ood_scores)), 0).cpu().numpy()
        auroc = sk.roc_auc_score(labels, scores)
        aupr = sk.average_precision_score(labels, scores)
        fpr = fpr_and_fdr_at_recall(labels, scores, recall_level)
        return id_mean, ood_mean, auroc * 100, aupr * 100, fpr * 100, labels, scores
    
    def get_scores(self, dataloader, model, temperature=1.0):
        model.eval()
        energy_scores = torch.tensor([]).cuda()
        msp_scores = torch.tensor([]).cuda()
        logsoftmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.cuda()
                pred = model(x)
                if self.energy:
                    energy_score = temperature * torch.logsumexp(pred / temperature, 1)
                    energy_scores = torch.cat((energy_scores, energy_score), 0)
                if self.msp:
                    pred = logsoftmax(pred)
                    msp_score = torch.max(logsoftmax(pred), 1)[0]
                    msp_scores = torch.cat((msp_scores, msp_score), 0)
        return energy_scores, msp_scores

    def get_odin(self, dataloader, model, epsilon=0.0014, temper=1000.0):
        model.eval()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        scores = []
        temper = Variable(torch.tensor(temper).cuda(), requires_grad=True)
        for X, _ in dataloader:
            X = Variable(X.cuda(), requires_grad=True)
            # X = X.cuda()
            outputs = model(X)
            maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
            outputs /= temper
            
            tmp_labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
            loss = criterion(outputs, tmp_labels)
            loss.requires_grad_(True)
            loss.backward()
            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(X.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            tempInputs = torch.add(X.data, gradient, alpha=-epsilon)
            outputs = model(Variable(tempInputs))
            outputs = outputs / temper
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu().numpy()
            nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

            odin = np.max(nnOutputs, axis=1)
            scores += odin.tolist()
        return torch.tensor(scores).cuda()

    def get_mahalanobis(self, dataloader, model, sample_mean, precision, best_regressor, best_magnitude, num_output):
        model.eval()
        # sample_mean, precision, best_regressor, best_magnitude, num_output = tune_mahalanobis_hyperparams(model, args.num_classes, trainloader, dataloader)
        num_classes = args.num_classes
        scores = []
        for X, _ in dataloader:
            X = X.cuda()
            Mahalanobis_scores = get_Mahalanobis_score(X, model, num_classes, sample_mean, precision, num_output, best_magnitude)
            score = -best_regressor.predict_proba(Mahalanobis_scores)[:, 1].flatten()
            scores += score.tolist()
        return torch.tensor(scores).cuda()