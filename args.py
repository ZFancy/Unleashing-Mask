import argparse
import sys
import yaml

from configs import parser as _parser

args = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    # General Config
    # basic training
    parser.add_argument("-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture")
    parser.add_argument('--layers', default= 100, type=int, help='total number of layers (default: 100) for DenseNet')
    parser.add_argument("--width-mult", default=1.0, help="How much to vary the width of the network.", type=float)
    parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
    parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='To not use bottleneck block')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
    parser.add_argument('--widen-factor', default=4, type=int, help='widen factor (default: 4)')

    parser.add_argument("--data", help="path to dataset base directory", default="/datasets")
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size (default: 256)")
    parser.add_argument("--set","--id-set", help="name of dataset", type=str, default="CIFAR10")
    parser.add_argument("--ood-set", help="name of ood dataset", type=lambda x: [str(a) for a in x.split(",")], default=["SVHN"])
    parser.add_argument("-j", "--workers", default=20, type=int, metavar="N", help="number of data loading workers (default: 20)")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--nesterov", default=False, action="store_true", help="Whether or not to use nesterov for SGD")
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr")
    parser.add_argument("--lr-policy", default="cosine_lr", type=str, help="learning rate policy")
    parser.add_argument("--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr")
    parser.add_argument("--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=None, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("--warmup_length", default=0, type=int, help="Number of warmup iterations")
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument("--no-bn-decay", action="store_true", default=False, help="No batchnorm decay")
    parser.add_argument("--scale-fan", action="store_true", default=False, help="scale fan")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")], help="Which GPUs to use for multigpu training")
    parser.add_argument("--low-data", default=1, help="Amount of data to use", type=float)
    parser.add_argument("--one-batch", action="store_true", help="One batch train set for debugging purposes (test overfitting)")
    parser.add_argument("--label-smoothing", type=float, help="Label smoothing to use, default 0.0", default=None)
    parser.add_argument("--trainer", type=str, default="default", help="cs, ss, or standard training")
    # about config
    parser.add_argument("--config", help="Config file to use (see configs dir)", default=None)
    parser.add_argument("--name", default=None, type=str, help="Experiment name to append to filepath")
    parser.add_argument("--log-dir", help="Where to save the runs. If None use ./runs", default=None)
    parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
    parser.add_argument("--save_every", default=1, type=int, help="Save every ___ epochs")
    # not about training
    parser.add_argument("--resume", default="", type=str,  metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--pretrain", action='store_true', help='pretrain a model and store it in addr[--pretrained]')
    #about pruning
    parser.add_argument("--random-subnet", action="store_true", help="Whether or not to use a random subnet when fine tuning for lottery experiments")
    parser.add_argument("--conv-type", type=str, default=None, help="What kind of sparsity to use for convolutional layers")
    parser.add_argument("--linear-type", type=str, default="DenseLinear", help="What kind of sparsity to use for linear layers")
    parser.add_argument("--freeze-weights", action="store_true", help="Whether or not to train only subnet (this freezes weights)")
    parser.add_argument("--freeze-subnet", action="store_true", help="Whether or not to train only subnet (this freezes subnet)")
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument("--nonlinearity", default="relu", help="Nonlinearity used by initialization")
    parser.add_argument("--first-layer-dense", action="store_true", help="First layer dense or sparse")
    parser.add_argument("--last-layer-dense", action="store_true", help="Last layer dense or sparse")
    parser.add_argument("--first-layer-type", type=str, default=None, help="Conv type of first layer")
    parser.add_argument("--score-init-constant", type=float, default=None, help="Sample Baseline Subnet Init")
    parser.add_argument("--init", default="kaiming_normal", help="Weight initialization modifications" )
    
    parser.add_argument('--distance', action='store_true', help='To use KL divergence as a distance measure')
    parser.add_argument('--msp', action='store_true', help='To use MSP as a measure')
    parser.add_argument('--energy', action='store_true', help='To use energy as a measure')
    parser.add_argument('--odin', action='store_true', help='TO use odin as a measure')
    parser.add_argument('--mahalanobis', action='store_true', help='To use mahalanobis as a measure')

    parser.add_argument('--temperature', default=1.0, type=float, help='Temperature for energy calculation')
    parser.add_argument('--alpha', default=0.0, type=float, help='Trade-off for energy based loss')
    # parser.add_argument('--beta', default=1.0, type=float, help='Trade-off for CrossENtropy in energy-based loss')
    parser.add_argument('--UM', default=0.0, type=float, help='Minimum training loss')
    parser.add_argument('--criterion', default="CrossEntropyLoss", type=str, help='Loss function')
    parser.add_argument('--gamma', default=0.0, type=float, help="KL divergence as a term in loss function")
    parser.add_argument("--prune-rate", default=0.0, help="Amount of pruning to do during sparse training", type=float)
    parser.add_argument('--prune-iterations', default=1, type=int, help="The iterations of prune")
    parser.add_argument('--prune_type', default="layer_wise", choices=["layer_wise", "model_wise"], help="Prune every layer to a same percentile or prune the parameters as a whole")
    parser.add_argument('--prune-subject', default='scores', choices=['scores', 'weight'], help='Prune weight | score of weight')
    parser.add_argument('--retrain-epochs', type=int, default=100, help="epochs when retraining")

    #poem
    parser.add_argument('--in-dataset', default="CIFAR10", type=str, help='in-distribution dataset e.g. CIFAR10')
    parser.add_argument('--save-epoch', default= 10, type=int, help='save the model every save_epoch') # freq; save model state_dict()
    # wideresnet
    parser.add_argument('--depth', default=40, type=int,  help='depth of wide resnet')
    parser.add_argument('--width', default=4, type=int,  help='width of resnet')
    ## network spec
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
    parser.add_argument('--beta', default=1.0, type=float, help='beta for out_loss')
    # ood sampling and mining
    parser.add_argument('--ood-batch-size', default= 2000, type=int, help='mini-batch size (default: 400) used for ood mining')
    parser.add_argument('--pool-size', default= 200, type=int, help='pool size')
    #posterior sampling
    parser.add_argument('--a0', type=float, default=6.0, help='a0')
    parser.add_argument('--b0', type=float, default=6.0, help='b0')
    parser.add_argument('--lambda_prior', type=float, default=0.25, help='lambda_prior')
    parser.add_argument('--sigma', type=float, default=20, help='control var for weights')
    parser.add_argument('--sigma_n', type=float, default=0.5, help='control var for noise')
    parser.add_argument('--conf', type=float, default=3.9, help='control ground truth for bayesian linear regression. 2.95--0.05; 3.9--0.98; 4.6 --0.99; 6.9--0.999')
    # saving, naming and logging
    parser.add_argument('--auxiliary-dataset', default='ImageNet', choices=['ImageNet','80m_tiny_images', 'partial_imagenet'], type=str, help='which auxiliary dataset to use')
    parser.add_argument('--log_name', help='Name of the Log File', type = str, default = "info.log")
    parser.add_argument('--ood_factor', type=float, default= 1, help='ood_dataset_size = len(train_loader.dataset) * ood_factor default = 2.0')
    parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
    parser.add_argument('--energy_model', default='True', type=bool, help='if use energy model')
    parser.add_argument('--debug', default='True', type=bool, help='if in debug mode')
    parser.add_argument('--m_in', type=float, default=-25., help='default: -25. margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='default: -7. margin for out-distribution; below this value will be penalized')
    parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
    parser.add_argument('--BUF_SIZE', type= int, default=4, help='# of data points (measured w.r.t. # of epochs) used for posterior update')
    parser.add_argument('--test_epochs', default = "80 90 100", type=str, help='# epoch to test performance')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--final', action='store_true', help='if need to plot histogram')
    parser.add_argument('--sample', default="thompson", choices=["thompson", "random"], help='if need to plot histogram')

    parser.add_argument('--UM-whole', default=0.0, type=float, help="UM as a whole")
    parser.add_argument('--UM-ce', default=0.0, type=float, help="CrossEntropy UM")
    parser.add_argument('--UM-ine', default=0.0, type=float, help="ID energy UM")
    parser.add_argument('--UM-oute', default=0.0, type=float, help="OOD energy UM")
    parser.add_argument('--UM-e', default=0.0, type=float, help="energy UM")
    parser.add_argument('--UMe', action="store_true", help="energy as UM")
    parser.add_argument('--oe', action="store_true", help="Outliers Exposure")

    # mask forget
    parser.add_argument('--reduction', default="mean", choices=["mean", "sum", "none"], help="reduction approach")

    parser.add_argument('--lr_drop_epoch', default=0.5, type=float, help="one drop learning rate")
    parser.add_argument('--lr_one_drop', default=0.05, type=float, help="learning drop rate")



    parser.set_defaults(augment=True)
    parser.set_defaults(bottleneck=True)
    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args

def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)

def run_args():
    global args
    if args is None:
        args = parse_arguments()

run_args()
