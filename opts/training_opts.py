from .base_opts import BaseOptions
from utils import none_or_true


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--start_epoch', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--dropout', default=None, type=none_or_true)
        parser.add_argument('--dropout_p_second_to_last_layer', default=0.0, type=float)
        parser.add_argument('--dropout_p_last_layer', default=0.0, type=float)
        parser.add_argument('--lambda_0', default=0.5, type=float)
        # training parameters
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='optimizer selected for training')
        parser.add_argument('--heteroscedastic', default=False, type=bool)
        parser.add_argument('--warmup_epochs', type=int, default=10,
                            help='number of warmup epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001,
                            help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--weight_decay', type=float, default=1.e-4,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--scheduler', type=str, default='anneal',
                            help='Scheduling algorithm')
        parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
        parser.add_argument('--dim_common', type=int, default=448)
        parser.add_argument('--experiment', type=str, default=None,
                            help='Experiment name convention')
        parser.add_argument('--loss_weight', type=float, default=0.5,
                            help='Experiment name convention')
        parser.add_argument('--loss_imaging', type=str, default="NTX",
                            help='Considers loss for imaging domain')
        parser.add_argument('--loss_tabular', type=str, default="NTX",
                            help='Considers loss for tabular domain')
        parser.add_argument('--estimator', default='HCL', type=str, help='Choose loss function')
        parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
        parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')
        parser.add_argument('--alpha_weight', default=0.75, type=float, help='Alpha weight values')
        parser.add_argument('--wandb_project', default="AQ Multimodal SSL", type=str, help='Project Name')
        parser.add_argument('--wandb_entity', default="Exp1-Multimodal SSL", type=str, help='Entitny Name')
        parser.add_argument('--offline', default=False, type=bool, help='Set offline learning for wandb')
        parser.add_argument('--log_images', action='store_true', help='Logs images views')
        parser.add_argument('--check_val_every_n_epoch', default=1, type=int, help='Validation every 1 epoch')
        self.isTrain = True
        return parser
