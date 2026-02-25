from opts.base_opts import BaseOptions
from utils import none_or_true


class RunOptions(BaseOptions):
    """
    Consolidated options class for the unified run.py script.
    Inherits from BaseOptions and adds task-specific arguments.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Core Task Definition
        parser.add_argument(
            '--task',
            type=str,
            required=True,
            choices=['pretrain', 'linear_eval', 'fine_tune', 'transfer_learning', 'transfer_segmentation'],
            help='Select the task to run.'
        )

        # Common evaluation/training args
        parser.add_argument('--start_epoch', type=int, default=1,
                            help='the starting epoch count')
        parser.add_argument('--max_num_epochs', type=int, default=90, help='number of evaluation/training epochs')
        parser.add_argument('--freeze', action='store_true', help='freeze backbone model')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        parser.add_argument('--ckpt_path', type=str, default=None,
                            help='File checkpoint path to load')
        parser.add_argument('--wandb_project', default="AQ Multimodal SSL", type=str, help='Project Name')
        parser.add_argument('--wandb_entity', default="Exp1-Multimodal SSL", type=str, help='Entity Name')
        parser.add_argument('--offline', action='store_true', help='Set offline learning for wandb')
        parser.add_argument('--log_images', action='store_true', help='Logs images views')
        parser.add_argument('--check_val_every_n_epoch', default=1, type=int, help='Validation every N epochs')

        # Pretraining-specific args (from training_opts.py)
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--dropout', default=None, type=none_or_true)
        parser.add_argument('--dropout_p_second_to_last_layer', default=0.0, type=float)
        parser.add_argument('--dropout_p_last_layer', default=0.0, type=float)
        parser.add_argument('--lambda_0', default=0.5, type=float)
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='optimizer selected for training')
        parser.add_argument('--heteroscedastic', action='store_true')
        parser.add_argument('--warmup_epochs', type=int, default=10,
                            help='number of warmup epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
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
        parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class prior')
        parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')
        parser.add_argument('--alpha_weight', default=0.75, type=float, help='Alpha weight values')

        # Transfer Learning / Segmentation args (from tf_opts.py)
        parser.add_argument('--tf_datapath', default="/data/angelisg/sociobee/CO2_MultiModal/data", type=str, help='transfer learning data path')
        parser.add_argument('--tf_channels', default="0,1,2,3,4,5,6,7,8,9,10,11", type=str)
        parser.add_argument("--img_size", default=224, type=int, help="image size of the input")

        self.isTrain = True
        return parser
