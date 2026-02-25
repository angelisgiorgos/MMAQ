from .base_opts import BaseOptions
from utils import none_or_true


class TransferLearningOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--tf_datapath', default="/data/angelisg/sociobee/CO2_MultiModal/data", type=str, help='freeze backbone model')
        parser.add_argument('--tf_channels', default="0,1,2,3,4,5,6,7,8,9,10,11", type=str)
        parser.add_argument('--start_epoch', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--max_num_epochs', type=int, default=90, help='number of linear evaluation epochs')
        parser.add_argument('--freeze', action='store_true', help='freeze backbone model')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate for training of linear regressor')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training of linear regressor')
        parser.add_argument('--ckpt_path', type=str, default=None,
                            help='File checkpoint initial value')
        parser.add_argument('--wandb_project', default="AQ Multimodal SSL - Transfer Learning", type=str, help='Project Name')
        parser.add_argument('--wandb_entity', default="Exp1-Transfer Learning", type=str, help='Entitny Name')
        parser.add_argument('--offline', default=False, type=bool, help='Set offline learning for wandb')
        parser.add_argument('--log_images', action='store_true', help='Logs images views')
        parser.add_argument("--img_size", default=224, type=int, help="image size of the input")
        parser.add_argument('--check_val_every_n_epoch', default=1, type=int, help='Validation every 1 epoch')
        self.isTrain = True
        return parser