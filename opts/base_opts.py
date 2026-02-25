import argparse
import torch


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--seed', default=0, type=int, help='seed for training')
        parser.add_argument('--dataroot', default='./data', help='path to images ')
        parser.add_argument('--samples_file', default="./data/editted/pollutant_ssl.csv", type=str)
        parser.add_argument('--datatype', default='multimodal', type=str, help='Trainings model using multmodal simclr')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--channels', type=int, default=12, help='Channels dimension form main sentinel backbone')
        parser.add_argument('--fusion_type', type=str, default='late', help='Fusion level for images')
        # model parameters
        parser.add_argument('--model', type=str, default='mmaq', help='chooses which model to use. [mmsimclr | mmbyol | mm_con ]')
        parser.add_argument('--network', type=str, default='resnet50', help='specify generator architecture [resnet50 | vit_b16 | vit_b32 | vit_l16 | vit_l32]')
        parser.add_argument('--s5pnet', type=str, default='resnet50', help='specify generator architecture [initial | resnet50 | vit_b16 | vit_b32 | vit_l16 | vit_l32]')
        parser.add_argument('--n_layers_tabular', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dim fof the SimCLR layer')
        parser.add_argument('--projection_dim', type=int, default=128, help='Projection dim fof the SimCLR layer')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--measurements', type=str, default='no2', help='Predicted variable')
        parser.add_argument('--tabular', action='store_true', help='enables tabular data')
        parser.add_argument('--tabular_net', type=str, default="initial", help="enables tabular network training")
        parser.add_argument('--decoupling', action='store_false')
        parser.add_argument('--imaging_embedding', type=int, default=2048, help='Sentinel-2 network '
                                                                             'number of features')
        parser.add_argument('--s5p_net_features', type=int, default=128, help='Sentinel-5P network '
                                                                             'number of features')
        parser.add_argument('--tabular_net_features', type=int, default=512, help='tabular net '
                                                                                 'number of '
                                                                                 'features')
        parser.add_argument('--tabular_input', type=int, default=8, help='input features in tab '
                                                                         'net')
        parser.add_argument('--head_features', type=int, default=8, help='head features')
        # dataset parameters
        parser.add_argument('--num_workers', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--max_epochs', default=500, type=int, help='Number of max epochs in training')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--output_filename', type=str, default='results.txt', help='results file name')
        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--lr_imaging', default=3.e-4, type=float, help='learning rate for imaging network')
        parser.add_argument('--lr_tabular', default=3.e-4, type=float, help='learning rate for tabular network')
        parser.add_argument('--corruption_rate', default=0.8, type=float, help='Corruption rate for tabular part')
        parser.add_argument('--augmentation_rate', default=0.6, type=float, help='Augmentation rate for tabular part')
        parser.add_argument('--correlation', default="canonical", type=str, help='identifies the correlation type in mmaq')
        parser.add_argument('--tabular_transform', default="marginal", type=str, help='Type of transformation that is applied in tabular part')
        parser.add_argument('--temperature', default=0.1, type=float, help='temperature value for pretraining')
        parser.add_argument('--limit_train_batches', default=1.0, type=float, help='limits training batches')
        parser.add_argument('--limit_val_batches', default=1.0, type=float, help='limits val batches')
        parser.add_argument('--enable_progress_bar', default=True, type=bool, help='Enables progress bar at training')
        parser.add_argument('--regressor_freq', default=5, type=int, help='Regressor frequency every N epochs')
        parser.add_argument('--linear_eval', default=False, type=bool, help='Is true only when linear evaluation or fine tuning is performed')
        parser.add_argument('--finetune_loss', default="mse", type=str, choices=['mse', 'rlp', 'contrastive_regression', 'ordinal'], help='Supervised fine tuning loss')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix


        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
                
        opt.num_gpus = len(opt.gpu_ids)
        opt.distributed = opt.num_gpus > 1
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt