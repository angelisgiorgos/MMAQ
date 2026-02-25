from .multimodal.mm_byol import MM_BYOL
from .multimodal.bestofbothworlds import BestofBothWorlds
from .multimodal.multimodalssl import MultimodalContrastiveSimCLR
from .multimodal.mm_barlow_twins import MM_BarlowTwins
from .multimodal.decur import DeCUR
from .simclr import SimCLR
from .barlowtwins import BarlowTwins
from .byol import BYOL
from .vicreg import VICReg
from .mocov2 import MoCoV2
from .simsiam import SimSiam
from .multimodal.mmaq import MMAQ
from .dino import DINO


def build_ssl_model(args, data_stats):
    if args.model == "mmcl":
        model = BestofBothWorlds(args, data_stats)
    elif args.model == "mmbyol":
        model = MM_BYOL(args)
    elif args.model == "mm_con":
        model = MultimodalContrastiveSimCLR(args)
    elif args.model == "mm_barlow_twins":
        model = MM_BarlowTwins(args)
    elif args.model == "decur":
        model = DeCUR(args, data_stats)
    elif args.model == "simclr":
        model = SimCLR(args, data_stats)
    elif args.model == "barlowtwins":
        model = BarlowTwins(args, data_stats)
    elif args.model == 'byol':
        model = BYOL(args, data_stats)
    elif args.model == "vicreg":
        model = VICReg(args, data_stats)
    elif args.model == "simsiam":
        model = SimSiam(args, data_stats)
    elif args.model == "mocov2":
        model = MoCoV2(args, data_stats)
    elif args.model == "mmaq":
        model = MMAQ(args, data_stats)
    elif args.model == "dino":
        model = DINO(args, data_stats)
    return model