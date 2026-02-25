from losses.clip_loss import CLIPLoss, MMNTXentLoss
from losses.ntx_ent_loss import NTXentLoss
from losses.mm_loss import Multimodal_Loss
from losses.hypersphere import HypersphereLoss
from losses.cosine_similarity_loss import CosineSimilarityLoss
from losses.tico import TiCoLoss
from losses.dcl import DCL, DCLW
from lightly.loss import NegativeCosineSimilarity
from lightly.loss import BarlowTwinsLoss
from losses.dro_loss import DRO_Loss
from losses.decur_loss import DeCURLoss


def select_loss_imaging(args):
    if args.loss_imaging == "NTX":
        imaging_loss = NTXentLoss(temperature=args.temperature)
    elif args.loss_imaging == "hypershpere":
        imaging_loss = HypersphereLoss()
    elif args.loss_imaging == "consine_similarity":
        imaging_loss = CosineSimilarityLoss()
    elif args.loss_imaging == "tico":
        imaging_loss = TiCoLoss()
    elif args.loss_imaging == "dcl":
        imaging_loss = DCL()
    elif args.loss_imaging == "dclw":
        imaging_loss = DCLW()
    elif args.loss_imaging == "negative_cosine":
        imaging_loss = NegativeCosineSimilarity()
    elif args.loss_imaging == "barlow_twins":
        imaging_loss = BarlowTwinsLoss()
    elif args.loss_imaging == "dro":
        imaging_loss = DRO_Loss(args)
    return imaging_loss


def select_loss_tabular(args):
    if args.loss_tabular == "NTX":
        tabular_loss = NTXentLoss(temperature=args.temperature)
    elif args.loss_tabular == "hypershpere":
        tabular_loss = HypersphereLoss()
    elif args.loss_tabular == "tico":
        tabular_loss = TiCoLoss()
    elif args.loss_tabular == "dcl":
        tabular_loss = DCL()
    elif args.loss_tabular == "dclw":
        tabular_loss = DCLW() 
    elif args.loss_tabular == "negative_cosine":
        tabular_loss = NegativeCosineSimilarity()
    elif args.loss_tabular == "barlow_twins":
        tabular_loss = BarlowTwinsLoss()
    elif args.loss_tabular == "dro":
        tabular_loss = DRO_Loss(args)
    return tabular_loss