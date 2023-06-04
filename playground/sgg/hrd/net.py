
from cvpods.modeling import build_resnet_backbone
from cvpods.modeling.backbone import Backbone
from cvpods.layers import ShapeSpec

import logging
import sys

from playground.sgg.hrd.modeling.meta_arch.hgtr import HGTR

sys.path.append("..")


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_model(cfg):
    cfg.build_backbone = build_backbone
    model = HGTR(cfg)
    model.cuda()
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
