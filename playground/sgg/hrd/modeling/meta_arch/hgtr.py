import copy
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch import relu

from cvpods.layers import ShapeSpec, position_encoding_dict
from cvpods.layers.mlp import MLP
from cvpods.layers.position_embedding import PositionEmbeddingSine
from cvpods.modeling import ResNet
from cvpods.modeling.backbone import Transformer
from cvpods.modeling.backbone.resnet import BasicStem
from cvpods.modeling.matcher import HungarianMatcher
from cvpods.modeling.meta_arch.detr import SetCriterion, PostProcess
from cvpods.modeling.meta_arch.one_stage_sgg.rel_detr_inference import RelPostProcess, RelPostProcessSingleBranch
from cvpods.modeling.meta_arch.one_stage_sgg.rel_detr_losses import RelSetCriterion, RelHungarianMatcher
from cvpods.structures import ImageList, Instances, Boxes
from cvpods.structures.boxes import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from cvpods.structures.relationship import Relationships
from playground.sgg.hrd.modeling.meta_arch.dab_transformer import DabDetrTransformerEncoder, DabDetrTransformerDecoder, \
    DabDetrTransformer
from playground.sgg.hrd.modeling.meta_arch.rel_detr import EntitiesIndexingHead, EntitiesIndexingHeadRuleBased
from playground.sgg.hrd.modeling.meta_arch.transformer import DetrTransformer, DetrTransformerEncoder, \
    DetrTransformerDecoder

import os

from playground.sgg.hrd.modeling.rel_detr_losses import AuxRelHungarianMatcher, AuxRelSetCriterion

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class HGTR(nn.Module):
    """Implement HGTR

    """

    def __init__(self, cfg):
        super().__init__()
        # define backbone and position embedding module
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        self.in_channels = cfg.MODEL.DETR.IN_CHANNELS
        self.in_features = cfg.MODEL.DETR.IN_FEATURES
        # project the backbone output feature
        # into the required dim for transformer block
        self.entity_embed_dim = self.cfg.MODEL.DETR.ENCODER.EMBED_DIM
        self.entity_position_embedding = position_encoding_dict[
            cfg.MODEL.DETR.POSITION_EMBEDDING
        ](
            num_pos_feats=self.entity_embed_dim // 2,
            temperature=cfg.MODEL.DETR.TEMPERATURE,
            normalize=True if cfg.MODEL.DETR.POSITION_EMBEDDING == "sine" else False,
            scale=None,
        )
        self.entity_input_proj = nn.Conv2d(self.in_channels, self.entity_embed_dim, kernel_size=1)

        # define learnable object queries and transformer module
        self.entity_transformer = Transformer(cfg)

        self.entity_num_queries = self.cfg.MODEL.ENTITY_NUM_QUERIES
        self.entity_num_classes = self.cfg.MODEL.ENTITY_NUM_CLASSES
        self.entity_query_embed = nn.Embedding(self.entity_num_queries, self.entity_embed_dim)

        # whether to freeze the initilized anchor box centers during training
        self.freeze_anchor_box_centers = self.cfg.MODEL.FREEZE_ANCHOR_BOX_CENTERS
        # define classification head and box head
        self.entity_class_embed = nn.Linear(self.entity_embed_dim, self.entity_num_classes + 1)
        self.entity_bbox_embed = MLP(input_dim=self.entity_embed_dim, hidden_dim=self.entity_embed_dim, output_dim=4,
                                     num_layers=3)

        self.weight_dict = {
            "loss_ce": cfg.MODEL.DETR.CLASS_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.DETR.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.DETR.GIOU_LOSS_COEFF,
        }

        losses = ["labels", "boxes", "cardinality"]

        entity_matcher = HungarianMatcher(
            cost_class=cfg.MODEL.DETR.COST_CLASS,
            cost_bbox=cfg.MODEL.DETR.COST_BBOX,
            cost_giou=cfg.MODEL.DETR.COST_GIOU,
        )

        # where to calculate auxiliary loss in criterion
        self.ent_aux_loss = not cfg.MODEL.DETR.NO_AUX_LOSS
        self.entity_criterion = SetCriterion(
            self.entity_num_classes,
            matcher=entity_matcher,
            weight_dict=self.weight_dict,
            eos_coef=cfg.MODEL.DETR.EOS_COEFF,
            losses=losses,
        )
        # normalizer for input raw images
        self.device = self.cfg.DEVICE
        pixel_mean = torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        if not cfg.MODEL.RESNETS.STRIDE_IN_1X1:
            # Custom or torch pretrain weights
            self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std
        else:
            # MSRA pretrain weights
            self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.relation_embed_dim = self.cfg.MODEL.RELATION_EMBED_DIM
        self.relation_position_embedding = PositionEmbeddingSine(
            num_pos_feats=self.cfg.MODEL.RELATION_POSITION_EMBEDDING.NUM_POS_FEATS,
            temperature=self.cfg.MODEL.RELATION_POSITION_EMBEDDING.TEMPERATURE,
            normalize=self.cfg.MODEL.RELATION_POSITION_EMBEDDING.NORMALIZE
        )
        self.relation_input_proj = nn.Conv2d(self.in_channels, self.relation_embed_dim, kernel_size=1)
        self.roi_pool = ops.RoIPool(output_size=(1, 1), spatial_scale=1)
        self.relation_encoder = DetrTransformerEncoder(
            embed_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENCODER.EMBED_DIM,
            num_heads=self.cfg.MODEL.RELATION_TRANSFORMER.ENCODER.NUM_HEADS,
            attn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENCODER.ATTN_DROPOUT,
            feedforward_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENCODER.FEEDFORWARD_DIM,
            ffn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENCODER.FFN_DROPOUT,
            num_layers=self.cfg.MODEL.RELATION_TRANSFORMER.ENCODER.NUM_LAYERS
        )
        self.relation_decoder_global = DetrTransformerDecoder(
            embed_dim=self.cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.EMBED_DIM,
            num_heads=self.cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.NUM_HEADS,
            attn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.ATTN_DROPOUT,
            feedforward_dim=self.cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.FEEDFORWARD_DIM,
            ffn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.FFN_DROPOUT,
            num_layers=self.cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.NUM_LAYERS,
            return_intermediate=self.cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.RETURN_INTERMEDIATE
        )
        self.relation_decoder_union = DetrTransformerDecoder(
            embed_dim=self.cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.EMBED_DIM,
            num_heads=self.cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.NUM_HEADS,
            attn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.ATTN_DROPOUT,
            feedforward_dim=self.cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.FEEDFORWARD_DIM,
            ffn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.FFN_DROPOUT,
            num_layers=self.cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.NUM_LAYERS,
            return_intermediate=self.cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.RETURN_INTERMEDIATE
        )
        self.relation_decoder_subject = DetrTransformerDecoder(
            embed_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.EMBED_DIM,
            num_heads=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_HEADS,
            attn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.ATTN_DROPOUT,
            feedforward_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.FEEDFORWARD_DIM,
            ffn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.FFN_DROPOUT,
            num_layers=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_LAYERS,
            return_intermediate=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.RETURN_INTERMEDIATE
        )
        self.relation_decoder_predicate = DetrTransformerDecoder(
            embed_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.EMBED_DIM,
            num_heads=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_HEADS,
            attn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.ATTN_DROPOUT,
            feedforward_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.FEEDFORWARD_DIM,
            ffn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.FFN_DROPOUT,
            num_layers=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_LAYERS,
            return_intermediate=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.RETURN_INTERMEDIATE
        )
        self.relation_decoder_object = DetrTransformerDecoder(
            embed_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.EMBED_DIM,
            num_heads=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_HEADS,
            attn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.ATTN_DROPOUT,
            feedforward_dim=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.FEEDFORWARD_DIM,
            ffn_dropout=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.FFN_DROPOUT,
            num_layers=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_LAYERS,
            return_intermediate=self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.RETURN_INTERMEDIATE
        )
        self.use_gt_box = cfg.MODEL.REL_DETR.USE_GT_ENT_BOX
        self.relation_num_class = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.relation_num_query = self.cfg.MODEL.RELATION_NUM_QUERY
        if cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
            self.relation_class_embed = nn.Linear(self.relation_embed_dim, self.relation_num_class)
        else:
            self.relation_class_embed = nn.Linear(self.relation_embed_dim, self.relation_num_class + 1)
        self.relation_vector_embed = nn.Linear(self.relation_embed_dim, 4)
        self.obj_class_embed = nn.Linear(self.relation_embed_dim, self.entity_num_classes + 1)
        self.obj_bbox_embed = MLP(self.relation_embed_dim, self.relation_embed_dim, 4, 3)
        self.sub_class_embed = nn.Linear(self.relation_embed_dim, self.entity_num_classes + 1)
        self.sub_bbox_embed = MLP(self.relation_embed_dim, self.relation_embed_dim, 4, 3)

        # parameter
        self.w_union = nn.Parameter(torch.randn(self.relation_embed_dim, self.relation_embed_dim))
        self.w_predicate = nn.Parameter(torch.randn(self.relation_embed_dim, self.relation_embed_dim))
        self.w_q = nn.Parameter(torch.randn(4, self.relation_embed_dim))

        if self.use_gt_box:
            self.gt_aux_class = nn.Linear(self.relation_embed_dim, self.num_classes + 1)
            self.gt_aux_bbox = MLP(self.relation_embed_dim, self.relation_embed_dim, 4, 3)
        self.rel_aux_loss = not cfg.MODEL.REL_DETR.NO_AUX_LOSS
        self.entities_indexing_heads = None
        # heads
        self.indexing_module_type = "rule_base"
        self.entities_indexing_heads_rule = EntitiesIndexingHeadRuleBased(cfg)
        if self.indexing_module_type == "feat_att":
            self.entities_indexing_heads = nn.ModuleDict({
                "sub": EntitiesIndexingHead(self.cfg),
                "obj": EntitiesIndexingHead(self.cfg),
            })
        elif self.indexing_module_type in ["rule_base", 'rel_vec']:
            self.entities_indexing_heads = self.entities_indexing_heads_rule
        else:
            assert False
            # 后置处理器
        self.post_processors = {
            "bbox": PostProcess(),
            "rel": RelPostProcess(cfg),
        }  # relationship PostProcess
        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.USE_ENTITIES_PRED:
            self.post_processors['rel'] = RelPostProcessSingleBranch(cfg)
        self.weight_dict.update(
            {
                "loss_rel_ce": cfg.MODEL.REL_DETR.CLASS_LOSS_COEFF,
                "loss_rel_vector": cfg.MODEL.REL_DETR.REL_VEC_LOSS_COEFF,
                "loss_aux_obj_entities_boxes": cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_BOX_L1_LOSS_COEFF,
                "loss_aux_sub_entities_boxes": cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_BOX_L1_LOSS_COEFF,
                "loss_aux_obj_entities_boxes_giou": cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_BOX_L1_LOSS_COEFF,
                "loss_aux_sub_entities_boxes_giou": cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_BOX_L1_LOSS_COEFF,
                "loss_aux_obj_entities_labels_ce": cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_CLS_LOSS_COEFF,
                "loss_aux_sub_entities_labels_ce": cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENT_CLS_LOSS_COEFF,
            }
        )
        rel_matcher = RelHungarianMatcher(
            cfg,
            cost_rel_class=cfg.MODEL.REL_DETR.COST_CLASS,
            cost_rel_vec=cfg.MODEL.REL_DETR.COST_REL_VEC,
            cost_class=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_ENT_CLS,
            cost_bbox=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_L1,
            cost_giou=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_GIOU,
            cost_indexing=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_INDEXING,
        )
        # relationship criterion,
        losses = set(copy.deepcopy(cfg.MODEL.REL_DETR.LOSSES))
        if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
            losses.add("rel_entities_aware")
            if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENTITIES_INDEXING:
                losses.add("rel_entities_indexing")

        if (
                cfg.MODEL.REL_DETR.DYNAMIC_QUERY
                and cfg.MODEL.REL_DETR.DYNAMIC_QUERY_AUX_LOSS_WEIGHT is not None
        ):
            losses.add("rel_dynamic_query")

        self.disalign_train = False
        try:
            if cfg.MODEL.DISALIGN.ENABLED:
                self.disalign_train = True
        except AttributeError:
            pass

        if self.disalign_train:
            losses.add("disalign_loss")
            self.rel_class_embed_disalign_weight_cp = False
            self.rel_class_embed_disalign = nn.Linear(self.relation_embed_dim, self.relation_num_class + 1)
        else:
            if cfg.MODEL.REL_DETR.FOCAL_LOSS.ENABLED:
                losses.add("rel_labels_fl")
            else:
                losses.add("rel_labels")

        if self.ent_aux_loss or self.rel_aux_loss:
            self.aux_weight_dict = {}
            for i in range(cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.NUM_LAYERS - 1):
                self.aux_weight_dict.update(
                    {
                        k + f"/global/layer{i}": v * cfg.MODEL.REL_DETR.AUX_LOSS_WEIGHT
                        for k, v in self.weight_dict.items()
                    }
                )
            for i in range(cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.NUM_LAYERS - 1):
                self.aux_weight_dict.update(
                    {
                        k + f"/union/layer{i}": v * cfg.MODEL.REL_DETR.AUX_LOSS_WEIGHT
                        for k, v in self.weight_dict.items()
                    }
                )
            for i in range(cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_LAYERS - 1):
                self.aux_weight_dict.update(
                    {
                        k + f"/entity/layer{i}": v * cfg.MODEL.REL_DETR.AUX_LOSS_WEIGHT
                        for k, v in self.weight_dict.items()
                    }
                )
            self.weight_dict.update(self.aux_weight_dict)

        if self.rel_aux_loss:
            losses = set(copy.deepcopy(cfg.MODEL.REL_DETR.LOSSES))

            if cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
                losses.add("rel_entities_aware")
        aux_weight_dict = copy.deepcopy(self.weight_dict)

        for i in range(cfg.MODEL.RELATION_TRANSFORMER.GLOBAL_DECODER.NUM_LAYERS - 1):
            for k, v in self.weight_dict.items():
                aux_weight_dict[k] = v

        for i in range(cfg.MODEL.RELATION_TRANSFORMER.UNION_DECODER.NUM_LAYERS - 1):
            for k, v in self.weight_dict.items():
                aux_weight_dict[k] = v

        for i in range(cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_LAYERS - 1):
            for k, v in self.weight_dict.items():
                aux_weight_dict[k] = v
        self.rel_criterion = RelSetCriterion(
            cfg,
            self.relation_num_class,
            matcher=rel_matcher,
            weight_dict=aux_weight_dict,
            eos_coef=cfg.MODEL.REL_DETR.EOS_COEFF,
            losses=list(losses),
        )
        self.rel_criterion_aux = AuxRelSetCriterion(
            cfg,
            self.relation_num_class,
            matcher=AuxRelHungarianMatcher(
                cfg,
                cost_rel_class=cfg.MODEL.REL_DETR.COST_CLASS,
                cost_rel_vec=cfg.MODEL.REL_DETR.COST_REL_VEC,
                cost_class=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_ENT_CLS,
                cost_bbox=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_L1,
                cost_giou=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_BOX_GIOU,
                cost_indexing=cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.COST_INDEXING,
            ),
            weight_dict=aux_weight_dict,
            eos_coef=cfg.MODEL.REL_DETR.EOS_COEFF,
            losses=list(losses),
        )

        self.iou_threshold = -1
        self._reset_parameters()
        # visualization
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.use_gt_box = cfg.MODEL.REL_DETR.USE_GT_ENT_BOX
        self.to(self.device)

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries.
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        # 1. 图片预处理
        images = self.preprocess_image(batched_inputs)
        targets = self.convert_anno_format(batched_inputs)

        device = images.tensor.device

        image_sizes = torch.stack(
            [
                torch.tensor(
                    [bi.get("height", img_size[0]), bi.get("width", img_size[1]), ],
                    device=self.device,
                )
                for bi, img_size in zip(batched_inputs, images.image_sizes)
            ]
        )
        B, C, H, W = images.tensor.shape
        device = images.tensor.device

        img_masks = torch.ones((B, H, W), dtype=torch.bool, device=device)
        spatial_shapes = []
        for img_shape, m in zip(images.image_sizes, img_masks):
            m[: img_shape[0], : img_shape[1]] = False
            spatial_shape = (img_shape[0], img_shape[1])
            spatial_shapes.append(spatial_shape)

        # 2. 目标检测
        # only use last level feature in DAB-DETR
        features = self.backbone(images.tensor)["res5"]
        # features = self.entity_input_proj(features)
        img_masks = F.interpolate(img_masks[None].float(), size=features.shape[-2:]).bool()[0]
        entity_pos_embed = self.entity_position_embedding(features, img_masks)

        entity_hidden_states, _ = self.entity_transformer(
            self.entity_input_proj(features), img_masks, self.entity_query_embed.weight, entity_pos_embed,
            enc_return_lvl=self.cfg.MODEL.REL_DETR.TRANSFORMER.SHARE_ENC_FEAT_LAYERS,

        )
        # Calculate entity_output coordinates and classes.
        entity_outputs_class = self.entity_class_embed(entity_hidden_states)
        entity_outputs_coord = self.entity_bbox_embed(entity_hidden_states).sigmoid()
        layers_output_class = entity_outputs_class[-1]
        layers_output_coord = entity_outputs_coord[-1]
        entity_output = {"pred_logits": layers_output_class, "pred_boxes": layers_output_coord}

        # 3. 关系预测
        relation_pos_embed = self.relation_position_embedding(img_masks)
        relation_input = self.relation_input_proj(features)
        bs, c, h, w = relation_input.shape
        relation_encoder_query = relation_input.view(bs, c, -1).permute(2, 0, 1)
        relation_encoder_embed = relation_pos_embed.view(bs, c, -1).permute(2, 0, 1)
        relation_encoder_masks = img_masks.view(bs, -1)
        memory_global = self.relation_encoder(
            query=relation_encoder_query,
            key=None, value=None,
            query_pos=relation_encoder_embed,
            query_key_padding_mask=relation_encoder_masks
        )

        union_box = self.get_union_box(layers_output_coord, H, W).to(self.device)
        union_box = box_xyxy_to_cxcywh(union_box).to(self.device)
        union_feature = self.get_union_feature(relation_input, union_box).to(self.device)
        query_embed_rel_init = self.query_tgt_initialization(union_box, union_feature)
        out_global = self.relation_decoder_global(
            query=query_embed_rel_init.transpose(0, 1),
            key=memory_global,
            value=memory_global
        )
        out_global = out_global.transpose(1, 2)
        query_union = torch.matmul(out_global[-1], self.w_union).to(self.device)
        out_union = self.relation_decoder_union(
            query=query_union,
            key=union_feature,
            value=union_feature
        )
        query_predicate = torch.matmul(out_union[-1], self.w_predicate).to(self.device)
        relation_hidden_states = self.relation_decoder_predicate(
            query=query_predicate.clone().transpose(0, 1),
            key=entity_hidden_states[-1].transpose(0, 1),
            value=entity_hidden_states[-1].transpose(0, 1)
        )
        relation_hidden_states_subject = self.relation_decoder_subject(
            query=query_predicate.clone().transpose(0, 1),
            key=entity_hidden_states[-1].transpose(0, 1),
            value=entity_hidden_states[-1].transpose(0, 1)
        )
        relation_hidden_states_object = self.relation_decoder_object(
            query=query_predicate.clone().transpose(0, 1),
            key=entity_hidden_states[-1].transpose(0, 1),
            value=entity_hidden_states[-1].transpose(0, 1)
        )

        relation_hidden_states = relation_hidden_states.transpose(1, 2)
        relation_hidden_states_object = relation_hidden_states_object.transpose(1, 2)
        relation_hidden_states_subject = relation_hidden_states_subject.transpose(1, 2)

        pred_rel_logits = self.relation_class_embed(relation_hidden_states)  # n_lyr, batch_size, num_queries, N
        pred_rel_vec = self.relation_vector_embed(relation_hidden_states)  # batch_size, num_queries, 4
        pred_rel_vec = pred_rel_vec.sigmoid()
        #  pack prediction results
        semantic_predictions = {
            "pred_logits": layers_output_class,
            "pred_boxes": layers_output_coord,
            "pred_rel_logits": pred_rel_logits[-1],
            # layer, batch_size, num_queries, 4 => batch_size, num_queries, 4
            "pred_rel_vec": pred_rel_vec[-1]
            # take the output from the last layer
        }

        (entity_hidden_states,
         pred_rel_obj_box,
         pred_rel_obj_logits,
         pred_rel_sub_box,
         pred_rel_sub_logits,
         relation_hidden_states_subject,
         relation_hidden_states_object,
         pred_ent_rel_vec) = self.predicate_rel_ent_semantics(entity_hidden_states,
                                                              image_sizes,
                                                              relation_hidden_states,
                                                              relation_hidden_states_subject,
                                                              relation_hidden_states_object,
                                                              semantic_predictions)
        pred_rel_confidence = None

        rel_aux_out = self.generate_aux_out(image_sizes, entity_hidden_states, layers_output_class,
                                            layers_output_coord,
                                            pred_rel_logits, pred_rel_vec,
                                            relation_hidden_states_subject, relation_hidden_states_object,
                                            pred_rel_sub_box, pred_rel_obj_box,
                                            pred_rel_obj_logits, pred_rel_sub_logits)

        use_pre_comp_box = batched_inputs[0]["relationships"].has_meta_info("precompute_prop")

        if use_pre_comp_box:
            semantic_predictions["precompute_prop"] = [
                each["relationships"].get_meta_info("precompute_prop")
                for each in batched_inputs
            ]
        if self.training:
            if self.use_gt_box or use_pre_comp_box:
                # apply supervision for feature extraction heads
                semantic_predictions['pred_logits'] = entity_outputs_class[-1]
                semantic_predictions['pred_boxes'] = entity_outputs_coord[-1]

            loss_dict = self.loss_eval(batched_inputs, image_sizes, images, entity_outputs_class, entity_outputs_coord,
                                       pred_rel_confidence, rel_aux_out, semantic_predictions)
            return loss_dict

        else:
            if self.rel_aux_loss:
                semantic_predictions['aux_outputs'] = rel_aux_out
            pred_res = self.inference(images, batched_inputs, targets, semantic_predictions)
            return pred_res

    def generate_aux_out(self, image_sizes, entity_hidden_states, entity_outputs_class, entity_outputs_coord,
                         pred_rel_logits, pred_rel_vec,
                         relation_hidden_states_subject, relation_hidden_states_object,
                         pred_rel_sub_box, pred_rel_obj_box,
                         pred_rel_obj_logits, pred_rel_sub_logits):
        aux_out = []

        for ir in range(len(pred_rel_logits) - 1):
            tmp_out = {
                # take the output from the last layer
                "pred_logits": entity_outputs_class,
                "pred_boxes": entity_outputs_coord,
                # layer, batch_size, num_queries, 4
                #     => batch_size, num_queries, 4
                "pred_rel_logits": pred_rel_logits[ir],
                "pred_rel_vec": pred_rel_vec[ir],
            }

            if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.ENABLED:
                tmp_out.update({
                    "pred_rel_obj_logits": pred_rel_obj_logits[ir],
                    "pred_rel_obj_box": pred_rel_obj_box[ir],
                    "pred_rel_sub_logits": pred_rel_sub_logits[ir],
                    "pred_rel_sub_box": pred_rel_sub_box[ir],
                })

            if self.entities_indexing_heads is not None:
                (sub_idxing, obj_idxing, sub_idxing_rule, obj_idxing_rule) = self.graph_assembling(
                    tmp_out, image_sizes, entity_hidden_states,
                    relation_hidden_states_subject[ir], relation_hidden_states_object[ir],
                )

                tmp_out.update({
                    "sub_entities_indexing": sub_idxing,
                    "obj_entities_indexing": obj_idxing,
                    "sub_ent_indexing_rule": sub_idxing_rule,
                    "obj_ent_indexing_rule": obj_idxing_rule,
                })

            aux_out.append(tmp_out)

        return aux_out

    # loss eval
    def loss_eval(self, batched_inputs, image_sizes, images, outputs_class, outputs_coord, pred_rel_confidence,
                  rel_aux_out, semantic_predictions):
        targets = self.convert_anno_format(batched_inputs)
        if self.ent_aux_loss:
            semantic_predictions["aux_outputs"] = [
                {"pred_logits": ent_logits, "pred_boxes": ent_box, }
                for ent_logits, ent_box in zip(
                    outputs_class[:-1], outputs_coord[:-1],
                )
            ]
        loss_dict, ent_match_idx = self.entity_criterion(semantic_predictions, targets, with_match_idx=True)
        ent_det_res = self.post_processors["bbox"](semantic_predictions, image_sizes)

        if semantic_predictions.get("aux_outputs") is not None:
            # clear the entities aux_loss
            semantic_predictions.pop("aux_outputs")
        aux_rel_loss_dict = {}
        if not self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.REUSE_ENT_MATCH:
            ent_match_idx = None
        rel_loss_dict, match_res = self.rel_criterion(
            semantic_predictions,
            targets,
            # re-use the entities matching results
            det_match_res=ent_match_idx,
        )
        # generate the entities regrouping predicitons of each layer prediction
        # for auxiliary loss
        aux_rel_match = None
        if self.rel_aux_loss:
            semantic_predictions['aux_outputs'] = rel_aux_out
            aux_rel_loss_dict, aux_rel_match = self.aux_rel_loss_cal(
                semantic_predictions,  # last layer pred
                pred_rel_confidence,
                targets, ent_match_idx, match_res,
            )
        loss_dict.update(rel_loss_dict)
        loss_dict.update(aux_rel_loss_dict)
        for k, v in loss_dict.items():
            loss_dict[k] = v * self.weight_dict[k] if k in self.weight_dict else v
        return loss_dict

    def aux_rel_loss_cal(self, out, pred_rel_confidence, targets, ent_match_idx, match_res, ):

        aux_rel_loss_dict, aux_rel_match = self.rel_criterion_aux(
            out,
            targets,
            match_res=match_res,
            aux_only=True,
            ent_match_idx=ent_match_idx,
            # re-use the foreground entities matching results
            entities_match_cache=self.rel_criterion.entities_match_cache,
        )

        return aux_rel_loss_dict, aux_rel_match

    # inference time
    def inference(self, images, batched_inputs, targets, semantic_out, out_res=None):
        """
        Run inference on the given inputs.

        Args:
            images:
            batched_inputs (list[dict]): same as in :meth:`forward`
            semantic_out: the dict of models prediction, concatenated batch together

        Returns:

        """
        assert not self.training

        target_sizes = torch.stack(
            [
                torch.tensor(
                    [bi.get("height", img_size[0]), bi.get("width", img_size[1])],
                    device=self.device,
                )
                for bi, img_size in zip(batched_inputs, images.image_sizes)
            ]
        )
        ent_det_res = self.post_processors["bbox"](semantic_out, target_sizes)

        # list[{"scores": s, "labels": l, "boxes": b}]

        (rel_det_res, init_rel_det_res) = self.post_processors["rel"](
            semantic_out, ent_det_res, target_sizes
        )

        pred_res = HGTR._postprocess(
            ent_det_res, rel_det_res, batched_inputs, images.image_sizes
        )

        outputs_without_aux = {
            k: v for k, v in semantic_out.items() if k != "aux_outputs" and k != "ref_pts_pred"
        }
        (indices, match_cost, detailed_cost_dict) = self.rel_criterion.matcher(
            outputs_without_aux, targets, return_init_idx=True
        )

        # save the matching for evaluation in test/validation time
        if "sub_entities_indexing" in outputs_without_aux:
            # add the indexing module performance in test time
            num_ent_pairs = self.cfg.MODEL.REL_DETR.NUM_ENTITIES_PAIRING
            indices_reduce = []
            for p, g in indices:
                indices_reduce.append((torch.div(p, num_ent_pairs, rounding_mode='trunc'), g))

            loss_ent_idx = self.rel_criterion.loss_aux_entities_indexing(
                outputs_without_aux, targets, indices_reduce, None
            )
            for each in pred_res:
                for k, v in loss_ent_idx.items():
                    if "acc" in k:
                        each["relationships"].add_meta_info(k, v)

        return pred_res

    def predicate_rel_ent_semantics(self,
                                    entity_hidden_states,
                                    image_sizes,
                                    relation_hidden_states,
                                    relation_hidden_states_subject,
                                    relation_hidden_states_object,
                                    semantic_predictions
                                    ):

        relation_hidden_states_subject = relation_hidden_states_subject
        relation_hidden_states_object = relation_hidden_states_object

        pred_rel_sub_box = []
        pred_rel_obj_box = []
        pred_rel_obj_logits = []
        pred_rel_sub_logits = []

        for lid in range(len(relation_hidden_states_subject)):
            pred_rel_sub_logits.append(self.sub_class_embed[lid](relation_hidden_states_subject[lid]))
            pred_rel_sub_box.append(self.sub_bbox_embed[lid](relation_hidden_states_subject[lid]).sigmoid())
            pred_rel_obj_logits.append(self.obj_class_embed[lid](relation_hidden_states_object[lid]))
            pred_rel_obj_box.append(self.obj_bbox_embed[lid](relation_hidden_states_object[lid]).sigmoid())

        pred_rel_sub_logits = torch.stack(pred_rel_sub_logits)
        pred_rel_sub_box = torch.stack(pred_rel_sub_box)
        pred_rel_obj_logits = torch.stack(pred_rel_obj_logits)
        pred_rel_obj_box = torch.stack(pred_rel_obj_box)

        pred_ent_rel_vec = torch.cat((pred_rel_sub_box[..., :2], pred_rel_obj_box[..., :2]), dim=-1)
        semantic_predictions.update(
            {
                "pred_rel_obj_logits": pred_rel_obj_logits[-1],
                "pred_rel_obj_box": pred_rel_obj_box[-1],
                "pred_rel_sub_logits": pred_rel_sub_logits[-1],
                "pred_rel_sub_box": pred_rel_sub_box[-1],
                "pred_ent_rel_vec": pred_ent_rel_vec[-1]
            }
        )
        if self.entities_indexing_heads is not None:
            entity_hidden_states = entity_hidden_states[-1]
            (
                subject_index, object_index, subject_index_rule, object_index_rule,
            ) = self.graph_assembling(semantic_predictions,
                                      image_sizes,
                                      entity_hidden_states,
                                      relation_hidden_states_subject[-1],
                                      relation_hidden_states_object[-1])
            semantic_predictions.update({
                "sub_entities_indexing": subject_index,
                "obj_entities_indexing": object_index,
                "sub_ent_indexing_rule": subject_index_rule,
                "obj_ent_indexing_rule": object_index_rule,
            })
        return (entity_hidden_states,
                pred_rel_obj_box,
                pred_rel_obj_logits,
                pred_rel_sub_box,
                pred_rel_sub_logits,
                relation_hidden_states_subject,
                relation_hidden_states_object,
                pred_ent_rel_vec)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def get_union_box(self, entity_outputs_coord, H, W):

        batch_size, num_boxes, _ = entity_outputs_coord.shape

        # 将box坐标从cxcywh格式转换为xyxy格式
        boxes = box_cxcywh_to_xyxy(entity_outputs_coord).to(self.device)
        boxes_t = boxes.permute(0, 2, 1)

        # 计算并集
        x1 = torch.min(boxes[..., 0][:, :, None], boxes_t[..., 0][..., None, :])
        y1 = torch.min(boxes[..., 1][:, :, None], boxes_t[..., 1][..., None, :])
        x2 = torch.max(boxes[..., 2][:, :, None], boxes_t[..., 2][..., None, :])
        y2 = torch.max(boxes[..., 3][:, :, None], boxes_t[..., 3][..., None, :])

        # 计算并集的面积
        intersection_area = torch.clamp_min((x2 - x1) * (y2 - y1), min=0)
        boxes_area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        boxes_t_area = (boxes_t[..., 2] - boxes_t[..., 0]) * (boxes_t[..., 3] - boxes_t[..., 1])
        union_area = boxes_area[:, :, None] + boxes_t_area[:, None, :] - intersection_area

        # 计算iou
        iou = intersection_area / union_area

        # 选择iou大于阈值的并集框
        mask = (iou > self.iou_threshold) & (iou < 1.0)
        indices = torch.nonzero(mask)
        union_boxes = torch.zeros((batch_size, num_boxes, num_boxes, 4), dtype=torch.float32, device=self.device)
        union_boxes[indices[:, 0], indices[:, 1], indices[:, 2], :] = torch.stack(
            [x1[mask]*H, y1[mask]*W, x2[mask]*H, y2[mask]*W], dim=-1)

        # 取出对角线上的并集框
        union_boxes = torch.narrow(union_boxes, dim=1, start=0, length=num_boxes)
        union_boxes = union_boxes[:,
                      torch.triu(torch.ones((num_boxes, num_boxes), dtype=torch.bool, device=self.device), diagonal=1)]
        union_boxes = union_boxes.view(batch_size, -1, 4)

        # 取出前relation_num_class个并集框
        return union_boxes[:]

    def get_union_feature(self, memory_global, union_box):
        # 获取 batch_size 和 num_union_boxes
        batch_size, num_union_boxes, _ = union_box.shape
        # 将 input_features 转换为 (batch_size, channel, H, W) 的形状
        input_features = memory_global
        # 将 union_box 转换为 (batch_size * num_union_boxes, 4) 的形状，并复制 input_features
        union_box = union_box.view(batch_size * num_union_boxes, 4)
        features = input_features.repeat(1, num_union_boxes, 1, 1)
        features = features.view(batch_size * num_union_boxes, *input_features.shape[1:])

        # 对每个并集框进行 RoIAlign 操作，得到对应的特征向量
        union_features = ops.roi_align(features, [union_box], output_size=(7, 7), spatial_scale=1.0)

        # 将 union_features 转换为 (batch_size, num_union_boxes, out_channel, 7, 7) 的形状
        union_features = union_features.view(batch_size, num_union_boxes, -1, 7, 7)

        # 将 union_features 进行均值池化，得到每个并集框的特征向量
        union_features = F.adaptive_avg_pool2d(union_features, (1, 1))
        union_features = union_features.view(batch_size, num_union_boxes, -1)
        # 将 union_features 按照最后一个维度进行全连接，得到最终的输出
        union_features = nn.Linear(self.entity_embed_dim, self.relation_embed_dim, device=self.device)(union_features)

        return union_features

    def query_tgt_initialization(self, union_box, union_feature):
        box_feature = relu(torch.matmul(union_box, self.w_q))
        return union_feature.reshape(box_feature.shape) + box_feature

    def graph_assembling(self, semantic_predictions,
                         image_sizes,
                         entity_hidden_states,
                         relation_hidden_states_subject,
                         relation_hidden_states_object):

        sub_idxing_rule, obj_idxing_rule = self.entities_indexing_heads_rule(
            semantic_predictions, image_sizes
        )
        if self.indexing_module_type in ["rule_base", "pred_att", 'rel_vec']:

            if self.indexing_module_type in ["rule_base", 'rel_vec']:
                sub_idxing, obj_idxing = sub_idxing_rule, obj_idxing_rule
            elif self.indexing_module_type == "pred_att":
                sub_idxing, obj_idxing = self.entities_indexing_heads(semantic_predictions)

        elif self.indexing_module_type == "feat_att":
            sub_idxing = self.entities_indexing_heads["sub"](
                entity_hidden_states, relation_hidden_states_subject
            )
            obj_idxing = self.entities_indexing_heads["obj"](
                entity_hidden_states, relation_hidden_states_object
            )
        return sub_idxing, obj_idxing, sub_idxing_rule, obj_idxing_rule

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(ent_det_res, rel_det_res, batched_inputs, image_sizes):
        """
        dump every attributes of prediction result into the Relationships structures
        """
        # note: private function; subject to changes

        processed_results = []
        # for results_per_image, input_per_image, image_size in zip(
        for det_res_per_image, rel_res_per_img, _, image_size in zip(
                ent_det_res, rel_det_res, batched_inputs, image_sizes
        ):

            if rel_res_per_img.get("rel_branch_box") is not None:
                rel_inst = Instances(image_size)
                rel_inst.pred_boxes = Boxes(rel_res_per_img["rel_branch_box"].float())
                rel_inst.scores = rel_res_per_img["rel_branch_score"].float()
                rel_inst.pred_classes = rel_res_per_img["rel_branch_label"]
                rel_inst.pred_score_dist = rel_res_per_img["rel_branch_dist"]
                det_result = rel_inst
            else:
                det_result = Instances(image_size)
                det_result.pred_boxes = Boxes(det_res_per_image["boxes"].float())
                det_result.scores = det_res_per_image["scores"].float()
                det_result.pred_classes = det_res_per_image["labels"]
                det_result.pred_score_dist = det_res_per_image["prob"]

            pred_rel = Relationships(
                instances=det_result,
                rel_pair_tensor=rel_res_per_img["rel_trp"][:, :2],
                pred_rel_classs=rel_res_per_img["rel_pred_label"],  # start from 1
                pred_rel_scores=rel_res_per_img["rel_score"],
                pred_rel_trp_scores=rel_res_per_img["rel_trp_score"],
                pred_rel_dist=rel_res_per_img["pred_prob_dist"],
                pred_init_prop_idx=rel_res_per_img["init_prop_indx"],
            )

            if rel_res_per_img.get("rel_vec") is not None:
                pred_rel.rel_vec = rel_res_per_img.get("rel_vec")

            if rel_res_per_img.get("pred_rel_confidence") is not None:
                pred_rel.pred_rel_confidence = rel_res_per_img[
                    "pred_rel_confidence"
                ].unsqueeze(-1)

            if rel_res_per_img.get("selected_mask") is not None:
                for k, v in rel_res_per_img.get("selected_mask").items():
                    pred_rel.__setattr__(k, v)

            if rel_res_per_img.get("pred_rel_ent_obj_box") is not None:
                for role in ['sub', 'obj']:
                    for k_name in [f'pred_rel_ent_{role}_box',
                                   f'pred_rel_ent_{role}_label',
                                   f'pred_rel_ent_{role}_score']:
                        pred_rel.__setattr__(k_name, rel_res_per_img.get(k_name))

                    if rel_res_per_img.get(f"dyna_anchor_{role}_box") is not None:
                        pred_rel.__setattr__(f"dyna_anchor_{role}_box",
                                             rel_res_per_img[f"dyna_anchor_{role}_box"])

            processed_results.append({"instances": det_result, "relationships": pred_rel})

        return processed_results

    # input pre-process
    def convert_anno_format(self, batched_inputs):
        targets = []
        for bi in batched_inputs:
            target = {}
            h, w = bi["image"].shape[-2:]

            boxes_xyxy = bi["instances"].gt_boxes.tensor / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes = box_xyxy_to_cxcywh(boxes_xyxy).to(self.device)

            # cxcywh 0-1, w-h
            target["boxes"] = boxes.to(self.device)
            target["boxes_init"] = box_xyxy_to_cxcywh(bi["instances"].gt_boxes.tensor).to(
                self.device)  # cxcy 0-1

            # xyxy 0-1, w-h
            target["boxes_xyxy_init"] = bi["instances"].gt_boxes.tensor.to(self.device)
            target["boxes_xyxy"] = boxes_xyxy.to(self.device)

            target["area"] = bi["instances"].gt_boxes.area().to(self.device)
            target["labels"] = bi["instances"].gt_classes.to(self.device)
            if hasattr(bi["instances"], "gt_masks"):
                target["masks"] = bi["instances"].gt_masks
            target["iscrowd"] = torch.zeros_like(target["labels"], device=self.device)
            target["orig_size"] = torch.tensor(
                [bi["height"], bi["width"]], device=self.device
            )
            target["size"] = torch.tensor([h, w], device=self.device)
            target["image_id"] = torch.tensor(bi["image_id"], device=self.device)

            ####  relationship parts
            target["relationships"] = bi["relationships"].to(self.device)
            target["rel_labels"] = bi["relationships"].rel_label
            target["rel_label_no_mask"] = bi["relationships"].rel_label_no_mask

            rel_pair_tensor = bi["relationships"].rel_pair_tensor
            target["gt_rel_pair_tensor"] = rel_pair_tensor
            target["rel_vector"] = torch.cat(
                (boxes[rel_pair_tensor[:, 0], :2], boxes[rel_pair_tensor[:, 1], :2]),
                dim=1,
            ).to(
                self.device
            )  # Kx2 + K x2 => K x 4

            targets.append(target)

        return targets

    def _reset_parameters(self):
        num_layers = self.cfg.MODEL.RELATION_TRANSFORMER.ENTITY_DECODER.NUM_LAYERS

        def initialize_ent_pred(class_embed, bbox_embed):
            class_embed = nn.ModuleList([class_embed for _ in range(num_layers)])
            bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_layers)])

            return class_embed, bbox_embed

        (self.obj_class_embed, self.obj_bbox_embed) = initialize_ent_pred(self.obj_class_embed, self.obj_bbox_embed)
        (self.sub_class_embed, self.sub_bbox_embed) = initialize_ent_pred(self.sub_class_embed, self.sub_bbox_embed)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
