#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging

import numpy as np

import torch.utils.data

from cvpods.utils import comm, seed_all_rng

from .detection_utils import check_sample_valid
from .registry import DATASETS, PATH_ROUTES, SAMPLERS, TRANSFORMS

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_dataset",
    "build_detection_train_loader",
    "build_detection_test_loader",
    "build_transform_gen",
]

logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')

def build_transform_gen(pipelines):
    """
    Create a list of :class:`TransformGen` from config.

    Transform list is a list of tuple which includes Transform name and parameters.
    Args:
        pipelines: cfg.INPUT.TRAIN_PIPELINES and cfg.INPUT.TEST_PIPELINES are used here

    Returns:
        list[TransformGen]: a list of several TransformGen.
    """

    def build(pipeline):
        tfm_gens = []

        for (aug, args) in pipeline:
            if "List" in aug:
                assert "transforms" in args, "List Transforms must contain a `transforms` key"
                sub_pipelines = args["transforms"]
                args["transforms"] = build_transform_gen(sub_pipelines)
                tfm = TRANSFORMS.get(aug)(**args)
            else:
                if aug == "ResizeShortestEdge":
                    check_sample_valid(args)
                if aug.startswith("Torch_"):
                    tfm = TRANSFORMS.get("TorchTransformGen")(args)
                else:
                    tfm = TRANSFORMS.get(aug)(**args)
            tfm_gens.append(tfm)

        return tfm_gens

    if isinstance(pipelines, dict):
        tfm_gens_dict = {}
        for key, tfms in pipelines.items():
            tfm_gens_dict[key] = build(tfms)
        return tfm_gens_dict
    else:
        return build(pipelines)


def _build_single_dataset(config, dataset_name, transforms=[], is_train=True):
    """
    Build a single dataset according to dataset_name.

    Args:
        config (BaseConfig): config.
        dataset_name (str): dataset_name should be of 'dataset_xxx_xxx' format,
            so that corresponding dataset can be acquired from the first token in this argument.
        transforms (List[TransformGen]): list of transforms configured in config file.
        is_train (bool): whether is in training mode or not.
    """
    print(dataset_name)
    dataset_type = dataset_name.split("_")[0].upper()
    print(dataset_type)
    assert dataset_type in PATH_ROUTES, "{} not found in PATH_ROUTES".format(dataset_type)
    name = PATH_ROUTES.get(dataset_type)["dataset_type"]
    dataset = DATASETS.get(name)(config, dataset_name, transforms=transforms, is_train=is_train)
    return dataset


def build_dataset(config, dataset_names, transforms=[], is_train=True):
    """
    dataset_names: List[str], in which elemements must be in format of "dataset_task_version"
    """
    datasets = [
        _build_single_dataset(config, dataset_name, transforms=transforms, is_train=is_train)
        for dataset_name in dataset_names
    ]
    custom_type, args = config.DATASETS.CUSTOM_TYPE
    # wrap all datasets, Dataset concat is the default behaviour
    dataset = DATASETS.get(custom_type)(datasets, **args)
    return dataset


def build_detection_train_loader(cfg):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:

       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    # For simulate large batch training
    num_devices = comm.get_world_size()
    rank = comm.get_rank()

    # use subdivision batchsize
    images_per_minibatch = cfg.SOLVER.IMS_PER_DEVICE // cfg.SOLVER.BATCH_SUBDIVISIONS

    logger = logging.getLogger(__name__)

    transform_gens = build_transform_gen(cfg.INPUT.AUG.TRAIN_PIPELINES)
    logger.info(f"TransformGens used: {transform_gens} in training")


    dataset = build_dataset(
        cfg, cfg.DATASETS.TRAIN, transforms=transform_gens, is_train=True
    )

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info("Using training sampler {}".format(sampler_name))

    assert sampler_name in SAMPLERS, "{} not found in SAMPLERS".format(sampler_name)
    if sampler_name == "TrainingSampler":
        sampler = SAMPLERS.get(sampler_name)(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = SAMPLERS.get(sampler_name)(
            dataset, cfg.DATALOADER.REPEAT_THRESHOLD)
    elif sampler_name == "DistributedGroupSampler":
        sampler = SAMPLERS.get(sampler_name)(
            dataset, images_per_minibatch, num_devices, rank)

    if cfg.DATALOADER.ENABLE_INF_SAMPLER:
        sampler = SAMPLERS.get('Infinite')(sampler)
        logger.info("Wrap sampler with infinite warpper...")


    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_minibatch,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        pin_memory=True,
    )
    print(torch.multiprocessing.get_sharing_strategy())

    return data_loader


def build_detection_test_loader(cfg):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a cvpods CfgNode

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    transform_gens = build_transform_gen(cfg.INPUT.AUG.TEST_PIPELINES)
    logger = logging.getLogger(__name__)
    logger.info(f"TransformGens used: {transform_gens} in testing")
    dataset = build_dataset(cfg,
                            cfg.DATASETS.TEST,
                            transforms=transform_gens,
                            is_train=False)
    sampler = SAMPLERS.get("InferenceSampler")(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        pin_memory=True,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2**31) + worker_id)
