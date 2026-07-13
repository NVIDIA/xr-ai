# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process semantic-SLAM engine.

This module wraps the per-frame pipeline implemented in
``slam.services.inference_pipeline.inference_consumer`` behind a small, clean,
importable API. It performs the same work -- segmentation, point-cloud
construction, CLIP feature extraction, and incremental object merging -- but
drops everything that only exists for the gRPC/queue server deployment
(visualization queue, gRPC calls, timing dictionaries).

Heavy / GPU-only dependencies (the segmentation, CLIP, detection and
captioning models, ``cv2``, the dataset loaders) are imported lazily inside
``__init__``/``push`` so that the pure helpers ``_rank_objects`` and
``_normalize_pose`` -- and therefore the CPU-only smoke test -- can import this
module without CUDA, open_clip or groundingdino installed.

Example
-------
    from semantic_slam.engine import SemanticSLAM

    slam = SemanticSLAM(dataset_type="replica", scene_name="room0")
    slam.push(rgb, depth, pose)              # numpy arrays
    hits = slam.query("a brown chair", top_k=5)
"""

import numpy as np
import torch

# Background classes are fused into the map as single objects, mirroring
# slam.services.inference_pipeline.BG_CLASSES.
BG_CLASSES = ["wall", "floor", "ceiling"]


def _normalize_pose(pose, frame_number):
    """Normalize a camera pose into the flat list the pipeline consumes.

    The replica producer in ``server/main.py`` feeds the pipeline a pose of the
    form ``[frame_number] + [16 floats]`` where the 16 floats are a row-major
    (C-order) 4x4 transform. This helper accepts either a 4x4 array-like or a
    flat length-16 sequence and returns that same ``[frame_number, *16]`` list.

    Parameters
    ----------
    pose : array-like
        Either a 4x4 matrix (ndarray or nested list) or a flat length-16
        sequence in row-major order.
    frame_number : int
        Frame index prepended to the flattened pose.

    Returns
    -------
    list
        ``[frame_number, f0, f1, ..., f15]`` with 17 elements total.

    Raises
    ------
    ValueError
        If ``pose`` is neither a 4x4 matrix nor a flat length-16 sequence.
    """
    pose_arr = np.asarray(pose, dtype=np.float64)

    if pose_arr.shape == (4, 4):
        flat = pose_arr.reshape(-1)  # C-order / row-major, matching the producer
    elif pose_arr.ndim == 1 and pose_arr.size == 16:
        flat = pose_arr
    else:
        raise ValueError(
            "pose must be a 4x4 matrix or a flat length-16 sequence; "
            f"got shape {pose_arr.shape}"
        )

    return [frame_number] + [float(x) for x in flat]


def _rank_objects(objects, text_feat, top_k):
    """Rank map objects against a query feature and format the top hits.

    This is the pure, GPU-free core of :meth:`SemanticSLAM.query`. It is kept at
    module scope so it can be unit-tested directly. ``objects`` only needs to
    expose ``compute_similarities`` (duck-typed) and be indexable, yielding
    dicts with ``pcd`` (open3d PointCloud) and ``bbox`` (open3d
    OrientedBoundingBox) entries.

    Parameters
    ----------
    objects : MapObjectList-like
        The semantic map. Empty maps yield an empty result list.
    text_feat : torch.Tensor or np.ndarray
        An L2-normalized query feature of shape ``(D,)``.
    top_k : int
        Maximum number of hits to return.

    Returns
    -------
    list[dict]
        Hits sorted by descending cosine similarity. Each dict has keys
        ``score``, ``class_name``, ``centroid``, ``num_points``,
        ``bbox_center`` and ``bbox_extent``.
    """
    if len(objects) == 0:
        return []

    sims = objects.compute_similarities(text_feat)
    sims = sims.detach().float().cpu()

    k = min(top_k, len(objects))
    top_scores, top_indices = torch.topk(sims, k)

    hits = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        obj = objects[idx]

        # class_name / class_id are stored as single-element lists (see
        # slam.core.utils.gobs_to_detection_list_optimized).
        class_name = None
        raw_name = obj.get("class_name")
        if raw_name:
            class_name = raw_name[0] if isinstance(raw_name, (list, tuple)) else raw_name

        points = np.asarray(obj["pcd"].points)
        centroid = points.mean(axis=0) if len(points) else np.zeros(3)

        bbox = obj["bbox"]
        hits.append(
            {
                "score": float(score),
                "class_name": class_name,
                "centroid": [float(c) for c in centroid],
                "num_points": int(len(points)),
                "bbox_center": [float(c) for c in np.asarray(bbox.center)],
                "bbox_extent": [float(e) for e in np.asarray(bbox.extent)],
            }
        )

    # torch.topk already returns descending order, but sort defensively so the
    # contract holds regardless of backend.
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


class SemanticSLAM:
    """In-process semantic-SLAM map builder and query engine.

    Wraps the existing per-frame pipeline. Construct once, ``push`` RGB-D frames
    with poses to incrementally build a semantic 3D map, then ``query`` it with
    natural-language text. See the module docstring for a usage example.

    Notes
    -----
    All model construction and the heavy per-frame work happen on a GPU; this
    class is not exercised by the CPU-only smoke test. The pure helpers
    ``_normalize_pose`` and ``_rank_objects`` are tested instead.
    """

    def __init__(
        self,
        dataset_type="replica",
        scene_name=None,
        config=None,
        use_detector=False,
        device="cuda:0",
    ):
        """Build the models and per-frame state.

        Parameters
        ----------
        dataset_type : str
            One of ``"replica"``, ``"scannet"``, ``"ipad"``. Selects the
            dataset loader used to back-project depth into point clouds.
        scene_name : str or None
            Scene identifier passed through to the dataset loader (used by
            ScanNet for intrinsics lookup).
        config : config.settings.Config or None
            Optional config override. When ``None`` the global config from
            ``get_config()`` is used. The provided config is installed globally
            via ``set_config`` so the pipeline utilities pick it up.
        use_detector : bool
            When ``True`` the open-vocabulary detector + captioning models are
            built (matching ``inference_consumer``); otherwise only segmentation
            + CLIP run with a single ``"item"`` class.
        device : str
            Torch device for the models, e.g. ``"cuda:0"``. Overrides the
            per-model device in the config.
        """
        # First-launch setup: install the model stack (chamferdist, pytorch3d,
        # gradslam, segment-anything) + SAM weights into the uv venv if they are
        # not already present. No-op once set up. Disable with
        # SEMANTIC_SLAM_AUTO_SETUP=0.
        from semantic_slam.bootstrap import ensure_setup
        ensure_setup()

        # Lazy imports: these pull in GPU / open_clip / groundingdino and must
        # not be at module top, or _rank_objects/_normalize_pose become
        # unimportable on a CPU-only host.
        from config.settings import get_config, set_config
        from slam.models.segmentation import SegmentationModel
        from slam.models.clip import clipModel
        from slam.core.slam_classes import MapObjectList
        from slam.utils.mapping_utils import get_dataset, setup

        if config is None:
            config = get_config()
        # Honor the caller's device override across all models.
        config.model.set_all_devices(device)
        # Utilities read the global config, so install it.
        set_config(config)

        self.config = config
        self.device = device
        self.dataset_type = dataset_type
        self.scene_name = scene_name
        self.use_detector = use_detector

        # Model-construction flags, mirroring inference_consumer lines 90-98.
        batched_sam = config.model.segmentation.batched_sam
        trt_sam = config.model.segmentation.trt_sam
        batched_clip = config.model.clip.batched_clip
        trt_clip = config.model.clip.trt_clip
        precision = config.model.clip.precision
        batch_size = config.model.clip.batch_size
        sam_variant = config.model.segmentation.sam_variant
        test_depth_downsampling = config.model.mapping.test_depth_downsampling

        if not use_detector and sam_variant != "sam":
            raise ValueError("If use_detector is False, sam_variant must be 'sam'.")

        # Detector + captioning only when requested (inference_consumer 106-113).
        self.captioning_model = None
        self.detection_model = None
        if use_detector:
            from slam.models.captioning import captioning
            from slam.models.detection import detector

            self.captioning_model = captioning(
                class_set=config.model.captioning.class_set,
                device=config.model.captioning.device,
                add_bg_classes=config.model.captioning.add_bg_classes,
                accumu_classes=config.model.captioning.accumu_classes,
            )
            self.detection_model = detector(
                detector="dino",
                device=device,
                box_threshold=0.2,
                text_threshold=0.2,
                nms_threshold=0.5,
            )

        self.segmentation_model = SegmentationModel(
            device=device,
            sam_variant=sam_variant,
            batched_sam=batched_sam,
            trt_sam=trt_sam,
            useDetector=use_detector,
        )
        self.clip_model = clipModel(
            device=device,
            batched_clip=batched_clip,
            trt_clip=trt_clip,
            precision=precision,
            batch_size=batch_size,
            clip_model_name=config.model.clip.model_name,
            pretrained=config.model.clip.pretrained,
        )

        # Dataset + cfg setup, mirroring inference_consumer 131-141.
        self.cfg = setup(use_detector, dataset_type)
        self.dataset = get_dataset(
            datasetClass=dataset_type,
            config_dict=self.cfg.dataset_config,
            desired_height=self.cfg.image_height,
            desired_width=self.cfg.image_width,
            device="cpu",
            dtype=torch.float,
            scene_name=scene_name,
            test_depth_downsampling=test_depth_downsampling,
        )

        # Per-frame map state (inference_consumer 143-154).
        self.objects = MapObjectList()  # bare list subclass; takes no kwargs
        self._history_map = {}
        self._next_index = 0
        self._idx = 0
        self._frame_counter = 0
        if not self.cfg.skip_bg:
            self._bg_objects = {c: None for c in BG_CLASSES}
        else:
            self._bg_objects = None

    def push(self, rgb, depth, pose, frame_number=None):
        """Process one RGB-D frame and merge its detections into the map.

        Replicates the per-frame body of ``inference_consumer`` (lines 235-392),
        minus the visualization-queue, gRPC and timing-dict bookkeeping.

        Parameters
        ----------
        rgb : np.ndarray
            ``HxWx3`` uint8 image in RGB channel order.
        depth : np.ndarray
            ``HxW`` depth image.
        pose : array-like
            Camera pose as a 4x4 matrix or a flat length-16 row-major sequence.
        frame_number : int or None
            Frame index. Defaults to an internal monotonically increasing
            counter when ``None``.

        Returns
        -------
        int
            The number of objects in the map after merging this frame.
        """
        import cv2
        import supervision as sv

        from slam.utils.general_utils import to_tensor
        from slam.utils.mapping_utils import create_pcd_parallel
        from slam.core.utils import merge_obj2_into_obj1, filter_objects, merge_objects, denoise_selected_objects
        from slam.core.mapping import (
            compute_spatial_similarities,
            compute_visual_similarities,
            aggregate_similarities,
            merge_detections_to_objects,
        )
        from slam.utils.ious import compute_2d_box_contained_batch
        from PIL import Image

        # --- Validate inputs at the boundary -------------------------------
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3; got shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8; got dtype {rgb.dtype}")
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(f"depth must be HxW; got shape {depth.shape}")

        if frame_number is None:
            frame_number = self._frame_counter
        self._frame_counter += 1

        # Build the same inputs the replica producer feeds the pipeline.
        image_pil = Image.fromarray(rgb)
        image_cv2_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pose_array = _normalize_pose(pose, frame_number)
        image_rgb = rgb  # rgb already; cv2 round-trip would be a no-op.

        # --- Detection / segmentation (inference_consumer 239-277) ---------
        if self.use_detector:
            caption, text_prompt = self.captioning_model.gen_caption(image_pil)
            classes = self.captioning_model.classes
            detections = self.detection_model.get_detections(image_cv2_bgr, classes)
            mask, _, _ = self.segmentation_model.run_segmentation(image_rgb, detections)
            detections.mask = mask
        else:
            classes = ["item"]
            mask, xyxy, conf = self.segmentation_model.run_segmentation(image_rgb, None)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )

        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_feats": None,
            "text_feats": None,
        }
        if self.use_detector:
            results["tagging_caption"] = caption
            results["tagging_text_prompt"] = text_prompt

        # --- Point cloud (parallel) + CLIP features (inference_consumer 283-296)
        image_np = np.array(image_pil)
        output_receiver_list = []
        # create_pcd_parallel mutates a time_dict; pass a throwaway one.
        create_pcd_parallel(
            image_np,
            depth,
            pose_array,
            frame_number,
            self.dataset,
            self.cfg,
            classes,
            results,
            output_receiver_list,
            False,  # pipelined_mapping
            self.dataset_type,
            {},  # time_dict (discarded)
        )

        image_crops, image_feats, text_feats = self.clip_model.get_clip_features(
            image_pil, image_rgb, detections, classes
        )

        fg_detection_list, bg_detection_list, _idx_to_keep = (
            output_receiver_list[0],
            output_receiver_list[1],
            output_receiver_list[2],
        )

        # --- Attach CLIP features (inference_consumer 313-320) -------------
        for obj in fg_detection_list:
            obj["clip_ft"] = to_tensor(image_feats[obj["mask_idx"][0]])
            obj["text_ft"] = to_tensor(text_feats[obj["mask_idx"][0]])
        for obj in bg_detection_list:
            obj["clip_ft"] = to_tensor(image_feats[obj["mask_idx"][0]])
            obj["text_ft"] = to_tensor(text_feats[obj["mask_idx"][0]])

        # --- Background fusion (inference_consumer 322-330) ----------------
        if len(bg_detection_list) > 0 and self._bg_objects is not None:
            for detected_object in bg_detection_list:
                class_name = detected_object["class_name"][0]
                if self._bg_objects[class_name] is None:
                    self._bg_objects[class_name] = detected_object
                else:
                    self._bg_objects[class_name] = merge_obj2_into_obj1(
                        self.cfg,
                        self._bg_objects[class_name],
                        detected_object,
                        run_dbscan=False,
                    )

        if len(fg_detection_list) == 0:
            return len(self.objects)

        # --- Contain-number bookkeeping (inference_consumer 336-340) -------
        contain_numbers = None
        if self.cfg.use_contain_number:
            xyxy = fg_detection_list.get_stacked_values_torch("xyxy", 0)
            contain_numbers = compute_2d_box_contained_batch(xyxy, self.cfg.contain_area_thresh)
            for i in range(len(fg_detection_list)):
                fg_detection_list[i]["contain_number"] = [contain_numbers[i]]

        # --- First-frame fast path (inference_consumer 342-351) ------------
        if len(self.objects) == 0:
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])
                fg_detection_list[i]["history_idx"] = self._next_index
                self._history_map[self._next_index] = i
                self._next_index += 1
            return len(self.objects)

        # --- Similarity + merge (inference_consumer 353-374) ---------------
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
        agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)

        if self.cfg.use_contain_number:
            contain_numbers_objects = torch.Tensor([obj["contain_number"][0] for obj in self.objects])
            detection_contained = (contain_numbers > 0).unsqueeze(1)
            object_contained = (contain_numbers_objects > 0).unsqueeze(0)
            xor = detection_contained ^ object_contained
            agg_sim[xor] = agg_sim[xor] - self.cfg.contain_mismatch_penalty

        agg_sim[agg_sim < self.cfg.dataset_config.mapping.sim_threshold] = float("-inf")

        (
            self.objects,
            edited_objects_idx,
            new_obj_idx,
            self._history_map,
            self._next_index,
        ) = merge_detections_to_objects(
            self.cfg,
            fg_detection_list,
            self.objects,
            agg_sim,
            self._history_map,
            self._next_index,
        )

        # --- Periodic post-processing (inference_consumer 382-392) ---------
        if self.cfg.denoise_interval > 0 and (self._idx + 1) % self.cfg.denoise_interval == 0:
            self.objects = denoise_selected_objects(self.cfg, self.objects, edited_objects_idx)
        if self.cfg.filter_interval > 0 and (self._idx + 1) % self.cfg.filter_interval == 0:
            self.objects, _removed, self._history_map = filter_objects(self.cfg, self.objects, self._history_map)
        if self.cfg.merge_interval > 0 and (self._idx + 1) % self.cfg.merge_interval == 0:
            self.objects, _removed2, _edited2, self._history_map = merge_objects(self.cfg, self.objects, self._history_map)

        self._idx += 1
        return len(self.objects)

    def query(self, text, top_k=5):
        """Query the semantic map with natural-language text.

        Encodes ``text`` with CLIP, L2-normalizes it, computes cosine
        similarity against each object's stored ``clip_ft``, and returns the
        top hits.

        Parameters
        ----------
        text : str
            The query, e.g. ``"a brown chair"``.
        top_k : int
            Maximum number of hits to return.

        Returns
        -------
        list[dict]
            See :func:`_rank_objects` for the per-hit schema. Empty when the
            map has no objects.

        Raises
        ------
        NotImplementedError
            If the CLIP model was built with the TensorRT path, which does not
            expose a plain text encoder here.
        """
        if len(self.objects) == 0:
            return []

        if getattr(self.clip_model, "trt_clip", False):
            raise NotImplementedError("Text query is not supported on the TRT CLIP path.")

        # clip_model is the clipModel wrapper; .clip_model is the inner open_clip
        # model, .clip_tokenizer is its tokenizer. The double attribute is
        # intentional, not a typo.
        tokens = self.clip_model.clip_tokenizer([text])
        with torch.no_grad():
            text_feat = self.clip_model.clip_model.encode_text(tokens.to(self.device))
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  # L2-normalize

        text_feat = text_feat.squeeze(0).float().cpu()
        return _rank_objects(self.objects, text_feat, top_k)

    def save_map(self, path):
        """Serialize the current semantic map to ``path``.

        Pickles ``self.objects.to_serializable()``. When ``path`` ends in
        ``.gz`` the pickle is gzip-compressed.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        import gzip
        import pickle

        data = self.objects.to_serializable()
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "wb") as f:
            pickle.dump(data, f)

    def load_map(self, path):
        """Load a semantic map previously written by :meth:`save_map`.

        Replaces ``self.objects`` with a fresh map loaded from ``path``. Gzip is
        detected from the ``.gz`` suffix. Per-frame bookkeeping
        (``_history_map``, indices) is reset since it cannot be recovered from
        the serialized form.

        Parameters
        ----------
        path : str
            Source file path.
        """
        import gzip
        import pickle

        from slam.core.slam_classes import MapObjectList

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            data = pickle.load(f)

        objects = MapObjectList(device=self.config.model.device)
        objects.load_serializable(data)
        self.objects = objects
        self._history_map = {}
        self._next_index = 0
        self._idx = 0

    def reset(self):
        """Clear the map and all per-frame state, ready to start a new scene."""
        self.objects.clear()
        self._history_map = {}
        self._next_index = 0
        self._idx = 0
        self._frame_counter = 0
        if not self.cfg.skip_bg:
            self._bg_objects = {c: None for c in BG_CLASSES}
        else:
            self._bg_objects = None
