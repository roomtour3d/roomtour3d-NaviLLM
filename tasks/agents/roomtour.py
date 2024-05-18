from .llava import LLaVAAgent
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from models.ops import pad_tensors_wgrad
from collections import defaultdict
from contextlib import nullcontext
from models.graph_utils import GraphMap
from typing import List
from models.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts

class RoomTourAgent(LLaVAAgent):
    name = "roomtour"

    # def __init__(self, args=None, shortest_distances=None, shortest_paths=None):
    #     self.args = args
    #     self.shortest_paths = shortest_paths
    #     self.shortest_distances = shortest_distances
    #     # buffer
    #     self.scanvp_cands = {}

    def update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']


    def get_prompt(self, task, *args, **kwargs):
        if task == 'video_desc':
            return self.get_object_trajectory_prompt(*args, **kwargs)
        elif task == 'video_desc_roomobj':
            return self.get_room_object_trajectory_prompt(*args, **kwargs)
        elif task == 'video_desc_roomobj_spatial':
            return self.get_room_object_trajectory_spatial_aug_prompt(*args, **kwargs)
        elif task == 'video_desc_roomloc':
            return self.get_room_location_trajectory_prompt(*args, **kwargs)
        else:
            raise NotImplementedError(task)

    def get_room_location_trajectory_prompt(self, ques, cand_num):
        obs_text = ' '.join(["({}) <cand>".format(i) for i in range(cand_num)])
        prompt = "Please describe the path of the camera movement takes by identifying the rooms that are visited in order.\n" + \
            "The following is the Observation, which includes multiple frames from an egocentric camera touring a house. Please list the rooms visited in each observased image.\n" + \
            "### Observation: {} \n".format(obs_text) + \
            "### Question: {}\n".format(ques) + \
            "### Answer: "
        return prompt

    def get_object_trajectory_prompt(self, ques, cand_num):
        obs_text = ' '.join(["({}) <cand>".format(i) for i in range(cand_num)])
        prompt = "Please describe the path of the camera movement takes by identifying the key objects that enter and leave the view.\n" + \
            "The following is the Observation, which includes multiple frames from an egocentric camera touring a house.\n" + \
            "### Observation: {} \n".format(obs_text) + \
            "### Question: {}\n".format(ques) + \
            "### Answer: "
        return prompt

    def get_room_object_trajectory_prompt(self, ques, cand_num):
        obs_text = ' '.join(["({}) <cand>".format(i) for i in range(cand_num)])
        prompt = "Please describe the camera's movement trajectory through the visited rooms, and also detailing the key objects that appear and disappear from the frame.\n" + \
            "The following is the Observation, which includes multiple frames from an egocentric camera touring a house.\n" + \
            "### Observation: {} \n".format(obs_text) + \
            "### Question: {}\n".format(ques) + \
            "### Answer: "
        return prompt

    def get_room_object_trajectory_spatial_aug_prompt(self, ques, cand_num):
        obs_text = ' '.join(["({}) <cand>".format(i) for i in range(cand_num)])
        prompt = "Please describe the camera's movement trajectory through the visited rooms, detailing and imagining the key objects that exist in the located rooms.\n" + \
            "The following is the Observation, which includes multiple frames from an egocentric camera touring a house. The frames are partially cropped on one side. Please imagine the content has been cropped, considering common household furnishings and recurring objects in the frames.\n" + \
            "### Observation: {} \n".format(obs_text) + \
            "### Question: {}\n".format(ques) + \
            "### Answer: "
        return prompt

    def train(
        self,
        name,
        batch,
        args,
        config,
        model,
        criterion=None,
        dataset=None,
        step=0,
        entropy_metric=None,
        instr_pred_metric=None,
        **kwargs
    ):
        assert name in ["RoomTour", "ScanQA", "LLaVA"], 'The task name must be in [RoomTour, ScanQA, LLaVA]'
        dataset_cfg = config.Pretrain if args.stage=='pretrain' else config.Multi
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)
        # construct prompt
        prompts = []
        batch_size = len(batch["question"])
        # update prompts
        batch["prompts"] = self.prepare_prompts(batch)

        # forward the model
        lm_loss = model("3dqa", batch).loss
        lm_loss *= loss_coef / args.gradient_accumulation_step
        lm_loss.backward()

        return lm_loss * args.gradient_accumulation_step


    def prepare_prompts(self, batch):
        prompts = []
        for bn in range(len(batch["question"])):
            prompts.append(
                self.get_prompt(
                    batch['data_type'][bn],
                    ques = batch["question"][bn],
                    cand_num = batch["features"][bn].shape[0]
                )
            )
        return prompts