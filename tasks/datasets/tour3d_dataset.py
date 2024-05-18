import copy
from pathlib import Path
import torch
import math
from collections import defaultdict

try:
    from .base_dataset import BaseDataset
except:
    from tasks.datasets.base_dataset import BaseDataset
import numpy as np
import networkx as nx
import json
try:
    from .room3d_envs import (
        EnvBatch, angle_feature,
        get_all_point_angle_feature, load_nav_graphs, construct_fake_simulator, VideoSim
    )
except:
    from tasks.datasets.room3d_envs import (
        EnvBatch, angle_feature,
        get_all_point_angle_feature, load_nav_graphs, construct_fake_simulator, VideoSim
    )


def get_anno_file_path(data_dir, dataset_path, filename):
    if dataset_path.startswith('/'):
        return Path(dataset_path) / filename
    return Path(data_dir) / dataset_path / filename


class Tour3DDataset(BaseDataset):
    name = 'tour3d'

    def __init__(
            self,
            args,
            config,
            training=False,
            logger=None,
            source=None,
    ):
        super().__init__()
        self.config = config
        self.angle_feat_size = self.config.angle_feat_size
        self.logger = logger
        self.training = training
        self.debug = args.debug
        self.source = source

        if self.training:
            self.split = "train"
            self.max_objects = self.config.max_objects
            self.multi_endpoints = True
        else:
            self.split = args.validation_split
            self.max_objects = None
            self.multi_endpoints = False

        self.batch_size = args.batch_size
        self.seed = args.seed
        self.feat_db = None
        self.obj_feat_db = None
        self.no_loc_fts = args.no_loc_fts
        # connectivity graph
        # self.connectivity_dir = str(args.data_dir/'connectivity')

        # load mp3d dataset
        msg = self._load_data(config, args.data_dir)
        self.buffered_state_dict = {}

        # simulator
        self.sim = VideoSim('/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/object_progression_clip_gpt4_trajectory_wroom_p1/')

        # angle features
        # self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size, self.connectivity_dir)
        if self.no_loc_fts:
            self.angle_feature = np.array([0, 0, 0, 0] * (self.angle_feat_size // 4),
                                    dtype=np.float32)
        else:
            self.angle_feature = angle_feature(0., 0., self.angle_feat_size)

        # navigation graph
        # no graph currently avaiable for web videos
        # self._load_nav_graphs()

        if logger is not None:
            logger.info('[INFO] %s loaded with %d instructions, using splits: %s' % (
                self.__class__.__name__, len(self.alldata), self.split))
            logger.info(msg)
        del self.data

    def init_feat_db(self, feat_db, obj_feat_db=None):
        self.feat_db = feat_db
        self.obj_feat_db = obj_feat_db

    def load_data(self, anno_file, max_instr_len=200, debug=False):
        """
        :param anno_file:
        :param max_instr_len:
        :param debug:
        :return:
        """
        with open(str(anno_file), "r") as f:
            data = json.load(f)
            
        new_data = []
        sample_index = 0

        for i, item in enumerate(data):
            if len(item['path']) < 3:
                continue
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['raw_idx'] = i
                new_item['sample_idx'] = sample_index
                new_item['instr_id'] = 'tour3d_{}_{}'.format(item['path_id'], j)

                new_item['instruction'] = instr
                del new_item['instructions']

                if 'instr_encodings' in new_item:
                    new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                    del new_item['instr_encodings']

                if 'new_instructions' in new_item and len(eval(item['new_instructions'])) > j:
                    new_item['fg_instruction'] = eval(item['new_instructions'])[j]
                    new_item['fg_instruction'] = [' '.join(instr) for instr in new_item['fg_instruction']]
                    del new_item['new_instructions']
                    new_item['fg_view'] = item['chunk_view'][j]
                    fg_view = []
                    for idx, index in enumerate(new_item['fg_view']):
                        index_num = index[1] - index[0]
                        fg_view += [idx] * index_num
                    new_item['fg_view'] = fg_view
                    del new_item['chunk_view']

                new_item['data_type'] = 'tour3d'
                new_data.append(new_item)
                sample_index += 1

        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['videoId'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }
        return new_data, gt_trajs


    def _load_data(self, config, data_dir):
        self.data = dict()
        self.alldata = []
        msg = ""
        if self.source == "Tour3D":
            anno_file = get_anno_file_path(data_dir, config.Tour3D.DIR, config.Tour3D.SPLIT[self.split])
            self.data['tour3d'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} Tour3D samples'.format(len(self.data['tour3d']))
        else:
            print("Dataset Source: {}".format(self.source))
            raise NotImplementedError

        for key, value in self.data.items():
            self.alldata += value

        msg += '\n- Dataset: load {} split: {} samples in total'.format(self.split, len(self.alldata))
        self.scans = set([x['videoId'] for x in self.alldata])
        msg += '\n- Dataset: load {} split: {} scans in total'.format(self.split, len(self.scans))

        return msg

    # def _load_nav_graphs(self):
    #     """
    #     load graph from self.scan,
    #     Store the graph {scan_id: graph} in self.graphs
    #     Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
    #     Store the distances in self.distances. (Structure see above)
    #     Load connectivity graph for each scan, useful for reasoning about shortest paths
    #     :return: None
    #     """
    #     # print('Loading navigation graphs for %d scans' % len(self.scans))
    #     self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
    #     self.shortest_paths = {}
    #     for scan, G in self.graphs.items():  # compute all shortest paths
    #         self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    #     self.shortest_distances = {}
    #     for scan, G in self.graphs.items():  # compute all shortest paths
    #         self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, index):
        item = copy.deepcopy(self.alldata[index])
        item = self.preprocess_item(item)
        data_type = item['data_type']
        scan = item['videoId']
        instr_id = item['instr_id']

        scanIds = [scan]
        viewpointIds = [item['path'][0]]
        headings = [item['heading']]

        env = EnvBatch(connectivity_dir='/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/object_progression_clip_gpt4_trajectory_wroom_p1/', batch_size=1)
        env.newEpisodes(scanIds, viewpointIds, headings)
        observations = self.get_obs(items=[item], env=env, data_type=data_type)[0]

        data_dict = {
            'sample_idx': index,
            'instr_id': instr_id,
            'observations': observations,
            'env': env,
            'item': item,
            'data_type': data_type,
        }

        return data_dict
    
    def preprocess_item(self, item):
        return item

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['NotImplemented']:
                    ret[key] = torch.stack(val, 0)
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
    
    def get_object_info(self, item):
        raise NotImplementedError

    def get_obs(self, items, env, data_type=None):
        obs = []

        for i, (feature, state) in enumerate(env.getStates()):
            item = items[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = self.feat_db.get_image_feature(state.videoId, state.location.frameId)

            features = [feature]
            angle_feats = [self.angle_feature]
            for loc in state.get_neighbours():
                features += [self.feat_db.get_image_feature(loc.videoId, loc.frameId)]
                #TODO: get the angle feats for each neigh frame
                angle_feats += [angle_feature(0., 0., self.angle_feat_size)]
            # Full features
            candidate = self.make_candidate(features, state.videoId, state.location.frameId, state.viewIndex)
            
            # #TODO: get features for the candidate frames, maybe precalculate the angle to the current frames
            # angle_features = get_angle_features()
            features = [np.concatenate((feature, afeature), -1) for feature, afeature in zip(features, angle_feats)]
            # cand_features = [cand['feature'] for cand in candidate]
            # features.extend(cand_features)
            features = np.stack(features, 0)
            # # [visual_feature, angle_feature] for views
            # feature = candidate['feature']

            ob = {
                'instr_id': item['instr_id'],
                'videoId': state.videoId,
                'scan': state.videoId,
                'viewpoint': state.location.frameId,
                'viewpointId': f"{state.videoId}%{state.location.frameId}",
                'viewIndex': state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': features,
                'candidate': candidate,
                # 'navigableLocations': state.navigableLocations,
                'instruction': item['instruction'],
                # 'instr_encoding': item['instr_encoding'],
                'gt_path': item['path'],
                'path_id': item['path_id'],
            }
            if 'fg_instruction' in item:
                ob.update({
                    'fg_instruction': item['fg_instruction'],
                    'fg_view': item['fg_view'],
                })
            if self.obj_feat_db is not None:
                obj_info = self.get_object_info(item, state)
                ob.update(obj_info)
                ob['distance'] = 0
            else:
                # RL reward. The negative distance between the state and the final state
                # There are multiple gt end viewpoints on REVERIE. 
                if False: # ob['instr_id'] in self.gt_trajs:
                    ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
                else:
                    ob['distance'] = 0
            obs.append(ob)
        return obs

    def make_candidate(self, feature, videoId, frameId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = 0
        base_elevation = 0

        adj_dict = {}
        long_id = "%s_%s" % (videoId, frameId)

        if long_id not in self.buffered_state_dict:
            self.sim.newEpisode([videoId], [frameId], [0], [0])
            state = self.sim.getState()[0]

            for ix in range(1+len(state.get_neighbours())):
                # ix = 0
                if ix != 0:
                    self.sim.getNextObs(1)
                    state = self.sim.getState()[0]
                assert state.viewIndex == ix
                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                if state.current_nav_type_to_base_view == 0:
                    continue
                visual_feat = feature[ix]
                # for j, loc in enumerate(state.neighbours):
                for j, loc in enumerate(state.get_navigable_neighbours()):
                    # Heading and elevation for the viewpoint center
                    distance = state.distance(loc)

                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    if self.no_loc_fts:
                        angle_feat = np.array([0, 0, 0, 0] * (self.angle_feat_size // 4),
                                                dtype=np.float32)
                    else:
                        angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)

                    if (loc.longFrameId not in adj_dict or
                            distance < adj_dict[loc.longFrameId]['distance']):
                        adj_dict[loc.longFrameId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': videoId,
                            'viewpointId': loc.longFrameId,  # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }

            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'position']}
                    # ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                # visual_feat = feature
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                if self.no_loc_fts:
                    angle_feat = np.array([0, 0, 0, 0] * (self.angle_feat_size // 4),
                                            dtype=np.float32)
                else:
                    angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

if __name__=='__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data', help="dataset root path")
    parser.add_argument('--cfg_file', type=str, default='/mnt/bn/kinetics-lp-maliva-v6/playground_projects/NaviLLM/configs/pretrain.yaml', help='dataset configs')
    parser.add_argument('--pretrained_model_name_or_path', default='/mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/', type=str, help="path to tokenizer")

    # local fusion
    parser.add_argument('--off_batch_task', action='store_true', default=False, help="whether all process is training same task")
    parser.add_argument('--debug', action="store_true", help="debug mode")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to ckpt to resume from")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--feat_dropout", type=float, default=0.4)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_steps_per_epoch", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_step", type=int, default=2)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument("--workers", type=int, default=0)

    # distributed training args
    parser.add_argument('--gpu', type=int, default=0, help='current gpu id, local rank')
    parser.add_argument('--world_size', type=int, default=0, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    # Save checkpoints
    parser.add_argument('--output_dir', type=str, default='output/debug', help="output logs and ckpts")
    parser.add_argument("--max_saved_checkpoints", type=int, default=0)
    parser.add_argument("--save_ckpt_per_epochs", type=int, default=10)
    parser.add_argument("--save_latest_states", action='store_true')
    parser.add_argument("--save_pred_results", action="store_true")
    parser.add_argument("--save_detail_results", action="store_true")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument("--stage", type=str, default="pretrain", choices=["pretrain", "multi"])
    parser.add_argument('--ignoreid', default=-100, type=int, help="criterion: ignore label")
    parser.add_argument('--enable_og', action='store_true', default=False, help="object grounding task")
    parser.add_argument("--enable_summarize", action="store_true", help="perform EQA or generate instructions")
    parser.add_argument("--enable_fgr2r", action="store_true", help="perform fgr2r for R2R")
    parser.add_argument("--gen_loss_coef", type=float, default=1.)
    parser.add_argument("--obj_loss_coef", type=float, default=1.)
    parser.add_argument("--teacher_forcing_coef", type=float, default=1.)
    parser.add_argument("--fuse_obj", action="store_true", help="whether fuse object features for REVERIE and SOON")
    parser.add_argument("--use_lora", action="store_true", help="whether using lora")
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha, usually starting from two times of rank")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    parser.add_argument('--lora_target', type=str, default=None, nargs='+')
    parser.add_argument("--freeze_llama", action="store_true", help="whether freezing llama")
    parser.add_argument("--tune_token_emb", action="store_true", help="whether tuning token embedding")


    # room tour
    parser.add_argument("--no_loc_fts", action="store_true", help="no loc fts during nav")

    # datasets
    parser.add_argument("--multi_endpoints", type=int, default=1)
    parser.add_argument("--path_type", type=str, default="trusted_path", choices=["planner_path", "trusted_path"])

    # evaluation
    parser.add_argument('--test_datasets', type=str, default=None, nargs='+')
    parser.add_argument('--validation_split', type=str, default="val_unseen", help="validation split: val_seen, val_unseen, test")
    parser.add_argument("--do_sample", action="store_true", help="do_sample in evaluation")
    parser.add_argument("--temperature", type=float, default=1.)


    # others
    parser.add_argument(
        "--max_datapoints",
        default=None,
        type=int,
        help="The number of datapoints used for debug."
    )
    
    args = parser.parse_args()
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))

    args.data_dir = Path(args.data_dir).resolve()

    # off-line image features from Matterport3D
    args.image_feat_size = global_cfg.Feature.image_feat_size
    args.obj_feat_size = global_cfg.Feature.obj_feat_size

    ############# Configurations ###############
    args.angle_feat_size = global_cfg.Feature.angle_feat_size
    args.enc_full_graph = global_cfg.Model.enc_full_graph
    args.expert_policy = global_cfg.Model.expert_policy
    args.num_pano_layers = global_cfg.Model.num_pano_layers

    dataset_cfg = global_cfg.Pretrain
    dataset_cfg = copy.deepcopy(global_cfg.Dataset)
    dataset_cfg.update(
        global_cfg.Pretrain
    )
    dataset_cfg.update(global_cfg.Feature)
    dataset = Tour3DDataset(args, dataset_cfg, training=True, source='Tour3D')