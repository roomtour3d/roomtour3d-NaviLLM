from pickle import NONE
import numpy as np
import math
import torch
import networkx as nx
import json
import os
import glob
import math
import pickle as pkl
import msgpack
import msgpack_numpy
from dataclasses import dataclass

msgpack_numpy.patch()


# def new_simulator(connectivity_dir):
#     # Simulator image parameters
#     WIDTH = 640
#     HEIGHT = 480
#     VFOV = 60

#     sim = MatterSim.Simulator()
#     sim.setNavGraphPath(connectivity_dir)
#     sim.setRenderingEnabled(False)
#     sim.setCameraResolution(WIDTH, HEIGHT)
#     sim.setCameraVFOV(math.radians(VFOV))
#     sim.setDiscretizedViewingAngles(True)
#     sim.setBatchSize(1)
#     sim.initialize()

@dataclass
class location:
    x: float
    y: float
    z: float
    heading: float
    elevation: float
    frameId: str

@dataclass
class location_neighbour:
    x: float
    y: float
    z: float
    videoId: str
    frameId: str
    longFrameId: str
    rel_heading: float
    rel_elevation: float
    index: int

def transform_loc_neigh_to_loc(loc_neigh, base_heading, base_elevation):
    loc = location(
                x=loc_neigh.x,
                y=loc_neigh.y,
                z=loc_neigh.z,
                heading=loc_neigh.rel_heading+base_heading,
                elevation=loc_neigh.rel_elevation+base_elevation,
                frameId=loc_neigh.frameId)
    return loc

@dataclass
class state:
    videoId: str
    longId: str
    location: location
    viewIndex: int
    trajectoryId: str
    heading: float 
    elevation: float

    def nextObs(self, step):
        self.viewIndex = self.viewIndex + step
        self.location = transform_loc_neigh_to_loc(self.neighbours[self.viewIndex], self.heading, self.elevation)
        #TODO heading/elevation; Especially, when the current index changes, the rel_heading and rel_elevation should also be adapted.
        self.current_nav_type_to_base_view = self.navigable[self.viewIndex]

        #in video sim, the shift of fake views doesnt change the heading and elevation (as alway keep on current spot with only one current view)
        # self.heading = self.location.heading
        # self.elevation = self.location.elevation

    def distance(self, frame_loc):
        # if frame_loc.videoId != self.videoId:
        #     return 500
        # distance = abs(int(frame_loc.frameId.replace('.png', '').replace('output_frame_', '')) \
        #            - int(self.location.frameId.replace('.png', '').replace('output_frame_', ''))) // 6 * 1.25
        return 0

    def set_nextView(self, nextViewId):
        self.nextViewId = nextViewId

    def set_nextFrame(self, frame_loc):
        self.nextFrame = frame_loc
    
    def set_irrelevantFrames(self, frame_locs):
        self.irrelevantFrames = frame_locs

    def set_relevantFrames(self, frame_locs):
        self.relevantFrames = frame_locs

    def get_navigable_neighbours(self):
        #TODO when current viewIndex changed, the content here should be adpated such as the rel_heading and rel_elevation
        return [self.neighbours[idx] for idx, navtype in enumerate(self.navigable) if navtype==1]

    def get_neighbours(self):
        return self.neighbours

    def get_neighbour_nav_type(self):
        return self.navigable

    def set_neighbours(self):
        navigable = []
        if hasattr(self, 'relevantFrames'):
            relevantFrames = self.relevantFrames
            navigable += [1] * len(relevantFrames)
        else:
            relevantFrames = []
        if hasattr(self, 'nextFrame'):
            assert isinstance(self.nextFrame, list)
            self.neighbours = self.nextFrame + relevantFrames + self.irrelevantFrames
            navigable = [1] * len(self.nextFrame) + navigable + [0]*len(self.irrelevantFrames)
        else:
            self.neighbours = relevantFrames + self.irrelevantFrames
            navigable = navigable + [0]*len(self.irrelevantFrames)

        # new_neighbours = sorted(self.neighbours, key=lambda x:x.longFrameId)
        # neighbour_idx_mapping = {newidx:self.neighbours.index(n) for newidx, n in enumerate(self.neighbours)}
        # self.navigable = [navigable[neighbour_idx_mapping[newidx]] for newidx in range(len(navigable))]
        # self.neighbours = new_neighbours
        self.navigable = navigable

def store_dict_to_hdf5(hdf_group, dictionary):
    for key, value in dictionary.items():
        str_key = str(key)  # Convert the key to a string
        if isinstance(value, dict):
            subgroup = hdf_group.create_group(str_key)
            store_dict_to_hdf5(subgroup, value)
        elif isinstance(value, np.ndarray):
            hdf_group.create_dataset(str_key, data=value)  # Store numpy arrays as datasets
        elif isinstance(value, (int, float)):
            hdf_group.create_dataset(str_key, data=value)  # Store numeric data as datasets
        elif isinstance(value, str):
            hdf_group.create_dataset(str_key, data=np.string_(value))  # Store strings as datasets
        else:
            raise TypeError(f"Unsupported data type for key {key}: {type(value)}")

#     return sim
class VideoSim:
    cached_vid_sim = None

    def __init__(self, video_trajectory_dir, anno_file):
        self.candidate_negative_frames_cnt = 6
        self.candidate_negative_interval = 20
        self.candidate_positive_frames_cnt = 2
        self.candidate_positive_interval = 2

        all_videos = glob.glob(video_trajectory_dir + '/*json')
        all_annos = json.load(open(anno_file))

        all_video_entries = {}
        # print("Creating video sim ...")
        for anno in all_annos:
            videoId = anno['videoId']
            trajectoryId = anno['longId']
            colmapId = trajectoryId.split('|')[0]
            optView = anno['optView']
            optView = optView if optView.endswith('.png') else optView.replace('.png', '')
            path = anno['path']
            # nextViewId = optView+'.png'

            turnViewId = optView
            idx_turn = path.index(turnViewId)
            turnFrameIds = list(map(int, trajectoryId.split('|')[1].split('%')))

            for idx_cur, curViewId in enumerate(path):
                curFrameId = int(curViewId.replace(f"{videoId}_output_frame_", '').replace('.png', ''))
                if idx_cur < len(path) - 1:
                    nextViewId = path[idx_cur+1]
                    nextFrameIds = [int(nextViewId.replace(f"{videoId}_output_frame_", '').replace('.png', ''))]
                    if nextViewId == optView:
                        nextViewId = trajectoryId
                        nextFrameIds = list(map(int, trajectoryId.split('|')[1].split('%')))
                else:
                    nextViewId = f"{videoId}_output_frame_{turnFrameIds[0]:04d}.png"
                    nextFrameIds = [turnFrameIds[0]]
                
                nav = [1] * len(nextFrameIds)
                nextViewIndex = list(range(len(nextFrameIds)))
                all_video_entries.setdefault(videoId, {}).setdefault(trajectoryId, {}).setdefault(curViewId, {'nextViewId': nextViewId, 
                                                    'curFrameId': curFrameId,
                                                    'nextViewIndex': nextViewIndex,
                                                    'nextFrameIds': nextFrameIds,
                                                    'nav': nav,
                                                    'trajectoryId': trajectoryId, 
                                                    'colmapId': colmapId})

        self.init_cached_sim()
        self.all_video_entries = all_video_entries

    @classmethod
    def init_cached_sim(self):
        if self.cached_vid_sim is None:
            print("Loaing sim cache ...")
            # self.cached_vid_sim = pkl.load(open('/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/geoinformation_colmap_p1/geo_trajectory.pkl', 'rb'))
            self.cached_vid_sim = pkl.load(open('data/RoomTour/geo_trajectory.pkl', 'rb'))
        
    def get_loc(self, colmapId, frameId):
        return self.cached_vid_sim[colmapId][frameId]['pos'] 

    def get_rel_heading(self, colmapId, frameId1, frameId2):
        #TODO
        return self.cached_vid_sim[colmapId][frameId2]['yaw'] - \
            self.cached_vid_sim[colmapId][frameId1]['yaw']
        # return 0.2

    def get_rel_elevation(self, colmapId, frameId1, frameId2):
        #TODO
        return self.cached_vid_sim[colmapId][frameId2]['pitch'] - \
            self.cached_vid_sim[colmapId][frameId1]['pitch']
        # return 0.1

    def get_heading(self, colmapId, frameId):
        #TODO
        return math.radians(self.cached_vid_sim[colmapId][frameId]['yaw'] % 360)
        # return 0.2

    def get_elevation(self, colmapId, frameId):
        #TODO
        return math.radians(self.cached_vid_sim[colmapId][frameId]['pitch'] % 360)
        # return 0.1

    def getNextObs(self, step):
        # raise NotImplementedError
        assert all([step + state.viewIndex <= len(state.neighbours) for state in self.state])
        [state.nextObs(step) for state in self.state]

    # def sample_irrelevant_frames(self, videoId, frameIds, idx):
    #     sampled_res = []
    #     for i in range(self.candidate_negative_frames_cnt):
    #         new_idx = (idx + self.candidate_negative_interval*i)%len(frameIds)
    #         sampled_res.append({'videoId': videoId,
    #                             'frameId': frameIds[new_idx]})
    #     return sampled_res

    # def sample_relevant_frames(self, videoId, frameIds, idx):
    #     sampled_res = []
    #     for i in range(self.candidate_positive_frames_cnt):
    #         interval = self.candidate_positive_interval if i==0 else -self.candidate_positive_interval
    #         new_idx = idx + interval
    #         if new_idx >= len(frameIds):
    #             continue 
    #         if new_idx < 0:
    #             continue
    #         sampled_res.append({'videoId': videoId,
    #                             'frameId': frameIds[new_idx]})
    #     return sampled_res

    def _set_state(self, videoId, longId, frameId, colmapId, trajectoryId, index, x=0., y=0., z=0., heading=0., elevation=0.):
        loc = location(x=x, 
                       y=y,
                       z=z,
                       heading=heading,
                       elevation=elevation,
                       frameId=frameId)
        
        st = state(videoId=videoId, longId=longId, location=loc, viewIndex=index, trajectoryId=trajectoryId, heading=self.get_heading(colmapId, frameId), elevation=self.get_elevation(colmapId, frameId))
        return st
    
    # this is to set a new current obs with the specified viewpoint, heading and elevation. in web videos, we only care the next frameid, as there is no such elevations and headings
    def newEpisode(self, videoIds, curViewIds, trajectoryIds, headings=None, elevations=None, cindex=0):
        self.state = []
        for videoId, curViewId, trajectoryId, heading, elevation in zip(videoIds, curViewIds, trajectoryIds, headings, elevations):
            curFrameId = self.all_video_entries[videoId][trajectoryId][curViewId]['curFrameId']
            nextViewId = self.all_video_entries[videoId][trajectoryId][curViewId]['nextViewId']
            nextFrameIds = self.all_video_entries[videoId][trajectoryId][curViewId]['nextFrameIds']
            nextViewIndex = self.all_video_entries[videoId][trajectoryId][curViewId]['nextViewIndex']
            assert len(nextViewIndex) == len(nextFrameIds)
            # negativeFrame = self.all_video_entries[videoId][curViewId]['negativeFrame']
            colmapId = self.all_video_entries[videoId][trajectoryId][curViewId]['colmapId']
            # if 'longId' in self.all_video_entries[videoId][curViewId]:
            #     longId = self.all_video_entries[videoId][curViewId]['longId']
            # else:
            #     longId = f"{nextFrame['videoId']}%{nextFrame['frameId']}"
            curLoc = self.get_loc(colmapId, curFrameId)
            state = self._set_state(videoId, curViewId, curFrameId, colmapId, trajectoryId, cindex, curLoc[0], curLoc[1], curLoc[2])
            
            if nextViewId != None:
                nextFrameloc = []
                for viewindex, nframeid in zip(nextViewIndex, nextFrameIds):
                    nextloc = self.get_loc(colmapId, nframeid)
                    nextFrameloc.append(location_neighbour(x=nextloc[0],
                                                    y=nextloc[1],
                                                    z=nextloc[2],
                                                    videoId=videoId, 
                                                    frameId=nframeid, 
                                                    longFrameId=nextViewId,
                                                    rel_heading=self.get_rel_heading(colmapId, curFrameId, nframeid),
                                                    rel_elevation=self.get_rel_elevation(colmapId, curFrameId, nframeid),
                                                    index=viewindex))
                    state.set_nextFrame(nextFrameloc)
                    state.set_nextView(nextViewId)
            # if len(negativeFrame) > 0: locs)
            state.set_irrelevantFrames([])
            state.set_neighbours()
            state.current_nav_type_to_base_view = 1

        self.state.append(state)

    def getNextFrame(self):
        return [state.nextFrameloc for state in self.state]

    def getIrreleventFrames(self):
        return [state.irrelevantFrames for state in self.state]

    def getState(self):
        return self.state


#TODO: fake a simulator for room tour videos
def construct_fake_simulator(anno_file):
    return VideoSim(anno_file)

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),
         math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_angle_feature_nextviews(state, angle_feat_size):
    base_heading = state.heading
    base_elevation = state.elevation
    feature = np.empty((len(state.get_neighbours()), angle_feat_size), np.float32)
    for ix, neighbour in enumerate(state.get_neighbours()):
        heading = neighbour.rel_heading - base_heading
        elevation = neighbour.rel_elevation - base_elevation
        assert ix == neighbour.index

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature


def get_all_point_angle_feature(sim, angle_feat_size, connectivity_dir=None):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]


def load_nav_graphs(anno_file):
    ''' Load connectivity graph for each scan 
        data json:
            heading: 0
            path id
            instructions
            scan
            path
            image_id - videoId_frameId (+all candidates)
            neighbours
                pose
    '''
    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pos'][0] - pose2['pos'][0]) ** 2 \
                + (pose1['pos'][1] - pose2['pos'][1]) ** 2 \
                + (pose1['pos'][2] - pose2['pos'][2]) ** 2) ** 0.5

    graphs = {}
    # cached_vid_sim = pkl.load(open('/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/geoinformation_colmap_p1/geo_trajectory.pkl', 'rb'))
    cached_vid_sim = pkl.load(open('data/RoomTour/geo_trajectory.pkl', 'rb'))
    all_annos = json.load(open(anno_file))

    all_video_entries = {}
    # print("Creating video sim ...")
    for anno in all_annos:
        G = nx.Graph()
        positions = {}

        videoId = anno['videoId']
        trajectoryId = anno['longId']
        colmapId = trajectoryId.split('|')[0]
        optView = anno['optView']
        optView = optView if optView.endswith('.png') else optView.replace('.png', '')
        path = anno['path']
        # nextViewId = optView+'.png'

        turnViewId = optView
        idx_turn = path.index(turnViewId)
        turnFrameIds = list(map(int, trajectoryId.split('|')[1].split('%')))

        for idx_cur, curViewId in enumerate(path):
            curFrameId = int(curViewId.replace(f"{videoId}_output_frame_", '').replace('.png', ''))
            if idx_cur < len(path) - 1:
                nextViewId = path[idx_cur+1]
                nextFrameIds = [int(nextViewId.replace(f"{videoId}_output_frame_", '').replace('.png', ''))]
                if nextViewId == optView:
                    nextViewId = trajectoryId
                    nextFrameIds = list(map(int, trajectoryId.split('|')[1].split('%')))
            else:
                nextViewId = f"{videoId}_output_frame_{turnFrameIds[0]:04d}.png"
                nextFrameIds = [turnFrameIds[0]]
            
            nav = [1] * len(nextFrameIds)
            nextViewIndex = list(range(len(nextFrameIds)))
            all_video_entries.setdefault(videoId, {}).setdefault(trajectoryId, {}).setdefault(curViewId, {'nextViewId': nextViewId, 
                                                'curFrameId': curFrameId,
                                                'nextViewIndex': nextViewIndex,
                                                'nextFrameIds': nextFrameIds,
                                                'nav': nav,
                                                'trajectoryId': trajectoryId, 
                                                'colmapId': colmapId})
            positions[curViewId] = np.array([cached_vid_sim[colmapId][curFrameId]['pos'][0],
                                            cached_vid_sim[colmapId][curFrameId]['pos'][1],
                                            cached_vid_sim[colmapId][curFrameId]['pos'][2]])
            G.add_edge(curViewId, nextViewId, weight=distance(cached_vid_sim[colmapId][curFrameId], 
                                                              cached_vid_sim[colmapId][nextFrameIds[0]]))
            # G.add_edge(nextViewId, curViewId, weight=distance(cached_vid_sim[colmapId][curFrameId], 
            #                                                   cached_vid_sim[colmapId][nextFrameIds[0]]))
        nx.set_node_attributes(G, values=positions, name='position')
        graphs[trajectoryId] = G

    return graphs


def normalize_angle(x):
    '''convert radians into (-pi, pi]'''
    pi2 = 2 * math.pi
    x = x % pi2 # [0, 2pi]
    if x > math.pi:
        x = x - pi2
    return x


def convert_heading(x):
    return x % (2 * math.pi) / (2 * math.pi)   # [0, 2pi] -> [0, 1)


def convert_elevation(x):
    return (normalize_angle(x) + math.pi) / (2 * math.pi)   # [0, 2pi] -> [0, 1)


class EnvBatch(object):
    def __init__(self, feat_db=None, anno_file=None, batch_size=1):
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.sims = []
        for i in range(batch_size):
            # print("Creating fake simulator")
            sim = construct_fake_simulator(anno_file)
            self.sims.append(sim)

    def newEpisodes(self, scanIds, viewpointIds, trajectoryIds, headings):
        for i, (scanId, viewpointId, trajectoryId, heading) in enumerate(zip(scanIds, viewpointIds, trajectoryIds, headings)):
            # print("Creating new episodes")
            self.sims[i].newEpisode([scanId], [viewpointId], [trajectoryId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]

            if self.feat_db is None:
                feature = None
            else:
                feature = self.feat_db.get_image_feature(state.nextViewId)
            # features = [feature]
            # features.append(self.feat_db.get_image_feature(loc.videoId, location.frameId) for loc in [state.nextFrame] + state.irrelevantFrames)
            # features = torch.stack(features, 0)
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])