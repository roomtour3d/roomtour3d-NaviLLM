from pickle import NONE
import numpy as np
import math
import torch
import networkx as nx
import json
import os
import glob
import math
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

@dataclass
class state:
    videoId: str
    location: location
    viewIndex: int
    heading: float 
    elevation: float

    def nextObs(self, step):
        self.viewIndex = self.viewIndex + step
        self.location = self.neighbours[self.viewIndex-1]
        #TODO heading/elevation; Especially, when the current index changes, the rel_heading and rel_elevation should also be adapted.
        self.current_nav_type_to_base_view = self.navigable[self.viewIndex-1]

    def distance(self, frame_loc):
        if frame_loc.videoId != self.videoId:
            return 500
        distance = abs(int(frame_loc.frameId.replace('.png', '').replace('output_frame_', '')) \
                   - int(self.location.frameId.replace('.png', '').replace('output_frame_', ''))) // 6 * 1.25
        return distance

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
            self.neighbours = [self.nextFrame] + relevantFrames + self.irrelevantFrames
            navigable = [1] + navigable + [0]*len(self.irrelevantFrames)
        else:
            self.neighbours = relevantFrames + self.irrelevantFrames
            navigable = navigable + [0]*len(self.irrelevantFrames)
        new_neighbours = sorted(self.neighbours, key=lambda x:x.longFrameId)
        neighbour_idx_mapping = {newidx:self.neighbours.index(n) for newidx, n in enumerate(new_neighbours)}
        self.navigable = [navigable[neighbour_idx_mapping[newidx]] for newidx in range(len(navigable))]
        self.neighbours = new_neighbours

#     return sim
class VideoSim:
    def __init__(self, video_trajectory_dir):
        self.candidate_negative_frames_cnt = 6
        self.candidate_negative_interval = 20
        self.candidate_positive_frames_cnt = 2
        self.candidate_positive_interval = 2

        all_videos = glob.glob(video_trajectory_dir + '/*json')

        all_video_entries = {}
        last_videoId = None
        last_video_frames = None
        for i, file in enumerate(all_videos):
            vid = os.path.basename(file).replace('.json', '')
            if last_video_frames is None:
                last_videoId = os.path.basename(all_videos[i-1]).replace('.json', '')
                last_video = all_videos[i-1] if i != 0 else all_videos[-1]
                last_video_frames = []
                for entry in json.load(open(last_video)):
                    last_video_frames.extend(entry['frames'])
                last_video_frames = list(set(last_video_frames))
                last_video_frames = sorted(last_video_frames)
            
            current_video_entries = {}
            current_video_frames = []
            current_total_idx = 0
            for entry in json.load(open(file)):
                current_video_frames.extend(entry['frames'])
            current_video_frames = list(set(current_video_frames))
            current_video_frames = sorted(current_video_frames)
            for entry in json.load(open(file)):
                for j, frame in enumerate(entry['frames']):
                    current_video_entries[frame] = {'nextFrame': {'videoId': vid, 'frameId': 
                                                                  entry['frames'][j+1]} if j < len(entry['frames'])-1 else None,
                                                    'relevantFrames': self.sample_relevant_frames(vid, current_video_frames, current_total_idx+j),
                                                    'irrelevantFrames': self.sample_irrelevant_frames(last_videoId, last_video_frames, current_total_idx+j)}
                current_total_idx += len(entry['frames'])

            all_video_entries[vid] = current_video_entries
            last_videoId = vid
            last_video_frames = current_video_frames
            
        self.all_video_entries = all_video_entries


    def getNextObs(self, step):
        assert all([step + state.viewIndex <= len(state.neighbours) for state in self.state])
        [state.nextObs(step) for state in self.state]

    def sample_irrelevant_frames(self, videoId, frameIds, idx):
        sampled_res = []
        for i in range(self.candidate_negative_frames_cnt):
            new_idx = (idx + self.candidate_negative_interval*i)%len(frameIds)
            sampled_res.append({'videoId': videoId,
                                'frameId': frameIds[new_idx]})
        return sampled_res

    def sample_relevant_frames(self, videoId, frameIds, idx):
        sampled_res = []
        for i in range(self.candidate_positive_frames_cnt):
            interval = self.candidate_positive_interval if i==0 else -self.candidate_positive_interval
            new_idx = idx + interval
            if new_idx >= len(frameIds):
                continue 
            if new_idx < 0:
                continue
            sampled_res.append({'videoId': videoId,
                                'frameId': frameIds[new_idx]})
        return sampled_res

    def _set_state(self, videoId, frameId, x=0., y=0., z=0., heading=0., elevation=0.):
        loc = location(x=x, 
                       y=y, 
                       z=z,
                       heading=heading,
                       elevation=elevation,
                       frameId=frameId)
        st = state(videoId=videoId, location=loc, viewIndex=0, heading=0., elevation=0)
        return st
    
    # this is to set a new current obs with the specified viewpoint, heading and elevation. in web videos, we only care the next frameid, as there is no such elevations and headings
    def newEpisode(self, videoIds, frameIds, headings=None, elevations=None):
        self.state = []
        for videoId, frameId, heading, elevation in zip(videoIds, frameIds, headings, elevations):
            state = self._set_state(videoId, frameId)
            nextFrame = self.all_video_entries[videoId][frameId]['nextFrame']
            irrelevantFrames = self.all_video_entries[videoId][frameId]['irrelevantFrames']
            relevantFrames = self.all_video_entries[videoId][frameId]['relevantFrames']
            
            if len(relevantFrames) > 0:
                relevantFramelocs = [location_neighbour(x=0.,
                                    y=0.,
                                    z=0.,
                                    videoId=frame["videoId"], 
                                    frameId=frame["frameId"], 
                                    longFrameId=f"{frame['videoId']}%{frame['frameId']}",
                                    rel_heading=0.,
                                    rel_elevation=0.) for frame in relevantFrames]
                state.set_relevantFrames(relevantFramelocs)
            if nextFrame != None:
                nextFrameloc = location_neighbour(x=0.,
                                                y=0.,
                                                z=0.,
                                                videoId=nextFrame["videoId"], 
                                                frameId=nextFrame["frameId"], 
                                                longFrameId=f"{nextFrame['videoId']}%{nextFrame['frameId']}",
                                                rel_heading=0.,
                                                rel_elevation=0.)
                state.set_nextFrame(nextFrameloc)
            irrelevantFramelocs = [location_neighbour(x=0.,
                                                     y=0.,
                                                     z=0.,
                                                     videoId=frame["videoId"], 
                                                     frameId=frame["frameId"], 
                                                     longFrameId=f"{frame['videoId']}%{frame['frameId']}",
                                                     rel_heading=0.,
                                                     rel_elevation=0.) for frame in irrelevantFrames]
            
            state.set_irrelevantFrames(irrelevantFramelocs)
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
def construct_fake_simulator(connectivity_dir):
    return VideoSim(connectivity_dir)

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),
         math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)


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


def load_nav_graphs(connectivity_dir, scans):
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
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            # for i, item in enumerate(data):
            #     if item['included']:
            #         for j, conn in enumerate(item['unobstructed']):
            #             if conn and data[j]['included']:
            #                 positions[item['image_id']] = np.array([item['pose'][3],
            #                                                         item['pose'][7], item['pose'][11]]);
            #                 assert data[j]['unobstructed'][i], 'Graph should be undirected'
            #                 G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            for i, item in enumerate(data):
                for j, conn in enumerate(item['neighbours']):
                    positions[item['image_id']] = np.array([item['pose'][3],
                                                  item['pose'][7], item['pose'][11]])
                    G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
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
    def __init__(self, connectivity_dir, feat_db=None, batch_size=1):
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.sims = []
        for i in range(batch_size):
            sim = construct_fake_simulator(connectivity_dir)
            self.sims.append(sim)

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

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
                feature = self.feat_db.get_image_feature(state.videoId, state.location.frameId)
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