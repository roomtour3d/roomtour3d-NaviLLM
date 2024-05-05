import os
import glob
import sys
import numpy as np
import json
import collections
# import cv2
import torch
import torch.nn as nn
import ray
from ray.util.queue import Queue
from torchvision import transforms
from PIL import Image
import math
# sys.path.append(mp3d_path)    # please add the simulator path to yout python path. 
# import MatterSim
import h5py
import argparse
import psutil
from more_itertools import batched

aug_frame_temporal=False
aug_frame_spatial=True

from PIL import Image
import random

# Define the crop strategy once so it can be reused
def get_crop_strategy(width, height, max_ratio=0.2):
    # Calculate the maximum crop dimensions based on the aspect ratio
    max_crop_width = int(width * max_ratio)
    max_crop_height = int(height * max_ratio)

    # Re-calculate crop dimensions to maintain the aspect ratio
    aspect_ratio = width / height
    crop_height = random.randint(0, max_crop_height)
    crop_width = int(crop_height * aspect_ratio)

    # Make sure the crop width does not exceed the max crop width
    crop_width = min(crop_width, max_crop_width)

    # Choose to crop top & left or top & right randomly
    if random.choice(['left', 'right']) == 'left':
        crop_strategy = ('left', crop_width, crop_height)
    else:
        crop_strategy = ('right', crop_width, crop_height)

    return crop_strategy

def crop_image(img, crop_strategy):
    # with Image.open(image_path) as img:
    width, height = img.size

    # Apply the crop strategy
    direction, crop_width, crop_height = crop_strategy
    
    if direction == 'left':
        left, upper = crop_width, crop_height
    else:
        left, upper = width - crop_width - img.size[0], crop_height

    # Calculate right and lower coordinates to keep the aspect ratio
    right = left + img.size[0]
    lower = upper + img.size[1]

    # Crop the image
    cropped_img = img.crop((left, upper, right, lower))

    # Return the cropped image without saving it
    return cropped_img


@ray.remote(num_gpus=1)
def process_features(proc_id, out_queue, scanvp_list, args):
    sys.path.append("/mnt/bn/kinetics-lp-maliva-v6/playground_projects/EVA/EVA-CLIP/rei")
    from eva_clip import create_model_and_transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('start proc_id: %d' % proc_id)

    # load visual encoder
    model, _, transform = create_model_and_transforms(args.model_name, args.pretrained, force_custom_clip=True)
    visual_encoder = model.visual.to(device)
    visual_encoder.eval()

    overlap_len = 2
    feat_clip_len = 6
    for i, d_meta in enumerate(scanvp_list):
        if aug_frame_spatial:
            vid, fid, crop = d_meta
            fidx = int(fid.split('.')[-2].split('_')[-2])
            new_fid = f"output_frame_{fidx:04d}.png"
        else:
            vid, fid = d_meta
            # Loop all discretized views from this location
            fidx = int(fid.split('.')[-2].split('_')[-1])
            new_fid = f"output_frame_{fidx+3:04d}.png"
        images = []
        if aug_frame_temporal:
            try:
                image = Image.open(os.path.join("/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/trajectory_scamera/", vid, "imgs_3fps_360p", new_fid))
                fid = new_fid
            except:
                image = Image.open(os.path.join("/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/trajectory_scamera/", vid, "imgs_3fps_360p", fid))
        else:
            image = Image.open(os.path.join("/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/trajectory_scamera/", vid, "imgs_3fps_360p", new_fid))
        if aug_frame_spatial:
            image = crop_image(image, crop)
        images.append(image)

        vision_x = [transform(image).unsqueeze(0).to(device) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        H, W = vision_x.shape[-2:]

        fts = []
        for k in range(0, len(images), args.batch_size):
            input_img = vision_x[k: k + args.batch_size]
            with torch.no_grad(), torch.cuda.amp.autocast():
                outs = visual_encoder.forward_features(input_img)
            outs = outs.data.cpu().numpy()
            fts.append(outs)
        fts = np.concatenate(fts, 0)

        out_queue.put((vid, fid, fts[0], [H,W], []))

        if i%1000==0:
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"Memory used by current process: {memory_info.rss / (1024 * 1024):.2f} MB")

    out_queue.put(None)

@ray.remote
def write_features(out_queue, total, num_workers, args):
    num_finished_workers = 0
    num_finished_vps = 0

    from progressbar import ProgressBar
    progress_bar = ProgressBar(total)
    progress_bar.start()

    with h5py.File(args.output_file, 'w' if not any([aug_frame_temporal]) else 'a') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                video_id, frame_id, fts, (H, W), logits = res
                key = '%s_%s' % (video_id, frame_id)
                if False:
                    data = np.hstack([fts, logits])
                else:
                    data = fts # shape=(36, 1408)
                if key not in outf:
                    # continue
                    outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                    outf[key][...] = data
                    outf[key].attrs['videoId'] = video_id
                    outf[key].attrs['frameId'] = frame_id
                    outf[key].attrs['image_w'] = W
                    outf[key].attrs['image_h'] = H
                    # outf[key].attrs['vfov'] = None

                num_finished_vps += 1
                if num_finished_vps % 20000 == 0:
                    print("num_finished_vps: ",num_finished_vps)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print("data shape: ", data.shape)
                progress_bar.update(num_finished_vps)

    progress_bar.finish()

import time
def main(args):

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    anno_vids = [os.path.basename(file).split('.')[0] for file in glob.glob(args.anno_dir + '/*.json')]
    viewpoint_ids = []
    for vid in anno_vids:
        included = []
        with open(os.path.join(anno_dir, '%s.json' % vid)) as f:
            data = json.load(f)
            for x in data:
                for fid in x['frames']:
                    if fid not in included:
                        included.append(fid)
                        if aug_frame_spatial:
                            viewpoint_ids.append((vid, fid, x['crop']))
                        else:
                            viewpoint_ids.append((vid, fid))

    print('Loaded %d viewpoints' % len(viewpoint_ids))
    scanvp_list = viewpoint_ids
    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    ray.init()
    out_queue = Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = process_features.remote(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        processes.append(process)

    process = write_features.remote(out_queue, len(scanvp_list), num_workers, args)
    processes.append(process)

    ray.get(processes)
    ray.shutdown()


if __name__ == '__main__':

    anno_dir = '/mnt/bn/kinetics-lp-maliva-v6/data/ytb_vln/object_progression_clip_gpt4_trajectory_spatial_cropping_p1/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EVA02-CLIP-L-14-336")
    parser.add_argument("--pretrained", type=str, default="data/models/EVA02_CLIP_L_336_psz14_s6B.pt", help='the path of pre-trained model')
    parser.add_argument('--connectivity_dir', default='data/connectivity', help='the path of connectivity')
    parser.add_argument('--anno_dir', default=anno_dir)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--output_file", type=str, default="data/eva_features/web_obj_prog_crop_p1_EVA02-CLIP-L-14-336.hdf5", help="the path of output features")
    args = parser.parse_args()

    main(args)

