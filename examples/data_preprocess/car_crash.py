# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess car crash detection dataset to parquet format for GRPO training
"""

import argparse
import json
import os
import random
from glob import glob

import datasets
import pandas as pd
import torchvision.io as io

# System and user prompts from evaluate_car_crash_video_api.py
# SYSTEM_PROMPT = """
# You are an AI assistant specialized in analyzing car accidents from dashcam video sequences. Your task is to process a series of images from the ego vehicle's perspective and determine if an accident occurred, based on 2D object tracking data.

# Input: A sequence of images from a dashcam. Each image has 2D object tracking annotations: bounding boxes around detected objects, with a track ID displayed at the top-left corner of each box. The ego vehicle is assigned track_id = 0.

# Prior Knowledge for Analysis:
# 1. The video perspective represents the ego vehicle's viewpoint. Sudden camera shakes, rapid viewpoint changes, or unusual movements indicate significant changes in the ego vehicle's pose, suggesting potential abnormalities or impacts.
# 2. When visual evidence of direct physical contact between objects is ambiguous, focus on detecting abrupt behavioral changes that may indicate accidents:
#    - Sudden, unexpected movement of vehicles (rapid acceleration, deceleration, or direction change) may indicate collision forces
#    - Abnormal vehicle behavior that deviates from predictable traffic patterns often signals external interference
#    - Chain reactions where one vehicle's sudden action causes others to react abruptly
# 3. Always base reasoning on observable visual evidence from the image sequence.

# Instructions:
# - Determine if a car accident occurred in the video sequence. An accident is defined as any abnormal event, including collisions between other vehicles, between other vehicles and the ego vehicle, or the ego vehicle losing control and hitting objects (e.g., walls) without involving other vehicles.
# - If an accident occurred, assess whether the ego vehicle is directly involved.
# - Output a list of track IDs for objects directly involved in the accident. If track IDs are unstable for the same object (e.g., due to tracking jumps), provide any valid track ID for that object. If the ego vehicle is involved, include track_id = 0 in the list. If no accident occurred, set this to an empty list.
# - Regardless of accident occurrence, provide a concise analysis: first describe what happened in the video, then analyze the cause of the accident (if any) based on observable evidence from the images. The analysis should be clear and reasoned.

# Output Format:
# Your response must be a valid JSON object with the following keys and types:
# - "is_accident": boolean (true or false)
# - "is_ego_involved": boolean (true or false; automatically false if no accident)
# - "object_id_involved": list of integers (e.g., [1, 2, 0] if objects with track_id 1, 2 and ego are involved; empty list if no accident)
# - "accident analysis": string (a concise explanation in English of the events and causes, based on visual evidence)

# Important:
# - Do not include any additional text or explanations outside the JSON object.
# - In your analysis, explicitly describe the reasoning process that connects visual observations to conclusions.
# - Use prior knowledge to interpret visual evidence, but always ground your analysis in what is observable in the images.

# Examples:

# ### Example 1: Accident occurs
# ```json
# {
#   "is_accident": true,
#   "is_ego_involved": true,
#   "object_id_involved": [0, 5],
#   "accident analysis": "The red car (track_id: 5) braked suddenly → ego car (track_id: 0) could not stop in time → rear-end collision. Cause: insufficient following distance by the ego car."
# }
# ```
# ### Example 2: No Accident
# ```json
# {
# "is_accident": false,
#   "is_ego_involved": false,
#   "object_id_involved": [],
#   "accident analysis": "A nearby white car (track_id: 2) changed lanes, but no collision occurred. The scene shows normal driving behavior."
# }
# """

# USER_PROMPT_TEMPLATE = """Analyze the following dashcam video for car accident detection. The video is from the ego vehicle's perspective and includes 2D object tracking with bounding boxes and track IDs (as described in the system prompt). Please apply prior knowledge about vehicle behavior and camera perspective changes to interpret potential accidents, and provide detailed reasoning in your analysis. Output the results in the specified JSON format.
# Note: A rule-based method has identified the following objects as potentially involved in an accident:
# {rule_base_crash_results}
# However, these track IDs may be inaccurate and should only be used as a reference. Always prioritize observable visual evidence from the video for your final analysis.
# <video>"""



##########################################################################################
# # add at_fault
# SYSTEM_PROMPT = """
# You are an AI assistant specialized in analyzing car accidents from dashcam video. Your task is to process a series of images from a ego vehicle perspective and analyze the cause, contributing factors, and formation process of the accident, based on 2D object tracking data.

# Input: A sequence of images from a dashcam video. Each image contains 2D object tracking annotations, including bounding boxes around detected objects, with a track ID displayed at the top-left corner of each box. The ego vehicle is assigned track_id = 0.

# Prior Knowledge for Analysis:
# 1. The perspective represents the ego vehicle's viewpoint. Sudden camera shakes, rapid viewpoint changes, or unusual movements indicate significant changes in the ego vehicle's pose, suggesting potential abnormalities or impacts.
# 2. When visual evidence of direct physical contact between objects is ambiguous, focus on detecting abrupt behavioral changes that may indicate accidents:
#    - Sudden, unexpected movements of vehicles (e.g., rapid acceleration, deceleration, or direction changes) may indicate collision forces.
#    - Abnormal vehicle behavior that deviates from predictable traffic patterns often signals external interference.
#    - Chain reactions where one vehicle's sudden action causes others to react abruptly.
# 3. For at-fault determination, objects that violate traffic rules or perform unsafe maneuvers are typically considered at fault. If responsibility is shared, all responsible objects should be marked as at fault.
# 4. Always base reasoning on observable visual evidence from the image sequence.

# Instructions:
# - The video sequence is guaranteed to contain a car accident. Always set "is_accident" to true.
# - Assess whether the ego vehicle is directly involved in the accident.
# - Output a list of track IDs for objects directly involved in the accident. If track IDs are unstable for the same object (e.g., due to tracking jumps), provide any valid track ID for that object. If the ego vehicle is involved, include track_id = 0.
# - From the involved objects, identify the at-fault object(s). The at-fault objects must be a subset of the involved objects.
# - Provide a concise analysis: first describe what happened in the video, then analyze the cause and formation process of the accident based on observable evidence.

# Output Format:
# Your response must be a valid JSON object with the following keys and types:
# - "is_accident": boolean (true or false)
# - "is_ego_involved": boolean (true or false)
# - "object_id_involved": list of integers (e.g., [1, 2, 0] if objects with track_id 1, 2, and the ego vehicle are involved)
# - "object_id_at_fault": list of integers (should be a subset of object_id_involved)
# - "accident analysis": string (a concise explanation in English of the events and causes, based on visual evidence)

# Important:
# - Do not include any additional text or explanations outside the JSON object.
# - In your analysis, explicitly describe the reasoning process that connects visual observations to conclusions.
# - Use prior knowledge to interpret visual evidence, but always ground your analysis in what is observable in the images.

# Examples:

# ### Example 1: Accident occurs
# ```json
# {
#   "is_accident": true,
#   "is_ego_involved": true,
#   "object_id_involved": [0, 5],
#   "object_id_at_fault": [0],
#   "accident analysis": "The red car (track_id: 5) braked suddenly → the ego vehicle (track_id: 0) could not stop in time → a rear-end collision occurred. Cause: insufficient following distance by the ego vehicle."
# }
# ```
# """
# USER_PROMPT_TEMPLATE = """Analyze the following dashcam video for car accident detection. The video is from the ego vehicle's perspective and includes 2D object tracking with bounding boxes and track IDs (as described in the system prompt). Please apply prior knowledge about vehicle behavior and camera perspective changes to interpret accidents, and provide detailed reasoning in your analysis. Output the results in the specified JSON format.
# <video>"""

##########################################################################################
# involved id only
SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing car accidents from dashcam video. Your task is to process a series of images from a ego vehicle perspective and analyze the cause, contributing factors, and formation process of the accident, based on 2D object tracking data.

Input: A sequence of images from a dashcam video. Each image contains 2D object tracking annotations, including bounding boxes around detected objects, with a track ID displayed at the top-left corner of each box. The ego vehicle is assigned track_id = 0.

Prior Knowledge for Analysis:
1. The perspective represents the ego vehicle's viewpoint. Sudden camera shakes, rapid viewpoint changes, or unusual movements indicate significant changes in the ego vehicle's pose, suggesting potential abnormalities or impacts.
2. When visual evidence of direct physical contact between objects is ambiguous, focus on detecting abrupt behavioral changes that may indicate accidents:
   - Sudden, unexpected movements of vehicles (e.g., rapid acceleration, deceleration, or direction changes) may indicate collision forces.
   - Abnormal vehicle behavior that deviates from predictable traffic patterns often signals external interference.
   - Chain reactions where one vehicle's sudden action causes others to react abruptly.
3. Always base reasoning on observable visual evidence from the image sequence.

Instructions:
- The video sequence is guaranteed to contain a car accident. Always set "is_accident" to true.
- Assess whether the ego vehicle is directly involved in the accident.
- Output a list of track IDs for objects directly involved in the accident. If track IDs are unstable for the same object (e.g., due to tracking jumps), provide any valid track ID for that object. If the ego vehicle is involved, include track_id = 0.
- Provide a concise analysis: first describe what happened in the video, then analyze the cause and formation process of the accident based on observable evidence.

Output Format:
Your response must be a valid JSON object with the following keys and types:
- "is_accident": boolean (true or false)
- "is_ego_involved": boolean (true or false)
- "object_id_involved": list of integers (e.g., [1, 2, 0] if objects with track_id 1, 2, and the ego vehicle are involved)
- "object_id_at_fault": list of integers (should be a subset of object_id_involved)
- "accident analysis": string (a concise explanation in English of the events and causes, based on visual evidence)

Important:
- Do not include any additional text or explanations outside the JSON object.
- In your analysis, explicitly describe the reasoning process that connects visual observations to conclusions.
- Use prior knowledge to interpret visual evidence, but always ground your analysis in what is observable in the images.

Examples:

### Example 1: Accident occurs
```json
{
  "is_accident": true,
  "is_ego_involved": true,
  "object_id_involved": [0, 5],
  "object_id_at_fault": [0],
  "accident analysis": "The red car (track_id: 5) braked suddenly → the ego vehicle (track_id: 0) could not stop in time → a rear-end collision occurred. Cause: insufficient following distance by the ego vehicle."
}
```
"""


USER_PROMPT_TEMPLATE = """Analyze the following dashcam video of a car accident (confirmed by human). The video is from the ego vehicle's perspective and includes 2D object tracking with bounding boxes and track IDs (as described in the system prompt). Please apply prior knowledge about vehicle behavior and camera perspective changes to interpret accidents, and provide detailed reasoning in your analysis. Output the results in the specified JSON format.
<video>"""


def parse_obj_id_ground_truth(obj_id_str: str) -> list[list[int]]:
    """
    Parse object ID ground truth string.
    Format: "1;3/5;6" -> [[1], [3, 5], [6]] (semicolon separates groups, slash means alternatives)
    """
    if not obj_id_str or obj_id_str == 'nan' or pd.isna(obj_id_str):
        return []

    obj_id_str = obj_id_str.strip('[]')
    groups = obj_id_str.split(';')

    parsed_groups = []
    for group in groups:
        group = group.strip()
        if '/' in group:
            ids = [int(x.strip()) for x in group.split('/') if x.strip().isdigit()]
            if ids:
                parsed_groups.append(ids)
        elif group.isdigit():
            parsed_groups.append([int(group)])

    return parsed_groups


def load_rule_base_results(rule_base_crash_detection_path: str, video_sub_dir: str) -> str:
    """Load rule-based crash detection results from SAM collision detection."""
    # json_filename = f"test_{video_sub_dir}_sam_collisions.json"
    json_filename = f"test_{video_sub_dir.rsplit("/", 1)[0].replace("/", "_")}_sam_collisions.json"
    json_path = os.path.join(rule_base_crash_detection_path, json_filename)

    if not os.path.exists(json_path):
        print("can not find rule base results")
        return "No rule-based detection results available."

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        descriptions = data['description']
        if descriptions == []:
            return "No accident detected."

        return "\n".join(descriptions)

    except Exception as e:
        print(f"Warning: Failed to load rule-based results from {json_path}: {e}")
        return "No rule-based detection results available."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_json_pattern",
                        default="/dev-shared/binfei/data/MM_AU/CAP-DATA_chunks/chunk_*/annotations_*.json",
                        help="Glob pattern for annotation JSON files")
    parser.add_argument("--video_base_path",
                        default="/dev-shared/binfei/data/MM_AU/CAP-DATA_chunks/tracking_visualization",
                        help="Base path to video directories")
    parser.add_argument("--rule_base_crash_detection_path",
                        default="/dev-shared/binfei/data/MM_AU/CAP-DATA_chunks/rule_base_det_results/",
                        help="Path to SAM crash detection results")
    parser.add_argument("--local_save_dir", default="/dev-shared/binfei/data/MM_AU/CAP-DATA_chunks/",
                        help="The save directory for the preprocessed dataset")
    parser.add_argument("--nframes", type=int, default=48,
                        help="Sample fixed number of frames (alternative to fps)")
    parser.add_argument("--train_ratio", type=float, default=0.95,
                        help="Ratio of training data (rest will be validation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--sampling_ratio", type=float, default=1.0,
                        help="Sampling ratio for dataset. >1.0 for upsampling (with replacement), <1.0 for downsampling")

    args = parser.parse_args()

    args.local_save_dir = os.path.expanduser(args.local_save_dir)

    # Load all JSON annotation files
    print(f"Loading annotations from {args.annotation_json_pattern}...")
    json_files = glob(args.annotation_json_pattern)
    print(f"Found {len(json_files)} JSON files")

    all_annotations = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_annotations.update(data)

    print(f"Loaded {len(all_annotations)} total annotations from all JSON files")

    # Filter by is_valid and prepare data list
    data_list = []
    total_entries = len(all_annotations)
    skipped_entries = 0

    for idx, (video_key, annotation) in enumerate(all_annotations.items()):
        if not annotation.get('is_valid', False):
            skipped_entries += 1
            continue

        video_subdir = os.path.dirname(video_key)
        filename = os.path.basename(video_key)

        # Convert obj_id list to semicolon-separated string
        obj_id_list = annotation.get('obj_id', [])
        obj_id_str = ';'.join(map(str, obj_id_list)) if obj_id_list else ""

        at_fault_list = annotation.get('at_fault', [])
        at_fault_str = ';'.join(map(str, at_fault_list)) if at_fault_list else ""

        # Extract frame numbers from filename (e.g., "10_1609_f37-90.mp4" -> 37, 90)
        if "frames_" in filename:
            frame_part = filename.split('frames_')[-1].replace('.mp4', '')
        else:
            frame_part = filename.split('_f')[-1].replace('.mp4', '')
        if '-' in frame_part:
            start_frame, end_frame = map(int, frame_part.split('-'))
        else:
            start_frame = end_frame = 0

        data_list.append({
            'idx': idx,
            'video_key': video_key,
            'video_subdir': video_subdir,
            'filename': filename,
            'is_accident': annotation.get('is_accident', False),
            'ego_involved': annotation.get('ego_involved', False),
            'obj_id': obj_id_str,
            'obj_id_list': obj_id_list,
            'at_fault': at_fault_str,
            'at_fault_list': at_fault_list,
            'accident_type': annotation.get('accident_type', ''),
            'start_frame': start_frame,
            'end_frame': end_frame,
        })

    print(f"Prepared {len(data_list)} valid samples from annotation files")
    print(f"  Total entries: {total_entries}, Valid: {len(data_list)}, Skipped: {skipped_entries}")

    # Shuffle and split train/val
    random.seed(args.seed)
    random.shuffle(data_list)

    train_size = int(len(data_list) * args.train_ratio)
    train_list = data_list[:train_size]
    val_list = data_list[train_size:]

    # Apply sampling ratio
    if args.sampling_ratio != 1.0:
        print(f"\nApplying sampling ratio: {args.sampling_ratio}")

        # Apply to train split
        target_train_size = int(len(train_list) * args.sampling_ratio)
        if args.sampling_ratio > 1.0:
            # Upsampling: sample with replacement
            train_list = random.choices(train_list, k=target_train_size)
        else:
            # Downsampling: sample without replacement
            train_list = random.sample(train_list, target_train_size)

        # Apply to val split
        target_val_size = int(len(val_list) * args.sampling_ratio)
        if args.sampling_ratio > 1.0:
            # Upsampling: sample with replacement
            val_list = random.choices(val_list, k=target_val_size)
        else:
            # Downsampling: sample without replacement
            val_list = random.sample(val_list, target_val_size)

    print(f"\nTrain/Val split:")
    print(f"  Train samples: {len(train_list)}")
    print(f"  Val samples: {len(val_list)}")

    def get_video_frame_count(video_path):
        """Get the number of frames in a video file"""
        try:
            from torchvision.io import VideoReader
            reader = VideoReader(video_path, "video")
            metadata = reader.get_metadata()
            duration = metadata['video']['duration'][0]
            fps = metadata['video']['fps'][0]
            frame_count = int(duration * fps)
            return frame_count
        except Exception:
            try:
                video, _, _ = io.read_video(video_path, pts_unit='sec')
                return video.shape[0]
            except Exception as e:
                print(f"Warning: Failed to get frame count for {video_path}: {e}")
                return None

    def process_split(split_list, split_name):
        """Process a single split (train or val) into dataset format"""
        processed_data = []

        for item in split_list:
            video_path = os.path.join(args.video_base_path, item['video_subdir'], item['filename'])

            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}, skipping...")
                continue

            frame_count = get_video_frame_count(video_path)
            if frame_count is None:
                print(f"Warning: Could not determine frame count for {video_path}, skipping...")
                continue

            # Align with qwen_vl_utils smart_nframes logic
            # FRAME_FACTOR = 2, FPS_MIN_FRAMES = 4
            nframes = min(frame_count, args.nframes)
            nframes = max(nframes, 4)  # Ensure min_frames >= 4
            nframes = (nframes // 2) * 2  # Floor to multiple of 2 (never exceed frame_count)            

            rule_base_crash_results = load_rule_base_results(
                args.rule_base_crash_detection_path,
                item['video_subdir']
            )

            user_prompt = USER_PROMPT_TEMPLATE.format(rule_base_crash_results=rule_base_crash_results)

            video_dict = {
                "type": "video",
                "video": f"file://{video_path}",
                "nframes": nframes
            }

            obj_id_groups = parse_obj_id_ground_truth(item['obj_id'])
            at_fault_groups = parse_obj_id_ground_truth(item['at_fault'])

            data = {
                "data_source": "car_crash",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "videos": [video_dict],
                "ability": "video_understanding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "is_accident": item['is_accident'],
                        "is_ego_involved": item['ego_involved'],
                        "obj_id_groups": obj_id_groups,
                        "obj_id_str": item['obj_id'],
                        "at_fault_groups": at_fault_groups,
                        "at_fault_str": item['at_fault']
                    }
                },
                "extra_info": {
                    "split": split_name,
                    "index": item['idx'],
                    "video_key": item['video_key'],
                    "accident_type": item['accident_type'],
                    "video_path": video_path,
                    "start_frame": item['start_frame'],
                    "end_frame": item['end_frame'],
                },
            }
            processed_data.append(data)

        return processed_data

    print("\nProcessing train split...")
    train_data = process_split(train_list, "train")
    print(f"Processed {len(train_data)} train samples")

    print("\nProcessing val split...")
    val_data = process_split(val_list, "val")
    print(f"Processed {len(val_data)} val samples")

    train_dataset = datasets.Dataset.from_list(train_data)
    val_dataset = datasets.Dataset.from_list(val_data)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train_wo_rulebase.parquet"))
    val_dataset.to_parquet(os.path.join(args.local_save_dir, "test_wo_rulebase.parquet"))

    print(f"\nDataset saved to {args.local_save_dir}")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    print("\nExample data format:")
    print(train_dataset[0])
