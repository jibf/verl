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

import datasets
import pandas as pd
import torchvision.io as io

# System and user prompts from evaluate_car_crash_video_api.py
SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing car accidents from dashcam video sequences. Your task is to process a series of images from the ego vehicle's perspective and determine if an accident occurred, based on 2D object tracking data.

Input: A sequence of images from a dashcam. Each image has 2D object tracking annotations: bounding boxes around detected objects, with a track ID displayed at the top-left corner of each box. The ego vehicle is assigned track_id = 0.

Prior Knowledge for Analysis:
1. The video perspective represents the ego vehicle's viewpoint. Sudden camera shakes, rapid viewpoint changes, or unusual movements indicate significant changes in the ego vehicle's pose, suggesting potential abnormalities or impacts.
2. When visual evidence of direct physical contact between objects is ambiguous, focus on detecting abrupt behavioral changes that may indicate accidents:
   - Sudden, unexpected movement of vehicles (rapid acceleration, deceleration, or direction change) may indicate collision forces
   - Abnormal vehicle behavior that deviates from predictable traffic patterns often signals external interference
   - Chain reactions where one vehicle's sudden action causes others to react abruptly
3. Always base reasoning on observable visual evidence from the image sequence.

Instructions:
- Determine if a car accident occurred in the video sequence. An accident is defined as any abnormal event, including collisions between other vehicles, between other vehicles and the ego vehicle, or the ego vehicle losing control and hitting objects (e.g., walls) without involving other vehicles.
- If an accident occurred, assess whether the ego vehicle is directly involved.
- Output a list of track IDs for objects directly involved in the accident. If track IDs are unstable for the same object (e.g., due to tracking jumps), provide any valid track ID for that object. If the ego vehicle is involved, include track_id = 0 in the list. If no accident occurred, set this to an empty list.
- Regardless of accident occurrence, provide a concise analysis: first describe what happened in the video, then analyze the cause of the accident (if any) based on observable evidence from the images. The analysis should be clear and reasoned.

Output Format:
Your response must be a valid JSON object with the following keys and types:
- "is_accident": boolean (true or false)
- "is_ego_involved": boolean (true or false; automatically false if no accident)
- "object_id_involved": list of integers (e.g., [1, 2, 0] if objects with track_id 1, 2 and ego are involved; empty list if no accident)
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
  "accident analysis": "The red car (track_id: 5) braked suddenly → ego car (track_id: 0) could not stop in time → rear-end collision. Cause: insufficient following distance by the ego car."
}
```
### Example 2: No Accident
```json
{
"is_accident": false,
  "is_ego_involved": false,
  "object_id_involved": [],
  "accident analysis": "A nearby white car (track_id: 2) changed lanes, but no collision occurred. The scene shows normal driving behavior."
}
"""

USER_PROMPT_TEMPLATE = """Analyze the following dashcam video for car accident detection. The video is from the ego vehicle's perspective and includes 2D object tracking with bounding boxes and track IDs (as described in the system prompt). Please apply prior knowledge about vehicle behavior and camera perspective changes to interpret potential accidents, and provide detailed reasoning in your analysis. Output the results in the specified JSON format.
Note: A rule-based method has identified the following objects as potentially involved in an accident:
{rule_base_crash_results}
However, these track IDs may be inaccurate and should only be used as a reference. Always prioritize observable visual evidence from the video for your final analysis.

<video>"""


def parse_obj_id_ground_truth(obj_id_str: str) -> list[list[int]]:
    """
    Parse object ID ground truth string.

    Format: "1;3/5;6" means:
    - Split by ';' to get groups
    - '3/5' means either 3 or 5 is correct
    - No order requirement

    Args:
        obj_id_str: String like "1;3/5;6"

    Returns:
        List of acceptable ID groups, e.g., [[1], [3, 5], [6]]
    """
    if not obj_id_str or obj_id_str == 'nan' or pd.isna(obj_id_str):
        return []

    # Remove brackets if present
    obj_id_str = obj_id_str.strip('[]')

    # Split by semicolon
    groups = obj_id_str.split(';')

    parsed_groups = []
    for group in groups:
        group = group.strip()
        if '/' in group:
            # Multiple acceptable IDs
            ids = [int(x.strip()) for x in group.split('/') if x.strip().isdigit()]
            if ids:
                parsed_groups.append(ids)
        elif group.isdigit():
            parsed_groups.append([int(group)])

    return parsed_groups


def load_rule_base_results(sam_crash_detection_path: str, type_val: int, video_val: int) -> str:
    """
    Load rule-based crash detection results from SAM collision detection.

    Args:
        sam_crash_detection_path: Path to SAM crash detection directory
        type_val: Type value (e.g., 7)
        video_val: Video value (e.g., 3)

    Returns:
        Formatted string with rule-based detection results
    """
    # Construct the JSON file path: test_{type}_{video:03d}_sam_collisions.json
    json_filename = f"test_{type_val}_{video_val:03d}_sam_collisions.json"
    json_path = os.path.join(sam_crash_detection_path, json_filename)

    # If file doesn't exist, return default message
    if not os.path.exists(json_path):
        return "No rule-based detection results available."

    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract description field
        descriptions = data.get('description', [])

        if not descriptions:
            return "No rule-based detection results available."

        # Format the descriptions into a readable string
        formatted_result = "\n".join(descriptions)
        return formatted_result

    except Exception as e:
        print(f"Warning: Failed to load rule-based results from {json_path}: {e}")
        return "No rule-based detection results available."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file",
                        default="/data02/home/binfei/data/DADA/dada_car_crash_evaluation.xlsx",
                        help="Path to the annotation Excel file")
    parser.add_argument("--video_base_path",
                        default="/data02/home/binfei/workspace/ByteTrack/Vehicle_crash_image_tracking_updated_results",
                        help="Base path to video directories")
    parser.add_argument("--sam_crash_detection_path",
                        default="/data02/home/binfei/workspace/Depth-Anything-V2/sam_crash_detection",
                        help="Path to SAM crash detection results")
    parser.add_argument("--local_save_dir", default="~/data/car_crash",
                        help="The save directory for the preprocessed dataset")
    parser.add_argument("--fps", type=float, default=None,
                        help="Sample video at specified FPS (e.g., 1.0 for 1 frame per second)")
    parser.add_argument("--nframes", type=int, default=None,
                        help="Sample fixed number of frames (alternative to fps)")
    parser.add_argument("--fps_min_frames", type=int, default=4,
                        help="Minimum frames when using fps sampling")
    parser.add_argument("--fps_max_frames", type=int, default=48,
                        help="Maximum frames when using fps sampling")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training data (rest will be validation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")

    args = parser.parse_args()

    # Expand path
    args.local_save_dir = os.path.expanduser(args.local_save_dir)

    # Read Excel annotation file
    print(f"Loading annotations from {args.annotation_file}...")
    df = pd.read_excel(args.annotation_file)
    print(f"Loaded {len(df)} rows from Excel")

    # Prepare data list (filter valid samples)
    data_list = []
    total_rows = 0
    skipped_rows = 0

    for idx, row in df.iterrows():
        total_rows += 1

        # Extract values
        accident_type = str(row['accident_type']).strip()
        obj_id = str(row['obj_id']).strip()

        is_normal = accident_type in ["normal cases", "normal cases (hard)"]

        # Check for missing or invalid data (skip only for accident cases)
        if not is_normal and (not accident_type or accident_type.lower() == 'nan' or pd.isna(row['accident_type'])):
            skipped_rows += 1
            continue

        if not is_normal and (not obj_id or obj_id.lower() == 'nan' or pd.isna(row['obj_id'])):
            skipped_rows += 1
            continue

        if not is_normal and (pd.isna(row['ego_involved'])):
            skipped_rows += 1
            continue

        # All checks passed, add to data list
        type_val = int(row['type'])
        video_val = int(row['video'])
        ego_involved = bool(row['ego_involved']) if not is_normal else False

        # Frame information
        start_frame = int(row['abnormal start frame'])
        end_frame = int(row['abnormal end frame'])

        data_list.append({
            'idx': idx,
            'type': type_val,
            'video': video_val,
            'accident_type': accident_type,
            'is_normal': is_normal,
            'ego_involved': ego_involved,
            'obj_id': obj_id if not is_normal else "",
            'start_frame': start_frame,
            'end_frame': end_frame,
        })

    print(f"Prepared {len(data_list)} valid samples from annotation file")
    print(f"  Total rows in file: {total_rows}")
    print(f"  Valid samples: {len(data_list)}")
    print(f"  Skipped (missing data): {skipped_rows}")

    # Random shuffle and split train/val
    random.seed(args.seed)
    random.shuffle(data_list)

    train_size = int(len(data_list) * args.train_ratio)
    train_list = data_list[:train_size]
    val_list = data_list[train_size:]

    print(f"\nTrain/Val split:")
    print(f"  Train samples: {len(train_list)}")
    print(f"  Val samples: {len(val_list)}")

    def get_video_frame_count(video_path):
        """Get the number of frames in a video file using torchvision"""
        try:
            # Use VideoReader to get metadata without loading the entire video
            from torchvision.io import VideoReader
            reader = VideoReader(video_path, "video")
            metadata = reader.get_metadata()
            # Calculate frame count from duration and fps
            duration = metadata['video']['duration'][0]
            fps = metadata['video']['fps'][0]
            frame_count = int(duration * fps)
            return frame_count
        except Exception as e:
            # Fallback: load video to get frame count (slower but more reliable)
            try:
                video, _, _ = io.read_video(video_path, pts_unit='sec')
                return video.shape[0]
            except Exception as e2:
                print(f"Warning: Failed to get frame count for {video_path}: {e2}")
                return None

    def process_split(split_list, split_name):
        """Process a single split (train or val) into dataset format"""
        processed_data = []

        for item in split_list:
            # Construct video path
            video_id = f"{item['type']}_{item['video']:03d}"
            video_filename = f"{video_id}_f{item['start_frame']}-{item['end_frame']}.mp4"
            video_path = os.path.join(args.video_base_path, video_id, video_filename)

            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}, skipping...")
                continue

            # Get video frame count
            frame_count = get_video_frame_count(video_path)
            if frame_count is None:
                print(f"Warning: Could not determine frame count for {video_path}, skipping...")
                continue

            # Determine nframes: use actual frame count if < 48, otherwise cap at 48
            nframes = min(frame_count, args.fps_max_frames)

            # Load rule-based crash detection results
            rule_base_crash_results = load_rule_base_results(
                args.sam_crash_detection_path,
                item['type'],
                item['video']
            )

            # Format user prompt with rule-based results
            user_prompt = USER_PROMPT_TEMPLATE.format(rule_base_crash_results=rule_base_crash_results)

            # Construct video dict following qwen2-vl format
            video_dict = {
                "type": "video",
                "video": f"file://{video_path}",
                "nframes": nframes
            }

            # Determine is_accident (False for normal cases, True for accidents)
            is_accident = not item['is_normal']

            # Parse object IDs (empty for normal cases)
            obj_id_groups = parse_obj_id_ground_truth(item['obj_id']) if not item['is_normal'] else []

            # Construct data in verl format
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
                        "is_accident": is_accident,
                        "is_ego_involved": item['ego_involved'],
                        "obj_id_groups": obj_id_groups,
                        "obj_id_str": item['obj_id']
                    }
                },
                "extra_info": {
                    "split": split_name,
                    "index": item['idx'],
                    "type": item['type'],
                    "video": item['video'],
                    "accident_type": item['accident_type'],
                    "video_path": video_path,
                    "start_frame": item['start_frame'],
                    "end_frame": item['end_frame'],
                },
            }
            processed_data.append(data)

        return processed_data

    # Process train and val splits
    print("\nProcessing train split...")
    train_data = process_split(train_list, "train")
    print(f"Processed {len(train_data)} train samples")

    print("\nProcessing val split...")
    val_data = process_split(val_list, "val")
    print(f"Processed {len(val_data)} val samples")

    # Convert to HuggingFace datasets
    train_dataset = datasets.Dataset.from_list(train_data)
    val_dataset = datasets.Dataset.from_list(val_data)

    # Save to parquet
    os.makedirs(args.local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))

    print(f"\nDataset saved to {args.local_save_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Example output format
    print("\nExample data format:")
    print(train_dataset[0])
