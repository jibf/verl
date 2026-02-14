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
Merge multiple parquet files into a single parquet file with optional sampling
"""

import argparse
import os
import random
from glob import glob

import datasets


def parse_input_with_ratio(input_arg):
    """
    Parse input argument which can be:
    - Simple path/pattern: "dir/train.parquet"
    - Path with ratio: "dir/train.parquet:2.0"
    Returns: (path, ratio)
    """
    if ':' in input_arg:
        parts = input_arg.rsplit(':', 1)
        try:
            ratio = float(parts[1])
            return parts[0], ratio
        except ValueError:
            # Not a valid ratio, treat as part of path
            return input_arg, 1.0
    return input_arg, 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple parquet files with optional sampling")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input parquet files/patterns, optionally with sampling ratios. "
                             "Format: 'path/to/file.parquet' or 'path/to/file.parquet:ratio'. "
                             "Examples: 'dir1/train.parquet:2.0' (upsample 2x), 'dir2/train.parquet:0.5' (downsample 50%%)")
    parser.add_argument("--output", required=True,
                        help="Output parquet file path")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the merged dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling and sampling")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Parse input patterns with ratios
    input_configs = []
    for input_arg in args.input:
        pattern, ratio = parse_input_with_ratio(input_arg)
        pattern = os.path.expanduser(pattern)
        matched_files = glob(pattern)

        if matched_files:
            for f in matched_files:
                input_configs.append((f, ratio))
        elif os.path.exists(pattern):
            input_configs.append((pattern, ratio))
        else:
            print(f"Warning: No files found for pattern: {pattern}")

    if not input_configs:
        raise ValueError("No input files found!")

    print(f"Found {len(input_configs)} parquet files to merge:")
    for file_path, ratio in input_configs:
        print(f"  - {file_path} (ratio: {ratio})")

    # Load and sample datasets
    datasets_list = []
    total_samples_before = 0
    total_samples_after = 0

    for file_path, ratio in input_configs:
        try:
            ds = datasets.load_dataset("parquet", data_files=file_path, split="train")
            original_size = len(ds)
            total_samples_before += original_size

            # Apply sampling ratio
            if ratio != 1.0:
                target_size = int(original_size * ratio)
                if ratio > 1.0:
                    # Upsampling: sample with replacement
                    indices = random.choices(range(original_size), k=target_size)
                    ds = ds.select(indices)
                else:
                    # Downsampling: sample without replacement
                    indices = random.sample(range(original_size), target_size)
                    ds = ds.select(indices)
                print(f"Loaded {file_path}: {original_size} -> {len(ds)} samples (ratio: {ratio})")
            else:
                print(f"Loaded {file_path}: {original_size} samples")

            total_samples_after += len(ds)
            datasets_list.append(ds)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not datasets_list:
        raise ValueError("No datasets were successfully loaded!")

    print(f"\nMerging {len(datasets_list)} datasets:")
    print(f"  Total samples before sampling: {total_samples_before}")
    print(f"  Total samples after sampling: {total_samples_after}")

    # Align features across all datasets to ensure compatibility
    # Find the first dataset with the most complete schema (non-null types)
    reference_features = None
    for ds in datasets_list:
        # Check if this dataset has a more complete schema
        if reference_features is None:
            reference_features = ds.features
        else:
            # Prefer features with actual types over null types
            for key in ds.features:
                if key in reference_features:
                    ref_type = str(reference_features[key])
                    ds_type = str(ds.features[key])
                    # If reference has null but current dataset has concrete type, update
                    if 'null' in ref_type.lower() and 'null' not in ds_type.lower():
                        reference_features[key] = ds.features[key]

    # Cast all datasets to use the reference features
    print("\nAligning dataset schemas...")
    aligned_datasets = []
    for i, ds in enumerate(datasets_list):
        try:
            aligned_ds = ds.cast(reference_features)
            aligned_datasets.append(aligned_ds)
        except Exception as e:
            print(f"Warning: Could not align dataset {i}: {e}")
            # Try to continue with original dataset
            aligned_datasets.append(ds)

    # Concatenate all datasets
    merged_dataset = datasets.concatenate_datasets(aligned_datasets)

    # Shuffle if requested
    if args.shuffle:
        print(f"Shuffling dataset with seed={args.seed}...")
        merged_dataset = merged_dataset.shuffle(seed=args.seed)

    # Save to output
    output_path = os.path.expanduser(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged dataset to {output_path}...")
    merged_dataset.to_parquet(output_path)

    print(f"\nMerge complete!")
    print(f"  Total samples: {len(merged_dataset)}")
    print(f"  Output: {output_path}")
