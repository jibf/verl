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
Car crash detection reward function.

Computes reward based on:
1. is_accident accuracy
2. is_ego_involved accuracy
3. object_id_involved precision and recall

The function parses JSON predictions and compares with ground truth.
"""

import json
import re
import logging


def parse_json_response(response: str) -> dict[str, any]:
    """
    Parse JSON response from model output.
    Handles cases where model outputs thinking with </think> tag before the final JSON.

    Args:
        response: Raw response string from model

    Returns:
        Dictionary with parsed fields

    Raises:
        ValueError: If JSON parsing fails
    """
    try:
        # First, handle </think> tag if present
        if '</think>' in response:
            # Find the last occurrence of </think>
            think_end = response.rfind('</think>')
            # Take everything after </think>
            json_str = response[think_end + 8:].strip()
        else:
            json_str = response

        # Try to extract JSON from markdown code blocks
        if "```json" in json_str:
            json_start = json_str.find("```json") + 7
            json_end = json_str.find("```", json_start)
            json_str = json_str[json_start:json_end].strip()
        elif "```" in json_str:
            json_start = json_str.find("```") + 3
            json_end = json_str.find("```", json_start)
            json_str = json_str[json_start:json_end].strip()

        # Try to find JSON object in the string
        if '{' in json_str and '}' in json_str:
            json_start = json_str.find('{')
            json_end = json_str.rfind('}') + 1
            json_str = json_str[json_start:json_end]

        parsed = json.loads(json_str)

        # Normalize keys (handle both "accident analysis" and "analysis")
        if 'analysis' in parsed and 'accident analysis' not in parsed:
            parsed['accident analysis'] = parsed['analysis']

        return {
            'is_accident': parsed.get('is_accident', False),
            'is_ego_involved': parsed.get('is_ego_involved', False),
            'object_id_involved': parsed.get('object_id_involved', []),
            'object_id_at_fault': parsed.get('object_id_at_fault', []),
            'analysis': parsed.get('accident analysis', parsed.get('analysis', '')),
            'parse_failed': False
        }
    except Exception as e:
        # Return default values if parsing fails
        print(f"Warning: Failed to parse JSON response: {e}")
        print(f"Response snippet: {response[:200]}")
        return {
            'is_accident': False,
            'is_ego_involved': False,
            'object_id_involved': [],
            'object_id_at_fault': [],
            'analysis': '',
            'parse_failed': True
        }


def evaluate_object_ids(pred_ids: list[int], gt_groups: list[list[int]]) -> dict[str, any]:
    """
    Evaluate object ID prediction with precision and recall.

    Args:
        pred_ids: Predicted object IDs (list of integers)
        gt_groups: Ground truth ID groups, e.g., [[1], [3, 5], [6]]
                   Each group represents alternative acceptable IDs

    Returns:
        Dictionary with precision, recall, tp, fp, total_gt_groups, total_preds
    """
    pred_set = set(pred_ids)
    num_gt_groups = len(gt_groups)
    num_preds = len(pred_set)

    # Count matches: for each GT group, check if any ID from that group is in predictions
    matched_groups = 0
    for group in gt_groups:
        if any(gt_id in pred_set for gt_id in group):
            matched_groups += 1

    # Count how many predictions match at least one GT group (TP)
    matched_preds = 0
    for pred_id in pred_set:
        for group in gt_groups:
            if pred_id in group:
                matched_preds += 1
                break

    # TP: correct predictions
    tp = matched_preds
    # FP: incorrect predictions
    fp = num_preds - matched_preds

    # Precision and Recall (handle edge cases)
    # Note: Empty predictions should give precision=0 (not 1.0) to avoid gaming the metric
    precision = tp / num_preds if num_preds > 0 else 0.0
    recall = matched_groups / num_gt_groups if num_gt_groups > 0 else 0.0
    if num_gt_groups == 0 and num_preds == 0:
        precision = 1.0
        recall = 1.0

    return {
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'total_gt_groups': num_gt_groups,
        'total_preds': num_preds
    }


def compute_score(predict_str: str, ground_truth: dict | str) -> float:
    """
    Compute reward score for car crash detection.

    Args:
        predict_str: Model's prediction string (should be JSON format)
        ground_truth: Ground truth dictionary with keys:
            - is_accident: bool
            - is_ego_involved: bool
            - obj_id_groups: list[list[int]]
            - obj_id_str: str (optional, for debugging)

    Returns:
        Float reward score in range [0, 1]

    Reward formula:
        reward = (is_accident_correct + is_ego_correct + precision + recall) / 4.0

    Where:
        - is_accident_correct: 1.0 if prediction matches GT, 0.0 otherwise
        - is_ego_correct: 1.0 if prediction matches GT, 0.0 otherwise
        - precision: TP / (TP + FP) for object IDs
        - recall: matched_groups / total_gt_groups for object IDs
    """
    # Parse ground truth if it's a string
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except:
            print(f"Warning: Failed to parse ground_truth string: {ground_truth[:100]}")
            return 0.0

    # Parse prediction
    try:
        pred = parse_json_response(predict_str)
    except Exception as e:
        print(f"Error parsing prediction: {e}")
        return -1.0

    # Check if parsing failed
    if pred.get('parse_failed', False):
        print(f"JSON parsing failed, returning penalty reward: -1.0")
        return -1.0

    # Extract GT values
    gt_is_accident = ground_truth.get('is_accident', True)  # Default to True for backward compatibility
    gt_is_ego = ground_truth.get('is_ego_involved', False)
    gt_at_fault_groups = ground_truth.get('at_fault_groups', [])
    gt_obj_id_groups = ground_truth.get('obj_id_groups', [])

    # Compute is_accident accuracy # if not match, set reward as 0
    is_accident_correct = 1.0 if pred['is_accident'] == gt_is_accident else -1.0

    # # Compute is_ego_involved accuracy
    # is_ego_correct = 1.0 if pred['is_ego_involved'] == gt_is_ego else 0.0

    # Compute object_id precision and recall
    obj_metrics = evaluate_object_ids(pred['object_id_involved'], gt_obj_id_groups)
    precision = obj_metrics['precision']
    recall = obj_metrics['recall']

    at_fault_mmetrics = evaluate_object_ids(pred['object_id_at_fault'], gt_at_fault_groups)
    at_fault_precision = at_fault_mmetrics['precision']
    at_fault_recall = at_fault_mmetrics['recall']

    # at_fault_accuracy

    # Compute final reward (equal weights for all 3 metrics)
    if is_accident_correct < 0:
        reward = -1.0
    elif len(gt_at_fault_groups) != 0:
        reward = (precision + recall + at_fault_precision + at_fault_recall) / 4.0
    else:
        reward = (precision + recall) / 2.0
    # reward = is_accident_correct

    print(f"reward: {reward}")


    return reward
