"""
Comprehensive evaluation metrics for YOLO-like object detection.
Computes IoU, Precision, Recall, mAP, F1, and confusion metrics.
"""
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Container for a single detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def to_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2, self.confidence)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics at a specific IoU threshold."""
    iou_threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    ap: float


class ObjectDetectionEvaluator:
    """
    Evaluator for object detection models with single class.
    Computes standard detection metrics: precision, recall, mAP, F1, confusion matrix.
    """

    def __init__(self, iou_thresholds: List[float] = None):
        """
        Args:
            iou_thresholds: List of IoU thresholds for evaluation.
                          Default: [0.5, 0.85] (matching training metrics)
        """
        self.iou_thresholds = iou_thresholds or [0.5, 0.85]
        self.results = {}

    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float],
                   box2: Tuple[float, float, float, float]) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.

        Args:
            box1: (x1, y1, x2, y2) in pixel coordinates
            box2: (x1, y1, x2, y2) in pixel coordinates

        Returns:
            IoU score in [0, 1]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # No intersection
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        # Intersection area
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def match_predictions_to_ground_truth(
        predictions: List[Tuple[float, float, float, float, float]],
        ground_truth: List[Tuple[float, float, float, float]],
        iou_threshold: float = 0.5
    ) -> Tuple[int, int, int, np.ndarray]:
        """
        Match predictions to ground truth boxes using greedy matching by highest IoU.

        Args:
            predictions: List of (x1, y1, x2, y2, confidence)
            ground_truth: List of (x1, y1, x2, y2)
            iou_threshold: Minimum IoU for a match

        Returns:
            (tp, fp, fn, iou_scores) where:
                tp: Number of true positives
                fp: Number of false positives
                fn: Number of false negatives
                iou_scores: Array of IoU scores for each prediction
        """
        if len(predictions) == 0:
            # No predictions
            fn = len(ground_truth)
            return 0, 0, fn, np.array([])

        if len(ground_truth) == 0:
            # No ground truth
            fp = len(predictions)
            return 0, fp, 0, np.array([0.0] * len(predictions))

        # Sort predictions by confidence (descending)
        sorted_pred_idx = np.argsort(
            [-p[4] for p in predictions]
        )

        # Track which ground truth boxes have been matched
        gt_matched = np.zeros(len(ground_truth), dtype=bool)
        tp = 0
        fp = 0
        iou_scores = []

        # Match each prediction to best ground truth
        for pred_idx in sorted_pred_idx:
            pred = predictions[pred_idx]
            pred_box = (pred[0], pred[1], pred[2], pred[3])

            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue

                iou = ObjectDetectionEvaluator.compute_iou(pred_box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            iou_scores.append(best_iou)

            # Check if match is valid
            if best_iou >= iou_threshold:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        fn = np.sum(~gt_matched)

        return tp, fp, fn, np.array(iou_scores)

    def evaluate_image(
        self,
        predictions: List[Tuple[float, float, float, float, float]],
        ground_truth: List[Tuple[float, float, float, float]],
        image_id: str = None
    ) -> Dict[float, Tuple[int, int, int]]:
        """
        Evaluate predictions for a single image at all IoU thresholds.

        Args:
            predictions: List of (x1, y1, x2, y2, confidence)
            ground_truth: List of (x1, y1, x2, y2)
            image_id: Optional image identifier for tracking

        Returns:
            Dict mapping IoU threshold -> (tp, fp, fn)
        """
        results = {}

        for iou_thresh in self.iou_thresholds:
            tp, fp, fn, _ = self.match_predictions_to_ground_truth(
                predictions, ground_truth, iou_thresh
            )
            results[iou_thresh] = (tp, fp, fn)

        if image_id:
            self.results[image_id] = {
                'predictions': predictions,
                'ground_truth': ground_truth,
                'matches': results
            }

        return results

    def evaluate_batch(
        self,
        batch_predictions: List[List[Tuple[float, float, float, float, float]]],
        batch_ground_truth: List[List[Tuple[float, float, float, float]]],
        image_ids: List[str] = None
    ) -> Dict[float, Tuple[int, int, int]]:
        """
        Evaluate predictions for multiple images.

        Args:
            batch_predictions: List of prediction lists
            batch_ground_truth: List of ground truth lists
            image_ids: Optional list of image identifiers

        Returns:
            Dict mapping IoU threshold -> (total_tp, total_fp, total_fn)
        """
        aggregated = {iou: (0, 0, 0) for iou in self.iou_thresholds}

        for i, (preds, gts) in enumerate(zip(batch_predictions, batch_ground_truth)):
            img_id = image_ids[i] if image_ids else f"image_{i}"
            results = self.evaluate_image(preds, gts, img_id)

            for iou, (tp, fp, fn) in results.items():
                total_tp, total_fp, total_fn = aggregated[iou]
                aggregated[iou] = (total_tp + tp, total_fp + fp, total_fn + fn)

        return aggregated

    @staticmethod
    def compute_precision_recall_f1(
        tp: int, fp: int, fn: int
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.

        Args:
            tp: True positives
            fp: False positives
            fn: False negatives

        Returns:
            (precision, recall, f1)
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def compute_confusion_matrix(
        tp: int, fp: int, fn: int, tn: int = None
    ) -> Dict[str, int]:
        """
        Return confusion matrix components.

        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            tn: True negatives (optional, not always applicable to detection)

        Returns:
            Dictionary with confusion matrix metrics
        """
        return {
            'TP': tp,  # Correctly detected objects
            'FP': fp,  # False alarms (detected but no object)
            'FN': fn,  # Missed objects
            'TN': tn   # Correct rejections (if applicable)
        }

    @staticmethod
    def compute_ap_from_pr_curve(
        precisions: np.ndarray,
        recalls: np.ndarray
    ) -> float:
        """
        Compute Average Precision from precision-recall curve.
        Uses all-points interpolation (COCO method).

        Args:
            precisions: Array of precision values
            recalls: Array of recall values (should be sorted)

        Returns:
            Average Precision (AP) in [0, 1]
        """
        if len(precisions) == 0 or len(recalls) == 0:
            return 0.0

        # Ensure precisions and recalls are sorted by recall
        sort_idx = np.argsort(recalls)
        precisions = precisions[sort_idx]
        recalls = recalls[sort_idx]

        # Compute average precision using 11-point interpolation
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        return ap

    def compute_map(
        self,
        all_predictions: List[List[Tuple[float, float, float, float, float]]],
        all_ground_truth: List[List[Tuple[float, float, float, float]]],
        confidence_thresholds: np.ndarray = None
    ) -> Dict[float, float]:
        """
        Compute mean Average Precision (mAP) at different IoU thresholds.

        Args:
            all_predictions: List of prediction lists per image
            all_ground_truth: List of ground truth lists per image
            confidence_thresholds: Confidence thresholds to evaluate at
                                 (default: 50 evenly spaced points [0, 1])

        Returns:
            Dict mapping IoU threshold -> mAP score
        """
        if confidence_thresholds is None:
            confidence_thresholds = np.linspace(0, 1, 101)

        map_scores = {}

        for iou_thresh in self.iou_thresholds:
            precisions = []
            recalls = []

            for conf_thresh in confidence_thresholds:
                # Filter predictions by confidence
                filtered_preds = []
                for preds in all_predictions:
                    filtered = [p for p in preds if p[4] >= conf_thresh]
                    filtered_preds.append(filtered)

                # Evaluate at this confidence threshold
                batch_results = self.evaluate_batch(
                    filtered_preds, all_ground_truth
                )

                tp, fp, fn = batch_results[iou_thresh]
                precision, recall, _ = self.compute_precision_recall_f1(tp, fp, fn)

                precisions.append(precision)
                recalls.append(recall)

            # Compute AP from PR curve
            ap = self.compute_ap_from_pr_curve(
                np.array(precisions), np.array(recalls)
            )
            map_scores[iou_thresh] = ap

        return map_scores

    def compute_miou(
        self,
        predictions: List[List[Tuple[float, float, float, float, float]]],
        ground_truth: List[List[Tuple[float, float, float, float]]],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Compute mean Intersection over Union (mIoU) for all true positive predictions.

        Args:
            predictions: List of prediction lists per image
            ground_truth: List of ground truth lists per image
            iou_threshold: IoU threshold to determine true positives

        Returns:
            Mean IoU across all true positive detections
        """
        all_tp_ious = []

        for preds, gts in zip(predictions, ground_truth):
            if len(preds) == 0 or len(gts) == 0:
                continue

            # Get IoU scores for this image
            tp, fp, fn, iou_scores = self.match_predictions_to_ground_truth(
                preds, gts, iou_threshold
            )

            # Collect IoU scores for true positives (IoU >= threshold)
            tp_ious = [iou for iou in iou_scores if iou >= iou_threshold]
            all_tp_ious.extend(tp_ious)

        if len(all_tp_ious) == 0:
            return 0.0

        return np.mean(all_tp_ious)

    def get_summary_metrics(
        self,
        batch_results: Dict[float, Tuple[int, int, int]]
    ) -> Dict[float, EvaluationMetrics]:
        """
        Generate summary metrics from batch evaluation results.

        Args:
            batch_results: Output from evaluate_batch()

        Returns:
            Dict mapping IoU threshold -> EvaluationMetrics
        """
        summary = {}

        for iou_thresh in self.iou_thresholds:
            tp, fp, fn = batch_results[iou_thresh]
            precision, recall, f1 = self.compute_precision_recall_f1(tp, fp, fn)

            # AP approximation (perfect precision at computed recall)
            ap = precision * recall if recall > 0 else 0.0

            summary[iou_thresh] = EvaluationMetrics(
                iou_threshold=iou_thresh,
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                ap=ap
            )

        return summary

    def print_results(self, summary: Dict[float, EvaluationMetrics]) -> None:
        """Pretty print evaluation results."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        for iou_thresh in sorted(summary.keys()):
            metrics = summary[iou_thresh]
            print(f"\nIoU Threshold: {iou_thresh}")
            print(f"  TP: {metrics.tp:5d}  FP: {metrics.fp:5d}  FN: {metrics.fn:5d}")
            print(f"  Precision: {metrics.precision:.4f}")
            print(f"  Recall:    {metrics.recall:.4f}")
            print(f"  F1 Score:  {metrics.f1:.4f}")
            print(f"  AP:        {metrics.ap:.4f}")

        print("\n" + "="*70)


def evaluate_predictions(
    predictions: List[List[Tuple[float, float, float, float, float]]],
    ground_truth: List[List[Tuple[float, float, float, float]]],
    iou_thresholds: List[float] = None,
    verbose: bool = True
) -> Dict[float, EvaluationMetrics]:
    """
    Convenience function to evaluate predictions against ground truth.

    Args:
        predictions: List of prediction lists per image
        ground_truth: List of ground truth lists per image
        iou_thresholds: IoU thresholds to evaluate at
        verbose: Whether to print results

    Returns:
        Dict mapping IoU threshold -> EvaluationMetrics

    Example:
        predictions = [
            [(10, 20, 100, 120, 0.95), (150, 160, 220, 240, 0.85)],
            [(50, 60, 150, 180, 0.92)]
        ]
        ground_truth = [
            [(12, 22, 102, 122), (152, 162, 222, 242)],
            [(52, 62, 152, 182)]
        ]
        results = evaluate_predictions(predictions, ground_truth)
    """
    evaluator = ObjectDetectionEvaluator(iou_thresholds)
    batch_results = evaluator.evaluate_batch(predictions, ground_truth)
    summary = evaluator.get_summary_metrics(batch_results)

    if verbose:
        evaluator.print_results(summary)

    return summary
