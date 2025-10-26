import numpy as np
import pytest
from evaluation import accuracy, precision, tpr, fpr, f1, metrics_summary, metrics


class TestMetricsSummary:
    """Test the metrics_summary function which expects values encoded as 1 and -1"""

    def test_perfect_predictions(self):
        """Test when all predictions are correct"""
        y_true = np.array([1, 1, -1, -1, 1, -1])
        y_pred = np.array([1, 1, -1, -1, 1, -1])

        acc, prec, tprate, fprate, f1score = metrics_summary(y_true, y_pred)

        assert acc == 1.0, f"Expected accuracy 1.0, got {acc}"
        assert prec == 1.0, f"Expected precision 1.0, got {prec}"
        assert tprate == 1.0, f"Expected TPR 1.0, got {tprate}"
        assert fprate == 0.0, f"Expected FPR 0.0, got {fprate}"
        assert f1score == 1.0, f"Expected F1 1.0, got {f1score}"

    def test_all_wrong_predictions(self):
        """Test when all predictions are wrong"""
        y_true = np.array([1, 1, -1, -1, 1, -1])
        y_pred = np.array([-1, -1, 1, 1, -1, 1])

        acc, prec, tprate, fprate, f1score = metrics_summary(y_true, y_pred)

        assert acc == 0.0, f"Expected accuracy 0.0, got {acc}"
        assert tprate == 0.0, f"Expected TPR 0.0, got {tprate}"
        assert fprate == 1.0, f"Expected FPR 1.0, got {fprate}"
        # When TPR=0, precision would be 0/3=0, F1 would be 0
        assert f1score == 0.0, f"Expected F1 0.0, got {f1score}"

    def test_mixed_predictions(self):
        """Test with a mix of correct and incorrect predictions"""
        # y_true: 2 positives (1), 2 negatives (-1)
        # y_pred: TP=1, FN=1, FP=1, TN=1
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, 1, -1])

        acc, prec, tprate, fprate, f1score = metrics_summary(y_true, y_pred)

        # Accuracy: (TP + TN) / Total = (1 + 1) / 4 = 0.5
        assert acc == 0.5, f"Expected accuracy 0.5, got {acc}"

        # Precision: TP / (TP + FP) = 1 / (1 + 1) = 0.5
        assert prec == 0.5, f"Expected precision 0.5, got {prec}"

        # TPR (Recall): TP / (TP + FN) = 1 / (1 + 1) = 0.5
        assert tprate == 0.5, f"Expected TPR 0.5, got {tprate}"

        # FPR: FP / (FP + TN) = 1 / (1 + 1) = 0.5
        assert fprate == 0.5, f"Expected FPR 0.5, got {fprate}"

        # F1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert f1score == 0.5, f"Expected F1 0.5, got {f1score}"

    def test_all_predicted_positive(self):
        """Test when all predictions are positive"""
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, 1, 1, 1])

        acc, prec, tprate, fprate, f1score = metrics_summary(y_true, y_pred)

        # Accuracy: (TP + TN) / Total = (2 + 0) / 4 = 0.5
        assert acc == 0.5, f"Expected accuracy 0.5, got {acc}"

        # Precision: TP / (TP + FP) = 2 / (2 + 2) = 0.5
        assert prec == 0.5, f"Expected precision 0.5, got {prec}"

        # TPR: TP / (TP + FN) = 2 / (2 + 0) = 1.0
        assert tprate == 1.0, f"Expected TPR 1.0, got {tprate}"

        # FPR: FP / (FP + TN) = 2 / (2 + 0) = 1.0
        assert fprate == 1.0, f"Expected FPR 1.0, got {fprate}"

        # F1: 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
        expected_f1 = 2 * (0.5 * 1.0) / (0.5 + 1.0)
        assert np.isclose(f1score, expected_f1), f"Expected F1 {expected_f1}, got {f1score}"

    def test_all_predicted_negative(self):
        """Test when all predictions are negative"""
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([-1, -1, -1, -1])

        acc, prec, tprate, fprate, f1score = metrics_summary(y_true, y_pred)

        # Accuracy: (TP + TN) / Total = (0 + 2) / 4 = 0.5
        assert acc == 0.5, f"Expected accuracy 0.5, got {acc}"

        # TPR: TP / (TP + FN) = 0 / (0 + 2) = 0.0
        assert tprate == 0.0, f"Expected TPR 0.0, got {tprate}"

        # FPR: FP / (FP + TN) = 0 / (0 + 2) = 0.0
        assert fprate == 0.0, f"Expected FPR 0.0, got {fprate}"

        # F1: should be 0 (no true positives)
        assert f1score == 0.0, f"Expected F1 0.0, got {f1score}"


class TestMetrics:
    """Test the metrics function which expects values encoded as 1 and 0"""

    def test_perfect_predictions(self):
        """Test when all predictions are correct"""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0])

        acc, recall, fpr_val, prec, f1_val = metrics(y_true, y_pred)

        assert acc == 1.0, f"Expected accuracy 1.0, got {acc}"
        assert prec == 1.0, f"Expected precision 1.0, got {prec}"
        assert recall == 1.0, f"Expected recall 1.0, got {recall}"
        assert fpr_val == 0.0, f"Expected FPR 0.0, got {fpr_val}"
        assert f1_val == 1.0, f"Expected F1 1.0, got {f1_val}"

    def test_all_wrong_predictions(self):
        """Test when all predictions are wrong"""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        acc, recall, fpr_val, prec, f1_val = metrics(y_true, y_pred)

        assert acc == 0.0, f"Expected accuracy 0.0, got {acc}"
        assert recall == 0.0, f"Expected recall 0.0, got {recall}"
        assert fpr_val == 1.0, f"Expected FPR 1.0, got {fpr_val}"
        assert f1_val == 0.0, f"Expected F1 0.0, got {f1_val}"

    def test_mixed_predictions(self):
        """Test with a mix of correct and incorrect predictions"""
        # y_true: 2 positives (1), 2 negatives (0)
        # y_pred: TP=1, FN=1, FP=1, TN=1
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])

        acc, recall, fpr_val, prec, f1_val = metrics(y_true, y_pred)

        # Accuracy: (TP + TN) / Total = (1 + 1) / 4 = 0.5
        assert acc == 0.5, f"Expected accuracy 0.5, got {acc}"

        # Precision: TP / (TP + FP) = 1 / (1 + 1) = 0.5
        assert prec == 0.5, f"Expected precision 0.5, got {prec}"

        # Recall (TPR): TP / (TP + FN) = 1 / (1 + 1) = 0.5
        assert recall == 0.5, f"Expected recall 0.5, got {recall}"

        # FPR: FP / (FP + TN) = 1 / (1 + 1) = 0.5
        assert fpr_val == 0.5, f"Expected FPR 0.5, got {fpr_val}"

        # F1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert f1_val == 0.5, f"Expected F1 0.5, got {f1_val}"

    def test_all_predicted_positive(self):
        """Test when all predictions are positive"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        acc, recall, fpr_val, prec, f1_val = metrics(y_true, y_pred)

        # Accuracy: (TP + TN) / Total = (2 + 0) / 4 = 0.5
        assert acc == 0.5, f"Expected accuracy 0.5, got {acc}"

        # Precision: TP / (TP + FP) = 2 / (2 + 2) = 0.5
        assert prec == 0.5, f"Expected precision 0.5, got {prec}"

        # Recall: TP / (TP + FN) = 2 / (2 + 0) = 1.0
        assert recall == 1.0, f"Expected recall 1.0, got {recall}"

        # FPR: FP / (FP + TN) = 2 / (2 + 0) = 1.0
        assert fpr_val == 1.0, f"Expected FPR 1.0, got {fpr_val}"

        # F1: 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
        expected_f1 = 2 * (0.5 * 1.0) / (0.5 + 1.0)
        assert np.isclose(f1_val, expected_f1), f"Expected F1 {expected_f1}, got {f1_val}"

    def test_all_predicted_negative(self):
        """Test when all predictions are negative"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        acc, recall, fpr_val, prec, f1_val = metrics(y_true, y_pred)

        # Accuracy: (TP + TN) / Total = (0 + 2) / 4 = 0.5
        assert acc == 0.5, f"Expected accuracy 0.5, got {acc}"

        # Recall: TP / (TP + FN) = 0 / (0 + 2) = 0.0
        assert recall == 0.0, f"Expected recall 0.0, got {recall}"

        # FPR: FP / (FP + TN) = 0 / (0 + 2) = 0.0
        assert fpr_val == 0.0, f"Expected FPR 0.0, got {fpr_val}"

        # F1: should be 0 (no true positives)
        assert f1_val == 0.0, f"Expected F1 0.0, got {f1_val}"


class TestComparison:
    """Test to compare both functions and identify discrepancies"""

    def test_same_data_different_encoding(self):
        """Test that both functions give same results when using correct encoding"""
        # Create test data for metrics (0/1 encoding)
        y_true_01 = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred_01 = np.array([1, 0, 1, 0, 1, 1, 0, 0])

        # Convert to -1/1 encoding for metrics_summary
        y_true_m11 = np.where(y_true_01 == 0, -1, 1)
        y_pred_m11 = np.where(y_pred_01 == 0, -1, 1)

        # Get results from both functions
        acc1, prec1, tprate1, fprate1, f1score1 = metrics_summary(y_true_m11, y_pred_m11)
        acc2, recall2, fpr2, prec2, f1_2 = metrics(y_true_01, y_pred_01)

        # Compare results
        print(f"\nmetrics_summary: acc={acc1}, prec={prec1}, tpr={tprate1}, fpr={fprate1}, f1={f1score1}")
        print(f"metrics:         acc={acc2}, prec={prec2}, recall={recall2}, fpr={fpr2}, f1={f1_2}")

        assert np.isclose(acc1, acc2), f"Accuracy mismatch: {acc1} vs {acc2}"
        assert np.isclose(prec1, prec2), f"Precision mismatch: {prec1} vs {prec2}"
        assert np.isclose(tprate1, recall2), f"TPR/Recall mismatch: {tprate1} vs {recall2}"
        assert np.isclose(fprate1, fpr2), f"FPR mismatch: {fprate1} vs {fpr2}"
        assert np.isclose(f1score1, f1_2), f"F1 mismatch: {f1score1} vs {f1_2}"

    def test_metrics_summary_with_wrong_encoding(self):
        """Test what happens when metrics_summary gets 0/1 encoding (wrong)"""
        # This test demonstrates the bug - using 0/1 encoding with metrics_summary
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])

        # This should FAIL or give wrong results because metrics_summary expects -1/1
        # The function checks for (y_true == -1) but gets 0 instead
        try:
            acc, prec, tprate, fprate, f1score = metrics_summary(y_true, y_pred)
            print(f"\nWrong encoding test - metrics_summary with 0/1:")
            print(f"  Results: acc={acc}, prec={prec}, tpr={tprate}, fpr={fprate}, f1={f1score}")
            print(f"  WARNING: These results are likely INCORRECT!")

            # The correct results should be acc=0.5, prec=0.5, recall=0.5, fpr=0.5, f1=0.5
            # But we'll get different results because the function treats 0 as neither 1 nor -1
        except Exception as e:
            print(f"\nmetrics_summary failed with 0/1 encoding: {e}")

    def test_metrics_with_wrong_encoding(self):
        """Test what happens when metrics gets -1/1 encoding (wrong)"""
        # This test demonstrates the bug - using -1/1 encoding with metrics
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, 1, -1])

        # This should FAIL or give wrong results because metrics expects 0/1
        # The function checks for (y_true == 1) and (y_predicted == 0) but gets -1 instead
        try:
            acc, recall, fpr_val, prec, f1_val = metrics(y_true, y_pred)
            print(f"\nWrong encoding test - metrics with -1/1:")
            print(f"  Results: acc={acc}, prec={prec}, recall={recall}, fpr={fpr_val}, f1={f1_val}")
            print(f"  WARNING: These results are likely INCORRECT!")

            # The -1 values will be treated as neither 0 nor 1, causing incorrect calculations
        except Exception as e:
            print(f"\nmetrics failed with -1/1 encoding: {e}")
