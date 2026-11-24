"""Every template result reports which stage produced its score.

The fine refinement can be discarded when it collapses (a false peak inside the
neighbourhood scores well below the coarse candidate it came from), in which
case the reported score describes the coarse pose. Without a stage marker two
scores are silently incomparable.
"""

import numpy as np
import pytest

from measure.measure_template import TemplatePoint, _prefer_fine_result


def _reference():
    """A target with a unique, well-conditioned feature and a smooth background."""
    yy, xx = np.mgrid[0:300, 0:400]
    image = np.clip(100 + yy * 0.1 + xx * 0.05, 0, 255).astype(np.uint8)
    patch = np.zeros((80, 80), np.uint8)
    for i in range(0, 80, 10):
        for j in range(0, 80, 10):
            patch[i:i+10, j:j+10] = 200 if (i // 10 + j // 10) % 2 else 60
    image[120:200, 160:240] = patch
    return image


def test_single_pass_reports_fine():
    reference = _reference()
    point = TemplatePoint(reference, click_row=160, click_col=200, template_size=80,
                          decimate_exponent=0)
    point.rotation_invariant = True
    point.angle_range = (-5.0, 5.0)
    point.angle_step = 1.0
    result = point.measure(reference)
    assert result["score_stage"] == "fine"


def test_two_stage_reports_fine_when_the_refinement_holds():
    reference = _reference()
    point = TemplatePoint(reference, click_row=160, click_col=200, template_size=80,
                          decimate_exponent=1, coarse_fine=False)
    point.match_score_threshold = 0.5
    result = point.measure(reference)
    assert result["score_stage"] == "fine"


def test_fallback_to_the_coarse_pose_is_marked():
    """A refinement that collapses hands back the coarse pose, and says so.

    The 90% rule lives at the end of ``_run_fine_search_on_roi``; the earlier
    exits (ROI too small for the template) return the coarse result before any
    refinement happens, so this drives the deciding branch itself.
    """
    reference = _reference()
    point = TemplatePoint(reference, click_row=160, click_col=200, template_size=80,
                          decimate_exponent=1, coarse_fine=False)
    coarse_pose = {"matched_row": 160.0, "matched_col": 200.0, "dx": 0.0, "dy": 0.0,
                   "match_score": 0.9, "valid": True, "best_rotation_deg": 0.0,
                   "best_scale": 1.0}
    collapsed = {"matched_row": 165.0, "matched_col": 205.0, "dx": 5.0, "dy": 5.0,
                 "match_score": 0.9 * 0.5, "valid": True,
                 "best_rotation_deg": 0.0, "best_scale": 1.0}

    # Reproduce the deciding tail of the method with its two candidate results.
    decision = _prefer_fine_result(coarse_pose, collapsed)
    assert decision["score_stage"] == "coarse_fallback"
    assert decision["matched_row"] == pytest.approx(160.0)
    assert decision["match_score"] == pytest.approx(0.9)

    held = _prefer_fine_result(coarse_pose, dict(collapsed, match_score=0.95))
    assert held["score_stage"] == "fine"
    assert held["matched_row"] == pytest.approx(165.0)


def test_validity_follows_the_reported_score():
    """A result whose score misses the threshold must not report valid.

    Each stage gates at its own threshold -- the coarse stage deliberately uses
    the relaxed coarse_score_threshold -- so a result can carry a validity
    decided by one stage and a score produced by another. The reported pair must
    agree, because callers branch on ``valid`` and log ``match_score``.
    """
    reference = _reference()
    point = TemplatePoint(reference, click_row=160, click_col=200, template_size=80,
                          decimate_exponent=1, coarse_fine=False)
    point.match_score_threshold = 0.95
    result = point.measure(reference)
    assert result["valid"] == (result["match_score"] >= point.match_score_threshold)
