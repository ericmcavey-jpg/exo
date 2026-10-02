from math import isclose

import pytest

from exo.likeness.identity import (
    IdentityPolicy,
    build_reference,
    cosine_similarity,
    judge_identity,
    required_similarity,
)

POLICY = IdentityPolicy(enforce=True, minimum_similarity=0.45, maximum_drop=0.10)


def test_cosine_similarity():
    assert isclose(cosine_similarity([1.0, 0.0], [2.0, 0.0]), 1.0, abs_tol=1e-9)
    assert isclose(cosine_similarity([1.0, 0.0], [0.0, 3.0]), 0.0, abs_tol=1e-9)
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_reference_drops_faces_that_are_not_you():
    you = [[1.0, 0.05, 0.0], [0.95, 0.0, 0.05], [1.0, 0.0, 0.0]]
    mistagged = [[0.0, 0.0, 1.0]]
    reference, discarded = build_reference(you + mistagged)
    assert discarded == 1
    assert cosine_similarity(reference, [1.0, 0.0, 0.0]) > 0.99
    with pytest.raises(ValueError):
        build_reference([])


def test_required_similarity_tracks_the_source_photo():
    assert isclose(required_similarity(POLICY, 0.80), 0.70, abs_tol=1e-9)
    assert isclose(required_similarity(POLICY, 0.50), 0.45, abs_tol=1e-9)
    # A weak source match only guards against further drift.
    assert isclose(required_similarity(POLICY, 0.40), 0.30, abs_tol=1e-9)


def test_judge_identity():
    assert judge_identity(POLICY, 0.80, 0.75).accepted
    assert not judge_identity(POLICY, 0.80, 0.60).accepted
    assert not judge_identity(POLICY, 0.80, None).accepted
    assert judge_identity(POLICY, None, None).accepted
    report_only = IdentityPolicy(enforce=False, require_face=False)
    assert judge_identity(report_only, 0.80, 0.10).accepted
