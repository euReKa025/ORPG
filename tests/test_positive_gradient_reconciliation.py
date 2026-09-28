import math
import random

import pytest
import torch

from cw_grpo.gradient_reconciliation import (
    GradientCollection, pcgrad_reconcile, correctness_priority_pcgrad_reconcile,
)
from cw_grpo.positive_gradient_reconciliation import (
    positive_direction_weights, positive_pcgrad_reconcile,
)


def collection(left, right):
    return GradientCollection([[torch.tensor(left, dtype=torch.float32)],
                               [torch.tensor(right, dtype=torch.float32)]])


@pytest.mark.parametrize('branch,base', [('symmetric', pcgrad_reconcile),
                                      ('priority', correctness_priority_pcgrad_reconcile)])
@pytest.mark.parametrize('left,right,strength', [([2., 1.], [1., 3.], 0.),
                                               ([2., 0.], [-1., 1.], 1.),
                                               ([0., 0.], [1., 3.], 1.)])
def test_literal_legacy_paths(branch, base, left, right, strength):
    g = collection(left, right)
    expected, old = base(g)
    actual, diag = positive_pcgrad_reconcile(g, strength=strength, negative_branch=branch)
    assert torch.equal(actual[0], expected[0])
    assert (diag.projection_rate, diag.solver_bypassed) == (old.projection_rate, old.solver_bypassed)


@pytest.mark.parametrize('rule', ['A', 'B', 'C'])
@pytest.mark.parametrize('left,right', [([1., 0.], [0., 3.]),
                                      ([1., 0.], [3., 0.]),
                                      ([2., 1.], [1., 2.])])
def test_boundary_identity(rule, left, right):
    g = collection(left, right)
    out, d = positive_pcgrad_reconcile(g, rule=rule, strength=1.)
    assert torch.equal(out[0], g.objectives[0][0]+g.objectives[1][0])
    assert d.positive_branch_entered and not d.positive_direction_changed


@pytest.mark.parametrize('rule', ['A', 'B', 'C'])
def test_direct_geometric_formula_and_measured_change(rule):
    left = torch.tensor([10., 0.], dtype=torch.float64)
    right = torch.tensor([.5, math.sqrt(.75)], dtype=torch.float64)
    c = .5; alpha = c if rule != 'B' else c*20/101
    q = .5 if rule == 'C' else 0.
    s = left+right
    v = left.norm()**q*left/left.norm()+right.norm()**q*right/right.norm()
    z = (1-alpha)*s+alpha*s.norm()*v/v.norm()
    expected = s.norm()*z/z.norm()
    out, d = positive_pcgrad_reconcile(collection(left.tolist(), right.tolist()), rule=rule, strength=1.)
    torch.testing.assert_close(out[0].double(), expected, atol=2e-6, rtol=2e-6)
    assert d.positive_direction_changed and d.projection_rate == 0
    assert d.output_norm_ratio == pytest.approx(1., abs=2e-6)
    expected_angle = math.acos(float(torch.dot(s, expected)/(s.norm()*expected.norm())))
    assert d.reference_angle_radians == pytest.approx(expected_angle, abs=2e-6)


def test_random_positive_descent_and_preserved_norm():
    rng = random.Random(20260907)
    for rule in ['A', 'B', 'C']:
        for _ in range(500):
            n1, n2 = 10**rng.uniform(-12, 12), 10**rng.uniform(-12, 12)
            c, strength = rng.random(), rng.random()
            x, y, _, _ = positive_direction_weights(n1, n2, c, rule=rule, strength=strength)
            assert x+c*y > 0 and y+c*x > 0
            before = math.sqrt(n1*n1+n2*n2+2*c*n1*n2)
            after = math.sqrt(x*x+y*y+2*c*x*y)
            assert after/before == pytest.approx(1., abs=1e-12)


def test_weak_gradient_limits_and_q_one():
    for rule in ['B', 'C']:
        x, y, _, angle = positive_direction_weights(1., 1e-12, .5, rule=rule, strength=1.)
        assert abs(x-1.) < 1e-5 and y < 1e-5 and angle < 1e-5
    _, y, _, angle = positive_direction_weights(1., 1e-12, .5, rule='A', strength=1.)
    assert y > .1 and angle > .1
    out, _ = positive_pcgrad_reconcile(collection([10., 0.], [.5, .866]),
                                      rule='C', q=1., strength=1.)
    assert torch.equal(out[0], torch.tensor([10.5, .866]))


def test_global_gram_changes_local_negative_branch_decision():
    # Rank0 sees a negative dot, but the full gradients have positive dot.
    local = collection([1., 0.], [-1., 1.])
    full = collection([1., 0., 3.], [-1., 1., 2.])
    expected, _ = positive_pcgrad_reconcile(full, rule='B', strength=.5)
    # Three branch Gram reductions, five legacy diagnostic reductions,
    # then actual normalized-difference reduction. Record the full reducer
    # outputs by executing the same algorithm on the full parameter vector.
    global_values = []
    def record(value):
        global_values.append(value.clone());return value
    positive_pcgrad_reconcile(full, rule='B', strength=.5, reduce_scalar=record)
    pending = iter(global_values)
    def global_reduce(value):
        return next(pending).to(value)
    actual, diag = positive_pcgrad_reconcile(local, rule='B', strength=.5, reduce_scalar=global_reduce)
    torch.testing.assert_close(actual[0], expected[0][:2], rtol=2e-6, atol=2e-6)
    assert diag.positive_branch_entered and diag.projection_rate == 0
    with pytest.raises(StopIteration): next(pending)


def test_no_norm_preservation_is_distinct_and_inputs_are_unchanged():
    g = collection([10., 0.], [.5, .8660254]);before=[x[0].clone() for x in g.objectives]
    _, d = positive_pcgrad_reconcile(g, strength=1., preserve_norm=False)
    assert d.output_norm_ratio < .999
    assert all(torch.equal(old, new[0]) for old, new in zip(before, g.objectives))


@pytest.mark.parametrize('kwargs', [{'strength': -1.}, {'strength': math.nan},
                                   {'q': 2.}, {'rule': 'unknown'}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):positive_pcgrad_reconcile(collection([1., 0.], [1., 1.]), **kwargs)
