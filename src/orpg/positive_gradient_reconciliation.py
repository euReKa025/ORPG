"""Compatible-gradient coordination with conflict resolution.

Gram values cover the same full parameter shards as the existing reconciler.
The negative branch delegates literally to the frozen PCGrad implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from orpg.gradient_reconciliation import (
    GradientCollection, ReconciliationDiagnostics, ScalarReducer,
    _diagnostics, _dot, _identity, _sum,
    correctness_priority_pcgrad_reconcile, pcgrad_reconcile,
)


@dataclass(frozen=True, slots=True)
class PositiveDiagnostics(ReconciliationDiagnostics):
    pair_opportunities: int = 0
    positive_branch_entered: bool = False
    positive_direction_changed: bool = False
    positive_alpha: float = 0.0
    reference_angle_radians: float = 0.0
    output_norm_ratio: float = 1.0


def positive_direction_weights(n1, n2, cosine, *, rule, strength, q=0.5,
                               preserve_norm=True):
    """Return amplitudes multiplying unit gradients, alpha and reference angle.

    Scaling by max(n1,n2) avoids squaring very different raw norms and avoids
    materializing potentially overflowing coefficients of tiny gradients.
    This helper is only for positive, active two-objective pairs.
    """
    if rule not in ('A', 'B', 'C'):
        raise ValueError('positive rule must be A, B or C')
    if not all(math.isfinite(x) for x in (n1, n2, cosine, strength, q)):
        raise ValueError('nonfinite positive coordination scalar')
    if min(n1, n2) <= 0 or not 0 <= cosine <= 1 or not 0 <= strength <= 1:
        raise ValueError('positive coordination domain violation')
    if not 0 <= q <= 1:
        raise ValueError('partial normalization q must be in [0,1]')
    scale = max(n1, n2)
    a, b = n1 / scale, n2 / scale
    alpha = strength * cosine
    if rule == 'B':
        alpha *= 2 * a * b / (a*a + b*b)
    exponent = q if rule == 'C' else 0.0
    v1, v2 = a**exponent, b**exponent
    snorm = math.sqrt(a*a + b*b + 2*cosine*a*b)
    vnorm = math.sqrt(v1*v1 + v2*v2 + 2*cosine*v1*v2)
    x = (1-alpha)*a + alpha*snorm*v1/vnorm
    y = (1-alpha)*b + alpha*snorm*v2/vnorm
    znorm = math.sqrt(x*x + y*y + 2*cosine*x*y)
    # Angle from the original sum, evaluated in a two-dimensional Gram basis.
    cross = abs(a*y-b*x) * math.sqrt(max(0.0, 1-cosine*cosine))
    dot = a*x + b*y + cosine*(a*y+b*x)
    angle = math.atan2(cross, dot)
    factor = snorm/znorm if preserve_norm else 1.0
    return scale*x*factor, scale*y*factor, alpha, angle


def positive_pcgrad_reconcile(gradients: GradientCollection, *, rule='A',
                             strength=0.0, q=0.5, preserve_norm=True,
                             negative_branch='symmetric',
                             reduce_scalar: ScalarReducer = _identity,
                             epsilon=1e-12):
    if negative_branch not in ('symmetric', 'priority'):
        raise ValueError('unknown negative branch')
    if len(gradients.objectives) != 2:
        raise ValueError('positive coordination requires two objectives')
    if rule not in ('A', 'B', 'C') or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('invalid positive rule or strength')
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('invalid partial normalization exponent')
    base = pcgrad_reconcile if negative_branch == 'symmetric' else correctness_priority_pcgrad_reconcile
    kwargs = dict(reduce_scalar=reduce_scalar, epsilon=epsilon)
    if strength == 0:
        return base(gradients, **kwargs)
    left, right = gradients.objectives
    a = float(_dot(left, left, reduce_scalar=reduce_scalar).detach().cpu())
    b = float(_dot(right, right, reduce_scalar=reduce_scalar).detach().cpu())
    d = float(_dot(left, right, reduce_scalar=reduce_scalar).detach().cpu())
    if not all(math.isfinite(x) for x in (a, b, d)):
        raise ValueError('nonfinite global Gram matrix')
    if a <= 0 or b <= 0 or d < 0:
        out, diag = base(gradients, **kwargs)
        return out, PositiveDiagnostics(**asdict(diag), pair_opportunities=int(a > 0 and b > 0))
    n1, n2 = math.sqrt(a), math.sqrt(b)
    c = max(0.0, min(1.0, d/(n1*n2)))
    x, y, alpha, angle = positive_direction_weights(
        n1, n2, c, rule=rule, strength=strength, q=q, preserve_norm=preserve_norm)
    before = _sum(gradients.objectives)
    # Preserve exact identity at zero mixing or mathematically unchanged direction.
    identity = alpha == 0 or c == 1 or a == b or (rule == 'C' and q == 1)
    if identity:
        out = before
    else:
        out = [((g.detach().float()/n1)*x + (h.detach().float()/n2)*y).to(g.dtype)
               for g, h in zip(left, right, strict=True)]
    diag = _diagnostics(gradients.objectives, before, out, projections=0, **kwargs)
    if not (math.isfinite(diag.combined_norm_after) and diag.combined_norm_after > 0):
        raise ValueError('nonfinite or zero positive output norm')
    # Actual normalized output difference, without subtracting cosines near one.
    # One temporary parameter block at a time avoids a full extra gradient copy.
    delta = 0.0
    if not identity:
        local_delta_sq = sum(
            ((g.float()/diag.combined_norm_after-h.float()/diag.combined_norm_before)
             .square().sum() for g, h in zip(out, before, strict=True)),
            start=torch.zeros((), device=left[0].device, dtype=torch.float32))
        delta = math.sqrt(max(0.0, float(reduce_scalar(local_delta_sq).detach().cpu())))
    changed = delta > 32 * torch.finfo(torch.float32).eps
    return out, PositiveDiagnostics(
        **asdict(diag), pair_opportunities=1, positive_branch_entered=True,
        positive_direction_changed=changed, positive_alpha=alpha,
        reference_angle_radians=2*math.asin(min(1.0, delta/2)),
        output_norm_ratio=diag.combined_norm_after/diag.combined_norm_before)
