import torch

from cw_grpo.objectives import (
    build_advantage_bundle,
    cw_grpo_policy_loss,
    gdpo_policy_loss,
)


def _mask(batch: int = 1, group: int = 1, tokens: int = 1) -> torch.Tensor:
    return torch.ones(batch, group, tokens)


def test_matched_components_sum_to_total_advantage() -> None:
    torch.manual_seed(0)
    rewards = torch.randn(3, 4, 2)
    weights = torch.tensor([1.0, 0.5])

    bundle = build_advantage_bundle(rewards, weights)

    assert torch.allclose(
        bundle.component_advantages.sum(dim=-1),
        bundle.total_advantage,
        atol=1e-6,
        rtol=1e-6,
    )
    assert abs(float(bundle.total_advantage.mean())) < 1e-6
    assert torch.allclose(
        bundle.total_advantage.std(unbiased=False),
        torch.tensor(1.0),
        atol=1e-5,
        rtol=1e-5,
    )


def test_zero_variance_component_becomes_zero() -> None:
    rewards = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
            ]
        ]
    )
    weights = torch.tensor([1.0, 1.0])

    bundle = build_advantage_bundle(rewards, weights)

    assert torch.allclose(
        bundle.component_advantages[..., 0],
        torch.zeros_like(bundle.component_advantages[..., 0]),
    )
    assert not bool(bundle.valid_component_mask[..., 0].any())
    assert bool(bundle.valid_component_mask[..., 1].all())


def test_no_clipping_gdpo_equals_cw_with_mixed_signs() -> None:
    component_adv = torch.tensor([[[1.0, -0.8]]])
    old_log_probs = torch.zeros(1, 1, 1)
    current_log_probs = torch.log(torch.tensor([[[1.05]]]))
    mask = _mask()

    gdpo = gdpo_policy_loss(current_log_probs, old_log_probs, mask, component_adv)
    cw = cw_grpo_policy_loss(current_log_probs, old_log_probs, mask, component_adv)

    assert torch.allclose(gdpo, cw, atol=1e-7, rtol=1e-7)


def test_same_sign_advantages_equal_even_when_clipped() -> None:
    component_adv = torch.tensor([[[1.0, 0.3]]])
    old_log_probs = torch.zeros(1, 1, 1)
    current_log_probs = torch.log(torch.tensor([[[1.4]]]))
    mask = _mask()

    gdpo = gdpo_policy_loss(current_log_probs, old_log_probs, mask, component_adv)
    cw = cw_grpo_policy_loss(current_log_probs, old_log_probs, mask, component_adv)

    assert torch.allclose(gdpo, cw, atol=1e-7, rtol=1e-7)


def test_mixed_sign_advantages_differ_when_clipped() -> None:
    component_adv = torch.tensor([[[1.0, -0.8]]])
    old_log_probs = torch.zeros(1, 1, 1)
    current_log_probs = torch.log(torch.tensor([[[1.4]]]))
    mask = _mask()

    gdpo = gdpo_policy_loss(current_log_probs, old_log_probs, mask, component_adv)
    cw = cw_grpo_policy_loss(current_log_probs, old_log_probs, mask, component_adv)

    assert not torch.allclose(gdpo, cw)
    assert cw > gdpo


def test_gradients_equal_without_clipping() -> None:
    component_adv = torch.tensor([[[1.0, -0.8]]])
    mask = _mask()

    x1 = torch.tensor([[[0.02]]], requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    old = torch.zeros_like(x1)

    loss_gdpo = gdpo_policy_loss(x1, old, mask, component_adv)
    loss_cw = cw_grpo_policy_loss(x2, old, mask, component_adv)

    loss_gdpo.backward()
    loss_cw.backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-7, rtol=1e-7)
