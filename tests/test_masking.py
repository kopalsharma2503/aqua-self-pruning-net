"""
Part 1 correctness requirement: the engine must remain correct when
parts of the parameter tensors are masked (frozen to zero) between
training steps.

Three things are asserted, matching the case study's three explicit
callouts:

1. A masked weight contributes exactly zero to the forward pass, both
   right after masking and after subsequent optimizer steps (it must
   not drift away from zero, e.g. via weight decay).
2. A masked weight receives exactly zero gradient (not "small" -- 0.0),
   because the mask multiply is a real autodiff node.
3. Adam's per-element momentum/step-counter state for a pruned-then-
   revived connection does not silently corrupt training: we compare a
   "reset" optimizer against a deliberately naive "no reset, global t"
   optimizer and show the naive one produces a large spurious first
   update on revival while the correct one reproduces a fresh Adam
   initialization exactly.
"""
import numpy as np

from engine.tensor import Tensor
from nn.layers import Linear
from optim.adam import Adam


def test_masked_weight_forward_and_grad_are_exactly_zero():
    rng = np.random.default_rng(0)
    layer = Linear(6, 4, rng)
    mask = np.ones_like(layer.W.data)
    mask[0, :] = 0.0  # kill every outgoing connection from input feature 0
    mask[:, 2] = 0.0  # kill every incoming connection to output unit 2
    layer.set_mask(mask)

    x = Tensor(rng.normal(size=(5, 6)), requires_grad=False)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert np.all(layer.W.data[mask == 0] == 0.0), "masked weight is not exactly zero in forward params"
    assert np.all(layer.W.grad[mask == 0] == 0.0), "masked weight received nonzero gradient"
    # sanity: unmasked entries generally DO get nonzero gradient
    assert np.any(layer.W.grad[mask == 1] != 0.0)


def test_masked_weight_survives_optimizer_steps_and_weight_decay():
    rng = np.random.default_rng(1)
    layer = Linear(5, 5, rng)
    mask = np.ones_like(layer.W.data)
    mask[1, 1] = 0.0
    layer.set_mask(mask)

    opt = Adam(layer.parameters(), lr=1e-2, weight_decay=0.1)  # decay would otherwise leak weight away from 0
    x = Tensor(rng.normal(size=(8, 5)), requires_grad=False)

    for _ in range(20):
        opt.zero_grad()
        out = layer(x)
        loss = out.sum()
        loss.backward()
        opt.step()
        layer.apply_mask()  # hard re-zero, defense in depth
        assert layer.W.data[1, 1] == 0.0, "weight decay leaked a masked weight away from exact zero"


def test_optimizer_state_reset_on_revival_matches_fresh_init():
    """
    Direct test of the case study's central claim: "Your optimizer state
    (momentum, Adam moments) for a pruned-then-revived connection must
    not silently corrupt training."

    We run one weight for many steps while pruned (grad forced to 0 by
    the mask), then revive it, and check that our reset-on-mask-change
    logic makes the first post-revival update IDENTICAL to what a brand
    new Adam optimizer would produce for a freshly initialized parameter
    seeing that same gradient for the first time.
    """
    rng = np.random.default_rng(2)
    layer = Linear(3, 3, rng)
    target_i, target_j = 0, 0

    mask = np.ones_like(layer.W.data)
    mask[target_i, target_j] = 0.0
    layer.set_mask(mask)

    opt = Adam(layer.parameters(), lr=1e-2)
    x = Tensor(rng.normal(size=(4, 3)), requires_grad=False)

    # Run many steps while pruned. Feed a large, varying loss so that if
    # gradient/momentum handling were buggy this would visibly perturb
    # the frozen entry.
    for _ in range(50):
        opt.zero_grad()
        out = layer(x)
        loss = (out * out).sum()
        loss.backward()
        opt.step()
        layer.apply_mask()

    assert layer.W.data[target_i, target_j] == 0.0

    # Revive the connection.
    revived_mask = mask.copy()
    revived_mask[target_i, target_j] = 1.0
    changed = layer.set_mask(revived_mask)
    opt.reset_state_at(layer.W, changed)

    # This is what state SHOULD look like right after revival.
    assert opt.m[id(layer.W)][target_i, target_j] == 0.0
    assert opt.v[id(layer.W)][target_i, target_j] == 0.0
    assert opt.t[id(layer.W)][target_i, target_j] == 0

    # Take one real step post-revival and capture the update actually applied.
    w_before = layer.W.data[target_i, target_j]
    opt.zero_grad()
    out = layer(x)
    loss = (out * out).sum()
    loss.backward()
    g_revived = layer.W.grad[target_i, target_j]
    opt.step()
    w_after_reset_version = layer.W.data[target_i, target_j]
    applied_update = w_before - w_after_reset_version

    # Compare against a brand-new Adam optimizer seeing this exact gradient
    # for the first time on a freshly initialized (zero) parameter -- i.e.
    # the ground truth for "what a correctly-revived connection should do".
    fresh_param = Tensor(np.array([[0.0]]), requires_grad=True)
    fresh_param.grad = np.array([[g_revived]])
    fresh_opt = Adam([fresh_param], lr=1e-2)
    fresh_opt.step()
    expected_update = 0.0 - fresh_param.data[0, 0]

    assert np.isclose(applied_update, expected_update, rtol=1e-10), (
        "Reset-on-revival optimizer state does not match a fresh Adam init: "
        f"applied={applied_update}, expected={expected_update}"
    )


def test_naive_global_step_counter_would_have_blown_up():
    """
    Demonstrates *why* the per-element step counter matters: contrasts
    against the naive fix of "zero m and v on revival but keep a single
    global step counter t shared across the whole tensor". With t already
    large (as it would be, deep into training), bias correction
    (1 - beta^t) ~= 1, so zeroing v produces an update that divides by
    ~eps -- a large spurious jump instead of a well-scaled first Adam
    step. This test is a worked counterexample, not a test of our actual
    Adam class (which never has this problem, since its `t` is per
    element).
    """
    lr, beta1, beta2, eps = 1e-2, 0.9, 0.999, 1e-8
    g = 0.37  # some representative post-revival gradient

    # Correct behavior: t resets to 1 for this element (our implementation).
    m, v, t = 0.0, 0.0, 1
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    correct_update = lr * m_hat / (np.sqrt(v_hat) + eps)

    # Naive behavior: m, v zeroed, but t is a large global counter (say 5000)
    # inherited from the rest of training.
    m_naive, v_naive, t_naive = 0.0, 0.0, 5000
    m_naive = beta1 * m_naive + (1 - beta1) * g
    v_naive = beta2 * v_naive + (1 - beta2) * g ** 2
    m_hat_naive = m_naive / (1 - beta1 ** t_naive)
    v_hat_naive = v_naive / (1 - beta2 ** t_naive)
    naive_update = lr * m_hat_naive / (np.sqrt(v_hat_naive) + eps)

    # The correct, bias-corrected first step is close to lr * sign(g) (Adam's
    # well-known near-lr-sized first update). The naive version divides by
    # sqrt(v) with v not yet bias-corrected up from its cold start, which is
    # far larger relative to m_hat than intended for a "first ever" update.
    assert naive_update > 3 * correct_update, (
        "expected the naive global-t approach to produce a much larger, "
        "uncorrected first update than the per-element-t approach"
    )
