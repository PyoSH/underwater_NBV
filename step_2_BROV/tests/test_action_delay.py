"""envs/action_delay.py 단위시험 — 순수 torch, Isaac 불필요.

DELAY_TRAINING_PLAN.md §4 "구현 시 지켜야 할 것"이 지목한 세 가지를 검사한다:
링버퍼 지연 정확성, 부분 reset 초기화, 이력 내용이 실행 행동과 일치.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.action_delay import (  # noqa: E402
    ActionDelayBuffer,
    ActionDelayConfig,
    ActionHistoryBuffer,
    ObservationStalenessModel,
)


# ── ActionDelayConfig ────────────────────────────────────────────────────────


def test_delay_step_range_matches_plan():
    """40~80 ms / 물리 10 ms = 4~8 스텝 (계획서 §2 공통)."""
    config = ActionDelayConfig(delay_ms_range=(40.0, 80.0), physics_dt_s=0.01)
    assert config.delay_step_range == (4, 8)


def test_delay_config_rejects_invalid_range():
    with pytest.raises(ValueError):
        ActionDelayConfig(delay_ms_range=(80.0, 40.0))
    with pytest.raises(ValueError):
        ActionDelayConfig(physics_dt_s=0.0)


# ── ActionDelayBuffer ────────────────────────────────────────────────────────


@pytest.mark.parametrize("delay_ms", [40.0, 50.0, 60.0, 70.0, 80.0])
def test_ring_buffer_delivers_exactly_d_steps_late(delay_ms):
    """출력[k] == 입력[k-d], 그 이전은 0(중립)."""
    config = ActionDelayConfig(delay_ms_range=(delay_ms, delay_ms), physics_dt_s=0.01)
    buffer = ActionDelayBuffer(3, 6, seed=0, config=config)
    buffer.reset()
    d = config.delay_step_range[0]
    assert int(buffer.delay_steps.min()) == d == int(buffer.delay_steps.max())

    inputs = []
    outputs = []
    for k in range(40):
        # 각 스텝을 유일한 값으로 표시해 어긋남이 바로 드러나게 한다.
        action = torch.full((3, 6), float(k + 1))
        inputs.append(action)
        outputs.append(buffer.step(action))

    for k in range(40):
        expected = torch.zeros(3, 6) if k < d else inputs[k - d]
        assert torch.equal(outputs[k], expected), f"step {k}"


def test_per_episode_delay_is_within_range_and_constant():
    config = ActionDelayConfig(delay_ms_range=(40.0, 80.0), physics_dt_s=0.01)
    buffer = ActionDelayBuffer(512, 6, seed=7, config=config)
    buffer.reset()
    delays = buffer.delay_steps.clone()
    assert int(delays.min()) >= 4 and int(delays.max()) <= 8
    # 실제로 범위 전체를 쓰는지(상수로 굳지 않았는지)
    assert delays.unique().numel() == 5
    for _ in range(20):
        buffer.step(torch.rand(512, 6))
    # 에피소드 내내 고정
    assert torch.equal(buffer.delay_steps, delays)


def test_partial_reset_only_clears_selected_envs():
    """부분 reset: 고른 env 만 버퍼가 비고 나머지 이력은 그대로 흘러야 한다."""
    config = ActionDelayConfig(delay_ms_range=(40.0, 40.0), physics_dt_s=0.01)
    buffer = ActionDelayBuffer(4, 2, seed=1, config=config)
    buffer.reset()

    for k in range(10):
        buffer.step(torch.full((4, 2), float(k + 1)))
    # k=9 까지 넣었으니 다음 출력은 입력 k=6 (=7.0)
    buffer.reset(torch.tensor([1, 3]))
    out = buffer.step(torch.full((4, 2), 99.0))
    assert torch.equal(out[0], torch.full((2,), 7.0))
    assert torch.equal(out[2], torch.full((2,), 7.0))
    # 리셋된 env 는 유령 행동 없이 0 부터 시작
    assert torch.equal(out[1], torch.zeros(2))
    assert torch.equal(out[3], torch.zeros(2))


def test_reset_resamples_only_selected_envs():
    config = ActionDelayConfig(delay_ms_range=(40.0, 80.0), physics_dt_s=0.01)
    buffer = ActionDelayBuffer(64, 6, seed=3, config=config)
    buffer.reset()
    before = buffer.delay_steps.clone()
    ids = torch.arange(0, 64, 2)
    buffer.reset(ids)
    after = buffer.delay_steps
    keep = torch.arange(1, 64, 2)
    assert torch.equal(before[keep], after[keep])


def test_step_rejects_wrong_shape():
    buffer = ActionDelayBuffer(2, 6)
    buffer.reset()
    with pytest.raises(ValueError):
        buffer.step(torch.zeros(2, 3))


# ── ActionHistoryBuffer ──────────────────────────────────────────────────────


def test_history_contains_executed_actions_most_recent_first():
    """이력 = 실행 행동 그대로, index 0 이 가장 최근."""
    history = ActionHistoryBuffer(2, 2, 6)
    history.reset()
    assert torch.equal(history.as_observation(), torch.zeros(2, 12))

    a1 = torch.randn(2, 6)
    a2 = torch.randn(2, 6)
    a3 = torch.randn(2, 6)
    history.push(a1)
    assert torch.equal(history.as_observation(), torch.cat([a1, torch.zeros(2, 6)], -1))
    history.push(a2)
    assert torch.equal(history.as_observation(), torch.cat([a2, a1], -1))
    history.push(a3)
    assert torch.equal(history.as_observation(), torch.cat([a3, a2], -1))


def test_history_partial_reset():
    history = ActionHistoryBuffer(3, 2, 2)
    history.reset()
    history.push(torch.ones(3, 2))
    history.push(2 * torch.ones(3, 2))
    history.reset(torch.tensor([1]))
    obs = history.as_observation()
    assert torch.equal(obs[0], torch.tensor([2.0, 2.0, 1.0, 1.0]))
    assert torch.equal(obs[1], torch.zeros(4))
    assert torch.equal(obs[2], torch.tensor([2.0, 2.0, 1.0, 1.0]))


def test_history_length_one_is_supported():
    history = ActionHistoryBuffer(1, 1, 3)
    history.reset()
    a = torch.tensor([[1.0, 2.0, 3.0]])
    history.push(a)
    assert torch.equal(history.as_observation(), a)


# ── ObservationStalenessModel ────────────────────────────────────────────────


def test_first_tick_after_reset_is_always_fresh():
    model = ObservationStalenessModel(256, 16, 1.0, seed=0)
    model.reset()
    obs = torch.randn(256, 16)
    assert not model.draw_stale_mask().any()
    assert torch.equal(model.apply(obs, model.draw_stale_mask()), obs)


def test_stale_tick_republishes_previous_observation():
    model = ObservationStalenessModel(4, 3, 1.0, seed=0)
    model.reset()
    first = torch.randn(4, 3)
    model.apply(first, model.draw_stale_mask())
    second = torch.randn(4, 3)
    mask = model.draw_stale_mask()
    assert mask.all()          # p=1.0
    published = model.apply(second, mask)
    assert torch.equal(published, first)
    # 연속 묵음이면 같은 값을 계속 발행
    published2 = model.apply(torch.randn(4, 3), model.draw_stale_mask())
    assert torch.equal(published2, first)


def test_stale_probability_is_approximately_respected():
    model = ObservationStalenessModel(20000, 4, 0.15, seed=11)
    model.reset()
    obs = torch.zeros(20000, 4)
    model.apply(obs, model.draw_stale_mask())    # 첫 틱 소진
    rates = []
    for _ in range(10):
        mask = model.draw_stale_mask()
        rates.append(float(mask.float().mean()))
        model.apply(torch.randn(20000, 4), mask)
    mean_rate = sum(rates) / len(rates)
    assert 0.14 < mean_rate < 0.16


def test_zero_probability_never_goes_stale():
    model = ObservationStalenessModel(32, 5, 0.0, seed=0)
    model.reset()
    for _ in range(20):
        mask = model.draw_stale_mask()
        assert not mask.any()
