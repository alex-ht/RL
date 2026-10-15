# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU tests for union top-k reverse KL and multi-teacher routing.

The brute-force union KL below is written out in this file so it cannot share
a bug with ``build_union_support``.
"""

import time

import pytest
import torch

from nemo_rl.algorithms.distillation_topk import (
    GEMMA4_FINAL_LOGIT_SOFTCAP,
    MAX_DISTILLATION_TEACHERS,
    apply_final_logit_softcap,
    build_union_support,
    group_row_indices_by_task,
    masked_support_log_softmax,
    merge_rows_in_order,
    per_token_distillation_kl,
    resident_teacher_node_count,
    schedule_teacher_topk,
    should_offload_student_for_teacher,
    validate_teacher_task_routing,
)


def _historical_per_token_kl(
    student_logprobs, teacher_logprobs, kl_type, mixed_kl_weight
):
    """Inlined pre-union DistillationLossFn reduction. Do not call the library."""
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    if kl_type == "forward":
        per_token = teacher_probs * (teacher_logprobs - student_logprobs)
    elif kl_type == "reverse":
        per_token = student_probs * (student_logprobs - teacher_logprobs)
    elif kl_type == "mixed":
        forward = teacher_probs * (teacher_logprobs - student_logprobs)
        reverse = student_probs * (student_logprobs - teacher_logprobs)
        per_token = mixed_kl_weight * forward + (1.0 - mixed_kl_weight) * reverse
    else:
        raise AssertionError(kl_type)
    return per_token.sum(dim=-1)


def _brute_union_reverse_kl(student_logits, teacher_logits, k, softcap=None):
    """Reverse KL on the union, renormalized with a Python set per position."""
    if softcap is not None:
        student_logits = softcap * torch.tanh(student_logits / softcap)
        teacher_logits = softcap * torch.tanh(teacher_logits / softcap)
    _, student_index = torch.topk(student_logits, k, dim=-1)
    teacher_values, teacher_index = torch.topk(teacher_logits, k, dim=-1)
    batch, seq_len, _ = student_logits.shape
    out = torch.empty(batch, seq_len)
    for row in range(batch):
        for pos in range(seq_len):
            order = []
            teacher_logit = {}
            for slot in range(k):
                token = int(teacher_index[row, pos, slot])
                if token not in teacher_logit:
                    order.append(token)
                    teacher_logit[token] = float(teacher_values[row, pos, slot])
            for slot in range(k):
                token = int(student_index[row, pos, slot])
                if token not in teacher_logit:
                    order.append(token)
                    teacher_logit[token] = float(teacher_logits[row, pos, token])
            student_support = torch.tensor(
                [float(student_logits[row, pos, token]) for token in order]
            )
            teacher_support = torch.tensor([teacher_logit[token] for token in order])
            student_log = torch.log_softmax(student_support, dim=-1)
            teacher_log = torch.log_softmax(teacher_support, dim=-1)
            out[row, pos] = (student_log.exp() * (student_log - teacher_log)).sum()
    return out


def _library_union_reverse_kl(student_logits, teacher_logits, k, softcap=None):
    if softcap is not None:
        student_logits = apply_final_logit_softcap(student_logits, softcap)
        teacher_logits = apply_final_logit_softcap(teacher_logits, softcap)
    _, student_index = torch.topk(student_logits, k, dim=-1)
    teacher_values, teacher_index = torch.topk(teacher_logits, k, dim=-1)
    teacher_at_student = torch.gather(teacher_logits, -1, student_index)
    union_index, union_mask, teacher_union = build_union_support(
        student_index, teacher_index, teacher_values, teacher_at_student
    )
    student_union = torch.gather(student_logits, -1, union_index)
    student_log = masked_support_log_softmax(student_union, union_mask)
    teacher_log = masked_support_log_softmax(teacher_union, union_mask)
    kl = per_token_distillation_kl(
        student_log,
        teacher_log,
        kl_type="reverse",
        mixed_kl_weight=0.0,
        support_mask=union_mask,
    )
    return kl, union_index, union_mask


def test_teacher_mode_matches_historical_kl_formula():
    torch.manual_seed(0)
    student_log = torch.log_softmax(torch.randn(3, 4, 5), dim=-1)
    teacher_log = torch.log_softmax(torch.randn(3, 4, 5), dim=-1)
    for kl_type, weight in (("forward", 0.5), ("reverse", 0.5), ("mixed", 0.3)):
        got = per_token_distillation_kl(
            student_log,
            teacher_log,
            kl_type=kl_type,
            mixed_kl_weight=weight,
            support_mask=None,
        )
        expected = _historical_per_token_kl(student_log, teacher_log, kl_type, weight)
        assert torch.allclose(got, expected)


def test_union_reverse_kl_matches_brute_force_and_renormalizes():
    torch.manual_seed(1)
    student_logits = torch.randn(2, 3, 6)
    teacher_logits = torch.randn(2, 3, 6)
    k = 2
    got, union_index, union_mask = _library_union_reverse_kl(
        student_logits, teacher_logits, k, softcap=GEMMA4_FINAL_LOGIT_SOFTCAP
    )
    expected = _brute_union_reverse_kl(
        student_logits, teacher_logits, k, softcap=GEMMA4_FINAL_LOGIT_SOFTCAP
    )
    assert torch.allclose(got, expected, atol=1e-5)
    assert union_mask.any()
    assert int(union_mask.sum(dim=-1).max()) <= 2 * k

    # Full-vocab probabilities restricted to the union are a different estimator.
    capped_student = apply_final_logit_softcap(
        student_logits, GEMMA4_FINAL_LOGIT_SOFTCAP
    )
    capped_teacher = apply_final_logit_softcap(
        teacher_logits, GEMMA4_FINAL_LOGIT_SOFTCAP
    )
    full_student = torch.log_softmax(capped_student, dim=-1)
    full_teacher = torch.log_softmax(capped_teacher, dim=-1)
    raw_student = torch.gather(full_student, -1, union_index)
    raw_teacher = torch.gather(full_teacher, -1, union_index)
    diff = torch.where(
        union_mask, raw_student - raw_teacher, torch.zeros_like(raw_student)
    )
    restricted = (raw_student.exp() * diff).sum(dim=-1)
    assert not torch.allclose(got, restricted, atol=1e-4)


def test_union_keeps_teacher_logit_on_overlap_and_pads():
    student_index = torch.tensor([[[1, 2]]])
    teacher_index = torch.tensor([[[2, 3]]])
    teacher_topk = torch.tensor([[[10.0, 11.0]]])
    teacher_at_student = torch.tensor([[[20.0, 21.0]]])
    union_index, union_mask, teacher_union = build_union_support(
        student_index, teacher_index, teacher_topk, teacher_at_student
    )
    assert union_mask.all()
    ids = union_index[0, 0].tolist()
    logits = teacher_union[0, 0].tolist()
    assert ids == [1, 2, 3]
    assert logits[ids.index(2)] == pytest.approx(10.0)
    assert logits[ids.index(1)] == pytest.approx(20.0)

    # A second row whose union is smaller gets padded, and the pad is not a real member.
    student_index = torch.tensor([[[1, 2]], [[4, 4]]])
    teacher_index = torch.tensor([[[2, 3]], [[4, 4]]])
    teacher_topk = torch.tensor([[[10.0, 11.0]], [[1.0, 1.0]]])
    teacher_at_student = torch.tensor([[[20.0, 21.0]], [[1.0, 1.0]]])
    _, union_mask, _ = build_union_support(
        student_index, teacher_index, teacher_topk, teacher_at_student
    )
    assert union_mask[0, 0].sum() == 3
    assert union_mask[1, 0].sum() == 1
    assert not union_mask[1, 0, -1]


def test_masked_union_kl_is_finite():
    logits = torch.tensor([[[1.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False]]])
    logprobs = masked_support_log_softmax(logits, mask)
    assert torch.isneginf(logprobs[0, 0, 2])
    kl = per_token_distillation_kl(
        logprobs,
        logprobs,
        kl_type="reverse",
        mixed_kl_weight=0.0,
        support_mask=mask,
    )
    assert torch.isfinite(kl).all()
    assert kl.item() == pytest.approx(0.0)


def test_softcap_is_gemma4_formula():
    logits = torch.tensor([-100.0, 0.0, 30.0, 90.0])
    capped = apply_final_logit_softcap(logits, GEMMA4_FINAL_LOGIT_SOFTCAP)
    expected = GEMMA4_FINAL_LOGIT_SOFTCAP * torch.tanh(
        logits / GEMMA4_FINAL_LOGIT_SOFTCAP
    )
    assert torch.allclose(capped, expected)
    assert apply_final_logit_softcap(logits, None) is logits
    with pytest.raises(ValueError, match="positive"):
        apply_final_logit_softcap(logits, 0.0)


def test_teacher_routing_limits_and_duplicates():
    routing = validate_teacher_task_routing(
        {"math": ["DAPOMath17K"], "code": ["OpenMathInstruct-2", "DeepScaler"]}
    )
    assert routing["DAPOMath17K"] == "math"
    assert routing["DeepScaler"] == "code"
    with pytest.raises(ValueError, match="claimed"):
        validate_teacher_task_routing({"a": ["math"], "b": ["math"]})
    with pytest.raises(ValueError, match="empty"):
        validate_teacher_task_routing({})
    with pytest.raises(ValueError, match="no task_names"):
        validate_teacher_task_routing({"a": []})
    too_many = {f"t{i}": [f"task{i}"] for i in range(MAX_DISTILLATION_TEACHERS + 1)}
    with pytest.raises(ValueError, match="At most"):
        validate_teacher_task_routing(too_many)


def test_group_and_merge_restore_original_order():
    routing = {"math": "math_teacher", "code": "code_teacher"}
    grouped = group_row_indices_by_task(["code", "math", "code", "math"], routing)
    assert grouped == {"math_teacher": [1, 3], "code_teacher": [0, 2]}
    merged = merge_rows_in_order(
        4,
        grouped,
        {
            "math_teacher": {"marker": torch.tensor([[11.0], [13.0]])},
            "code_teacher": {"marker": torch.tensor([[10.0], [12.0]])},
        },
        alias_to_id={"math_teacher": 0, "code_teacher": 1},
    )
    assert merged["marker"].view(-1).tolist() == [10.0, 11.0, 12.0, 13.0]
    assert merged["teacher_ids"].tolist() == [1, 0, 1, 0]
    with pytest.raises(ValueError, match="more than one"):
        merge_rows_in_order(
            2,
            {"a": [0], "b": [0, 1]},
            {
                "a": {"marker": torch.tensor([[1.0]])},
                "b": {"marker": torch.tensor([[2.0], [3.0]])},
            },
            alias_to_id={"a": 0, "b": 1},
        )
    with pytest.raises(ValueError, match="not routed"):
        merge_rows_in_order(
            3,
            {"a": [0, 2]},
            {"a": {"marker": torch.tensor([[1.0], [2.0]])}},
            alias_to_id={"a": 0},
        )
    with pytest.raises(ValueError, match="No teacher"):
        group_row_indices_by_task(["unknown"], routing)


def test_schedule_submits_every_teacher_before_collecting():
    events = []

    class _FakePolicy:
        def __init__(self, name):
            self.name = name

        def submit_topk_logits(self, subset, k, timer=None):
            del subset, k, timer
            events.append(("submit", self.name))
            return self.name

        def collect_topk_logits(self, handle):
            events.append(("collect", handle))
            time.sleep(0.01)
            value = 1.0 if handle == "a" else 2.0
            return {"topk_logits": torch.tensor([[value]])}

    outputs, elapsed = schedule_teacher_topk(
        {"a": _FakePolicy("a"), "b": _FakePolicy("b")},
        {"a": [0], "b": [1], "c": []},
        lambda data, indices: [data[i] for i in indices],
        ["row0", "row1"],
        k=4,
    )
    submits = [name for kind, name in events if kind == "submit"]
    first_collect = next(i for i, (kind, _) in enumerate(events) if kind == "collect")
    assert submits == ["a", "b"]
    assert all(kind == "submit" for kind, _ in events[:first_collect])
    assert set(outputs) == {"a", "b"}
    assert elapsed["a"] >= 0.0 and elapsed["b"] >= 0.0


def test_resident_teacher_budget_and_offload():
    assert resident_teacher_node_count([4, 4, 4, 4], 8) == 2
    with pytest.raises(ValueError, match="whole number"):
        resident_teacher_node_count([4, 4, 4], 8)
    assert (
        should_offload_student_for_teacher(
            resident_teachers=True, colocated_inference=False
        )
        is False
    )
    assert (
        should_offload_student_for_teacher(
            resident_teachers=False, colocated_inference=False
        )
        is True
    )
    assert (
        should_offload_student_for_teacher(
            resident_teachers=False, colocated_inference=True
        )
        is False
    )
