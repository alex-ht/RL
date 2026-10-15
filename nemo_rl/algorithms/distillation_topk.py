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
"""Top-k union reverse KL and multi-teacher routing for on-policy distillation.

Renormalization
---------------
Union mode computes reverse KL on the union of the student top-k and teacher
top-k token ids at each position. Both distributions are ``log_softmax``'d on
that union only: padded slots are ``-inf`` and drop out of the normalizer, so
the result is a KL between two distributions supported on the union.

That is the same estimator the historical teacher-only path uses when
``zero_outside_topk`` is false (log-softmax inside the provided index set,
not full-vocabulary probabilities restricted to the set). Full-vocabulary mass
outside the union is discarded on both sides. The tail correction
(``zero_outside_topk: true``) is rejected in union mode because it is defined
for the complement of the teacher top-k, not the union.

Gemma-4 final logit softcapping
-------------------------------
Gemma-4 applies ``softcap * tanh(logits / softcap)`` with ``softcap=30`` inside
the model forward (Hugging Face / Automodel and Megatron). Top-k is unchanged
by that map because it is strictly increasing, and applying it to the gathered
union logits equals gathering after the full-vocab softcap. Set
``loss_fn.final_logit_softcapping`` only when the backend returns raw LM-head
logits. Leave it unset when the forward already softcapped, or the cap is
applied twice.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch

# Gemma-4 ``final_logit_softcapping`` (text config / Megatron output layer).
GEMMA4_FINAL_LOGIT_SOFTCAP = 30.0

MAX_DISTILLATION_TEACHERS = 4


def apply_final_logit_softcap(
    logits: torch.Tensor, softcap: Optional[float]
) -> torch.Tensor:
    """Apply Gemma-style final logit softcapping.

    Args:
        logits: Pre-softmax logits, any shape.
        softcap: Positive cap. ``None`` returns ``logits`` unchanged.

    Returns:
        ``softcap * tanh(logits / softcap)`` when ``softcap`` is set.

    Raises:
        ValueError: If ``softcap`` is not positive.
    """
    if softcap is None:
        return logits
    if softcap <= 0:
        raise ValueError(f"final_logit_softcapping must be positive, got {softcap}.")
    return softcap * torch.tanh(logits / softcap)


def per_token_distillation_kl(
    student_topk_logprobs: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    *,
    kl_type: str,
    mixed_kl_weight: float,
    support_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-position KL on an aligned top-k (or union) support.

    Args:
        student_topk_logprobs: ``[B, S, K]`` student log-probabilities.
        teacher_topk_logprobs: ``[B, S, K]`` teacher log-probabilities.
        kl_type: ``forward``, ``reverse``, or ``mixed``.
        mixed_kl_weight: Weight on the forward term when ``kl_type`` is mixed.
        support_mask: Optional ``[B, S, K]`` bool mask. ``False`` slots
            contribute 0. Required when padded slots are ``-inf``, because
            ``0 * (-inf - -inf)`` is NaN. ``None`` keeps the historical
            dense top-k reduction.

    Returns:
        Per-position KL, shape ``[B, S]``.
    """
    student_probs = student_topk_logprobs.exp()
    teacher_probs = teacher_topk_logprobs.exp()
    if kl_type == "forward":
        per_token_kl = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
    elif kl_type == "reverse":
        per_token_kl = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
    elif kl_type == "mixed":
        kl_forward = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
        kl_reverse = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
        per_token_kl = (
            mixed_kl_weight * kl_forward + (1.0 - mixed_kl_weight) * kl_reverse
        )
    else:
        raise ValueError(f"Invalid KL type: {kl_type}")

    if support_mask is not None:
        # Recompute from a masked difference so invalid -inf slots cannot NaN.
        valid = support_mask.to(device=per_token_kl.device)
        if kl_type == "forward":
            diff = teacher_topk_logprobs - student_topk_logprobs
            diff = torch.where(valid, diff, torch.zeros_like(diff))
            per_token_kl = teacher_probs * diff
        elif kl_type == "reverse":
            diff = student_topk_logprobs - teacher_topk_logprobs
            diff = torch.where(valid, diff, torch.zeros_like(diff))
            per_token_kl = student_probs * diff
        else:
            per_token_kl = torch.where(
                valid, per_token_kl, torch.zeros_like(per_token_kl)
            )
    return per_token_kl.sum(dim=-1)


def build_union_support(
    student_indices: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_logits_at_student: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-position union of student and teacher top-k ids.

    Overlapping ids keep the teacher top-k logit (the two forwards agree; the
    teacher value is the one stored for the teacher distribution). The union
    is packed to the left and padded to the widest union in the batch.

    Args:
        student_indices: ``[..., k]`` student top-k token ids.
        teacher_indices: ``[..., k]`` teacher top-k token ids.
        teacher_topk_logits: Teacher logits at ``teacher_indices``.
        teacher_logits_at_student: Teacher logits at ``student_indices``.

    Returns:
        Tuple ``(union_indices, union_mask, teacher_union_logits)`` with last
        dimension ``U <= 2k``. ``union_mask`` is true on real union members.

    Raises:
        ValueError: If shapes disagree or ``k`` is 0.
    """
    if student_indices.shape != teacher_indices.shape:
        raise ValueError(
            "student and teacher top-k indices must share a shape, got "
            f"{tuple(student_indices.shape)} and {tuple(teacher_indices.shape)}."
        )
    if teacher_topk_logits.shape != teacher_indices.shape:
        raise ValueError(
            "teacher_topk_logits must match teacher_indices, got "
            f"{tuple(teacher_topk_logits.shape)} and {tuple(teacher_indices.shape)}."
        )
    if teacher_logits_at_student.shape != student_indices.shape:
        raise ValueError(
            "teacher_logits_at_student must match student_indices, got "
            f"{tuple(teacher_logits_at_student.shape)} and {tuple(student_indices.shape)}."
        )
    k = int(student_indices.shape[-1])
    if k <= 0:
        raise ValueError(f"topk must be positive, got {k}.")

    leading = student_indices.shape[:-1]
    combined_idx = torch.cat([teacher_indices.long(), student_indices.long()], dim=-1)
    combined_logits = torch.cat(
        [
            teacher_topk_logits,
            teacher_logits_at_student.to(dtype=teacher_topk_logits.dtype),
        ],
        dim=-1,
    )
    flat_idx = combined_idx.reshape(-1, 2 * k)
    flat_logits = combined_logits.reshape(-1, 2 * k)
    # Stable sort keeps the teacher logit first inside an id tie because
    # teacher ids occupy the first half of ``combined_idx``.
    order = torch.argsort(flat_idx, dim=-1, stable=True)
    sorted_idx = torch.gather(flat_idx, -1, order)
    sorted_logits = torch.gather(flat_logits, -1, order)
    duplicate = torch.zeros_like(sorted_idx, dtype=torch.bool)
    duplicate[:, 1:] = sorted_idx[:, 1:] == sorted_idx[:, :-1]
    valid = ~duplicate
    rank = valid.to(dtype=torch.long).cumsum(dim=-1) - 1
    union_width = int(valid.sum(dim=-1).max().item())
    if union_width <= 0:
        raise ValueError("union support is empty.")

    rows = flat_idx.shape[0]
    out_idx = torch.zeros(
        rows, union_width, dtype=sorted_idx.dtype, device=sorted_idx.device
    )
    out_logits = torch.zeros(
        rows, union_width, dtype=sorted_logits.dtype, device=sorted_logits.device
    )
    out_mask = torch.zeros(
        rows, union_width, dtype=torch.bool, device=sorted_idx.device
    )
    row_ids = torch.arange(rows, device=sorted_idx.device)
    for col in range(2 * k):
        take = valid[:, col]
        if not bool(take.any()):
            continue
        dest = rank[:, col]
        out_idx[row_ids[take], dest[take]] = sorted_idx[take, col]
        out_logits[row_ids[take], dest[take]] = sorted_logits[take, col]
        out_mask[row_ids[take], dest[take]] = True

    return (
        out_idx.reshape(*leading, union_width),
        out_mask.reshape(*leading, union_width),
        out_logits.reshape(*leading, union_width),
    )


def masked_support_log_softmax(
    logits: torch.Tensor, support_mask: torch.Tensor
) -> torch.Tensor:
    """``log_softmax`` on ``support_mask``, with invalid slots set to ``-inf``."""
    masked = logits.masked_fill(~support_mask.to(device=logits.device), float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def validate_teacher_task_routing(
    task_names_by_alias: Mapping[str, list[str]],
) -> dict[str, str]:
    """Validate the resident-teacher map and return ``task_name -> alias``.

    Args:
        task_names_by_alias: Teacher alias to the ``DatumSpec.task_name`` values
            it scores.

    Returns:
        Routing table from task name to teacher alias.

    Raises:
        ValueError: If there are no teachers, more than four, an empty task
            list, or two teachers claim the same task.
    """
    if not task_names_by_alias:
        raise ValueError("teachers map is empty.")
    if len(task_names_by_alias) > MAX_DISTILLATION_TEACHERS:
        raise ValueError(
            f"At most {MAX_DISTILLATION_TEACHERS} resident teachers are supported, "
            f"got {len(task_names_by_alias)}."
        )
    task_to_alias: dict[str, str] = {}
    for alias, task_names in task_names_by_alias.items():
        if not task_names:
            raise ValueError(f"Teacher {alias!r} has no task_names.")
        for task_name in task_names:
            owner = task_to_alias.get(task_name)
            if owner is not None:
                raise ValueError(
                    f"task_name {task_name!r} is claimed by teachers {owner!r} and {alias!r}."
                )
            task_to_alias[task_name] = alias
    return task_to_alias


def resident_teacher_node_count(
    teacher_gpu_counts: list[int], cluster_gpus_per_node: int
) -> int:
    """Whole cluster nodes consumed by resident teachers.

    Partial-GPU teachers (for example 4x EP4 on 8-GPU nodes) are packed into
    whole nodes. The total must divide the cluster width so the student and
    vLLM budgets stay on full nodes.

    Args:
        teacher_gpu_counts: GPU count of each teacher (``num_nodes * gpus_per_node``).
        cluster_gpus_per_node: GPUs on one physical node.

    Returns:
        Number of cluster nodes to subtract from the training budget.

    Raises:
        ValueError: If the GPU total is not a positive multiple of the node width.
    """
    if cluster_gpus_per_node <= 0:
        raise ValueError(
            f"cluster.gpus_per_node must be positive, got {cluster_gpus_per_node}."
        )
    total = sum(teacher_gpu_counts)
    if total <= 0:
        raise ValueError("Resident teachers must request at least one GPU.")
    if total % cluster_gpus_per_node != 0:
        raise ValueError(
            f"Resident teachers use {total} GPUs, which is not a whole number of "
            f"cluster nodes ({cluster_gpus_per_node} GPUs/node)."
        )
    return total // cluster_gpus_per_node


def should_offload_student_for_teacher(
    *, resident_teachers: bool, colocated_inference: bool
) -> bool:
    """Whether the student optimizer must be offloaded before teacher inference.

    A legacy teacher shares the training GPUs, and non-colocated vLLM no longer
    offloads the optimizer on the refit path, so the distillation loop offloads
    it before the teacher forward. Resident teachers sit on their own clusters,
    so the student optimizer stays on the training GPUs.
    """
    return (not resident_teachers) and (not colocated_inference)


def group_row_indices_by_task(
    task_names: list[Optional[str]], task_to_alias: Mapping[str, str]
) -> dict[str, list[int]]:
    """Route batch rows to teacher aliases by ``DatumSpec.task_name``.

    Args:
        task_names: Per-row task name, original batch order.
        task_to_alias: Routing table from :func:`validate_teacher_task_routing`.

    Returns:
        Alias to row indices. Aliases with no rows in this batch map to ``[]``.

    Raises:
        ValueError: If a row's task name is not in the routing table.
    """
    grouped: dict[str, list[int]] = {
        alias: [] for alias in dict.fromkeys(task_to_alias.values())
    }
    for row, task_name in enumerate(task_names):
        alias = task_to_alias.get(task_name) if task_name is not None else None
        if alias is None:
            raise ValueError(
                f"No teacher for task_name={task_name!r} at row {row}. "
                f"Known task names: {sorted(task_to_alias)}."
            )
        grouped[alias].append(row)
    return grouped


def merge_rows_in_order(
    batch_size: int,
    row_indices_by_alias: Mapping[str, list[int]],
    tensors_by_alias: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    alias_to_id: Mapping[str, int],
) -> dict[str, torch.Tensor]:
    """Scatter per-teacher rows back to original batch order.

    Every row must be routed exactly once. Tensor keys must match across the
    teachers that received rows. Sequence and top-k dimensions must already
    agree (callers pad the union axis before merging if widths differ).

    Args:
        batch_size: Original batch length.
        row_indices_by_alias: Row indices each teacher scored.
        tensors_by_alias: Per-teacher tensors shaped ``[n_rows, ...]``.
        alias_to_id: Stable integer id per alias, written to ``teacher_ids``.

    Returns:
        Merged tensors plus ``teacher_ids`` of shape ``[batch_size]``.

    Raises:
        ValueError: If rows are missing, duplicated, or tensor layouts differ.
    """
    active = {
        alias: list(indices)
        for alias, indices in row_indices_by_alias.items()
        if indices
    }
    if not active:
        raise ValueError("No teacher rows to merge.")
    keys: Optional[set[str]] = None
    for alias in active:
        alias_keys = set(tensors_by_alias[alias])
        if keys is None:
            keys = alias_keys
        elif alias_keys != keys:
            raise ValueError(
                f"Teacher {alias!r} returned keys {sorted(alias_keys)}, expected {sorted(keys)}."
            )
    if not keys:
        raise ValueError("Teacher outputs are empty.")

    seen = [-1] * batch_size
    for alias, indices in active.items():
        teacher_id = alias_to_id[alias]
        for row in indices:
            if row < 0 or row >= batch_size:
                raise ValueError(f"Row index {row} is outside batch size {batch_size}.")
            if seen[row] != -1:
                raise ValueError(f"Row {row} was routed to more than one teacher.")
            seen[row] = teacher_id
    missing = [row for row, teacher_id in enumerate(seen) if teacher_id < 0]
    if missing:
        raise ValueError(f"Rows were not routed to a teacher: {missing[:8]}.")

    merged: dict[str, torch.Tensor] = {
        "teacher_ids": torch.tensor(seen, dtype=torch.long)
    }
    for key in keys:
        template = tensors_by_alias[next(iter(active))][key]
        acc = template.new_empty((batch_size, *template.shape[1:]))
        for alias, indices in active.items():
            src = tensors_by_alias[alias][key]
            if src.shape[1:] != template.shape[1:]:
                raise ValueError(
                    f"Teacher {alias!r} key {key!r} has shape {tuple(src.shape)}, "
                    f"expected leading-dim only to differ from {tuple(template.shape)}."
                )
            if src.shape[0] != len(indices):
                raise ValueError(
                    f"Teacher {alias!r} key {key!r} has {src.shape[0]} rows, "
                    f"but {len(indices)} were routed."
                )
            acc[indices] = src
        merged[key] = acc
    return merged


def schedule_teacher_topk(
    policies: Mapping[str, Any],
    row_indices_by_alias: Mapping[str, list[int]],
    select_rows: Callable[[Any, list[int]], Any],
    data: Any,
    k: int,
    timer: Any = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Submit every non-empty teacher top-k, then collect them.

    Submits are issued on the caller thread before any collect starts, so the
    teacher forwards overlap. Elapsed seconds are measured from that teacher's
    submit to its collect and are safe to log as per-teacher time.

    Args:
        policies: Alias to an object with ``submit_topk_logits`` and
            ``collect_topk_logits``.
        row_indices_by_alias: Rows for each alias. Empty aliases are skipped.
        select_rows: Slice ``data`` down to a list of row indices.
        data: Full batch the teachers score.
        k: Top-k width.
        timer: Optional timer forwarded to ``submit_topk_logits``.

    Returns:
        ``(outputs_by_alias, elapsed_seconds_by_alias)``.
    """
    jobs = [
        (alias, list(indices))
        for alias, indices in row_indices_by_alias.items()
        if indices
    ]
    if not jobs:
        raise ValueError("schedule_teacher_topk received no rows.")

    submit_started: dict[str, float] = {}
    handles: list[tuple[str, Any]] = []
    for alias, indices in jobs:
        subset = select_rows(data, indices)
        submit_started[alias] = time.perf_counter()
        handle = policies[alias].submit_topk_logits(subset, k, timer=timer)
        handles.append((alias, handle))

    def _collect(alias: str, handle: Any) -> tuple[str, Any, float]:
        output = policies[alias].collect_topk_logits(handle)
        elapsed = time.perf_counter() - submit_started[alias]
        return alias, output, elapsed

    outputs: dict[str, Any] = {}
    elapsed_by_alias: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(handles)) as executor:
        futures = [
            executor.submit(_collect, alias, handle) for alias, handle in handles
        ]
        for future in futures:
            alias, output, elapsed = future.result()
            outputs[alias] = output
            elapsed_by_alias[alias] = elapsed
    return outputs, elapsed_by_alias
