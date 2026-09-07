from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    PrefixDataset,
    RuleMetadata,
    collate_current_graphs,
    collate_prefixes,
    source_state_counts,
)
from model_factory import build_model
from threshold_inference import (
    load_threshold_config,
    threshold_candidate_tensors,
)
from train import autocast_context, move_batch


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_totals(totals: dict[str, int]) -> dict:
    exact = totals["exact_matches"]
    eligible = totals["eligible_pairs"]
    above = totals["above_threshold_pairs"]
    capped = totals["after_source_cap_pairs"]
    decoded = totals["after_structural_decode"]
    result = dict(totals)
    result.update(
        {
            "threshold_candidate_retained_fraction": safe_ratio(above, eligible),
            "threshold_candidate_filtered_fraction": (
                1.0 - above / eligible if eligible else None
            ),
            "source_cap_candidate_retained_fraction": safe_ratio(capped, eligible),
            "decoded_candidate_retained_fraction": safe_ratio(decoded, eligible),
            "exact_above_threshold_recall": safe_ratio(
                totals["exact_above_threshold"], exact
            ),
            "exact_after_source_cap_recall": safe_ratio(
                totals["exact_after_source_cap"], exact
            ),
            "exact_final_recall": safe_ratio(totals["exact_final_retained"], exact),
            "decoded_exact_precision": safe_ratio(
                totals["exact_final_retained"], decoded
            ),
        }
    )
    return result


def empty_totals() -> dict[str, int]:
    return {
        "states": 0,
        "eligible_pairs": 0,
        "above_threshold_pairs": 0,
        "after_source_cap_pairs": 0,
        "after_structural_decode": 0,
        "exact_matches": 0,
        "exact_eligible": 0,
        "exact_above_threshold": 0,
        "exact_after_source_cap": 0,
        "exact_final_retained": 0,
        "teacher_actions": 0,
        "teacher_final_retained": 0,
        "states_with_all_exact_matches_retained": 0,
    }


def add_totals(destination: dict[str, int], source: dict[str, int]) -> None:
    for key in destination:
        destination[key] += int(source[key])


def empty_exact_totals() -> dict[str, int]:
    return {
        "exact_matches": 0,
        "exact_eligible": 0,
        "exact_above_threshold": 0,
        "exact_after_source_cap": 0,
        "exact_final_retained": 0,
    }


def summarize_exact_totals(rows: dict[str, int]) -> dict:
    return {
        **rows,
        "exact_above_threshold_recall": safe_ratio(
            rows["exact_above_threshold"], rows["exact_matches"]
        ),
        "exact_after_source_cap_recall": safe_ratio(
            rows["exact_after_source_cap"], rows["exact_matches"]
        ),
        "exact_final_recall": safe_ratio(
            rows["exact_final_retained"], rows["exact_matches"]
        ),
    }


def source_frequency_bucket(count: int) -> str:
    if count == 0:
        return "unseen"
    if count == 1:
        return "1"
    if count < 8:
        return "2-7"
    if count < 64:
        return "8-63"
    if count < 512:
        return "64-511"
    return "512+"


def prefix_training_state_count(
    trajectories: list[dict],
    *,
    include_terminal: bool,
    terminal_only_repeat: int,
) -> int:
    total = 0
    for trajectory in trajectories:
        terminal_only = bool(trajectory.get("terminal_only_supervision", False))
        if not terminal_only:
            total += len(trajectory["steps"])
        if include_terminal and "terminal_matches" in trajectory:
            total += terminal_only_repeat if terminal_only else 1
    return total


def exclude_trajectory_partition(
    trajectories: list[dict],
    *,
    modulo: int,
    remainder: int,
) -> tuple[list[dict], int]:
    if modulo < 1 or not 0 <= remainder < modulo:
        raise ValueError("trajectory partition is invalid")
    retained = [
        trajectory
        for trajectory in trajectories
        if int(trajectory["trajectory_id"]) % modulo != remainder
    ]
    return retained, len(trajectories) - len(retained)


@torch.no_grad()
def audit_recall(
    *,
    model,
    loader,
    device: torch.device,
    threshold_config: dict,
    max_source_matches: int,
    training_source_counts: list[int],
    log_every_batches: int = 0,
) -> tuple[dict, list[dict]]:
    totals = empty_totals()
    group_totals = {"near": empty_totals(), "far": empty_totals()}
    length_totals: dict[int, dict[str, int]] = {}
    frequency_totals: dict[str, dict[str, int]] = {}
    states = []
    state_index = 0
    for batch_number, cpu_batch in enumerate(loader, start=1):
        batch = move_batch(cpu_batch, device)
        with autocast_context(device):
            encoded, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(encoded, live, gate_types)

        distance = batch["current_rewrite_distance"]
        near_anchor = distance.le(2)
        near_config = threshold_config["groups"]["near"]
        far_config = threshold_config["groups"]["far"]
        scale = torch.where(
            near_anchor,
            torch.tensor(near_config["scale"], device=device),
            torch.tensor(far_config["scale"], device=device),
        )
        bias = torch.where(
            near_anchor,
            torch.tensor(near_config["bias"], device=device),
            torch.tensor(far_config["bias"], device=device),
        )
        threshold = torch.where(
            near_anchor,
            torch.tensor(near_config["raw_threshold"], device=device),
            torch.tensor(far_config["raw_threshold"], device=device),
        )
        calibrated_logits = logits.float() * scale.unsqueeze(-1) + bias.unsqueeze(-1)
        above = eligible & logits.ge(threshold.unsqueeze(-1))

        candidates = threshold_candidate_tensors(
            model,
            batch,
            logits,
            eligible,
            threshold_config,
            max_candidates_per_state=max_source_matches,
        )

        for batch_index, exact_rows in enumerate(cpu_batch["positives"]):
            exact = {
                (int(source), tuple(map(int, binding)))
                for source, binding in exact_rows
            }
            exact_by_group = {"near": set(), "far": set()}
            for source, binding in exact:
                group = (
                    "near"
                    if bool(near_anchor[batch_index, binding[0]].item())
                    else "far"
                )
                exact_by_group[group].add((source, binding))

            flat_above = above[batch_index].flatten().nonzero(
                as_tuple=False
            ).squeeze(1)
            if flat_above.numel() > max_source_matches:
                flat_scores = calibrated_logits[batch_index].flatten()[flat_above]
                flat_above = flat_above[
                    flat_scores.topk(max_source_matches).indices
                ]
            capped_pairs = {
                (
                    int(position.remainder(model.num_sources).item()),
                    int(
                        torch.div(
                            position,
                            model.num_sources,
                            rounding_mode="floor",
                        ).item()
                    ),
                )
                for position in flat_above
            }

            candidate_mask = candidates.batch_ids.eq(batch_index)
            decoded = {
                (
                    int(source),
                    tuple(int(slot) for slot in binding if int(slot) >= 0),
                )
                for source, binding in zip(
                    candidates.sources[candidate_mask].cpu().tolist(),
                    candidates.bindings[candidate_mask].cpu().tolist(),
                )
            }

            exact_eligible = {
                row
                for row in exact
                if bool(eligible[batch_index, row[1][0], row[0]].item())
            }
            exact_above = {
                row
                for row in exact
                if bool(above[batch_index, row[1][0], row[0]].item())
            }
            exact_capped = {
                row for row in exact if (row[0], row[1][0]) in capped_pairs
            }
            exact_final = exact & decoded

            for source, binding in exact:
                source_length = int(model.source_lengths[source].item())
                bucket = length_totals.setdefault(
                    source_length,
                    empty_exact_totals(),
                )
                exact_row = (source, binding)
                frequency = training_source_counts[source]
                frequency_bucket = frequency_totals.setdefault(
                    source_frequency_bucket(frequency), empty_exact_totals()
                )
                for rows in (bucket, frequency_bucket):
                    rows["exact_matches"] += 1
                    rows["exact_eligible"] += int(exact_row in exact_eligible)
                    rows["exact_above_threshold"] += int(exact_row in exact_above)
                    rows["exact_after_source_cap"] += int(exact_row in exact_capped)
                    rows["exact_final_retained"] += int(exact_row in exact_final)

            teacher = cpu_batch["target_actions"][batch_index]
            teacher_key = None
            teacher_details = None
            if teacher is not None:
                teacher_key = (
                    int(teacher["source_id"]),
                    tuple(map(int, teacher["binding_slots"])),
                )
                teacher_source, teacher_binding = teacher_key
                teacher_anchor = teacher_binding[0]
                teacher_eligible = bool(
                    eligible[
                        batch_index,
                        teacher_anchor,
                        teacher_source,
                    ].item()
                )
                teacher_raw_logit = float(
                    logits[
                        batch_index,
                        teacher_anchor,
                        teacher_source,
                    ].float().item()
                )
                teacher_calibrated_logit = float(
                    calibrated_logits[
                        batch_index,
                        teacher_anchor,
                        teacher_source,
                    ].item()
                )
                teacher_above_threshold = bool(
                    above[
                        batch_index,
                        teacher_anchor,
                        teacher_source,
                    ].item()
                )
                eligible_scores = calibrated_logits[batch_index][
                    eligible[batch_index]
                ]
                above_scores = calibrated_logits[batch_index][above[batch_index]]
                teacher_details = {
                    "xfer_id": int(teacher["xfer_id"]),
                    "source_id": teacher_source,
                    "binding_slots": list(teacher_binding),
                    "anchor_slot": teacher_anchor,
                    "group": (
                        "near"
                        if bool(near_anchor[batch_index, teacher_anchor].item())
                        else "far"
                    ),
                    "eligible": teacher_eligible,
                    "raw_logit": teacher_raw_logit,
                    "raw_threshold": float(
                        threshold[batch_index, teacher_anchor].item()
                    ),
                    "calibrated_logit": teacher_calibrated_logit,
                    "eligible_rank": (
                        1
                        + int(
                            eligible_scores.gt(teacher_calibrated_logit)
                            .sum()
                            .item()
                        )
                        if teacher_eligible
                        else None
                    ),
                    "above_threshold": teacher_above_threshold,
                    "above_threshold_rank": (
                        1
                        + int(
                            above_scores.gt(teacher_calibrated_logit).sum().item()
                        )
                        if teacher_above_threshold
                        else None
                    ),
                    "after_source_cap": (
                        teacher_source,
                        teacher_anchor,
                    )
                    in capped_pairs,
                    "final_retained": teacher_key in decoded,
                }

            row_totals = empty_totals()
            row_totals.update(
                {
                    "states": 1,
                    "eligible_pairs": int(eligible[batch_index].sum().item()),
                    "above_threshold_pairs": int(above[batch_index].sum().item()),
                    "after_source_cap_pairs": len(capped_pairs),
                    "after_structural_decode": len(decoded),
                    "exact_matches": len(exact),
                    "exact_eligible": len(exact_eligible),
                    "exact_above_threshold": len(exact_above),
                    "exact_after_source_cap": len(exact_capped),
                    "exact_final_retained": len(exact_final),
                    "teacher_actions": int(teacher_key is not None),
                    "teacher_final_retained": int(
                        teacher_key is not None and teacher_key in decoded
                    ),
                    "states_with_all_exact_matches_retained": int(
                        len(exact_final) == len(exact)
                    ),
                }
            )
            add_totals(totals, row_totals)

            state_groups = {}
            for name in ("near", "far"):
                group_exact = exact_by_group[name]
                group_above_pairs = above[batch_index] & (
                    near_anchor[batch_index].unsqueeze(-1)
                    if name == "near"
                    else ~near_anchor[batch_index].unsqueeze(-1)
                )
                group_eligible_pairs = eligible[batch_index] & (
                    near_anchor[batch_index].unsqueeze(-1)
                    if name == "near"
                    else ~near_anchor[batch_index].unsqueeze(-1)
                )
                group_row = empty_totals()
                group_row.update(
                    {
                        "states": 1,
                        "eligible_pairs": int(group_eligible_pairs.sum().item()),
                        "above_threshold_pairs": int(group_above_pairs.sum().item()),
                        "after_source_cap_pairs": sum(
                            (
                                "near"
                                if bool(near_anchor[batch_index, anchor].item())
                                else "far"
                            )
                            == name
                            for _, anchor in capped_pairs
                        ),
                        "after_structural_decode": sum(
                            (
                                "near"
                                if bool(
                                    near_anchor[batch_index, binding[0]].item()
                                )
                                else "far"
                            )
                            == name
                            for _, binding in decoded
                        ),
                        "exact_matches": len(group_exact),
                        "exact_eligible": len(group_exact & exact_eligible),
                        "exact_above_threshold": len(group_exact & exact_above),
                        "exact_after_source_cap": len(group_exact & exact_capped),
                        "exact_final_retained": len(group_exact & exact_final),
                        "teacher_actions": int(
                            teacher_key is not None and teacher_key in group_exact
                        ),
                        "teacher_final_retained": int(
                            teacher_key is not None
                            and teacher_key in group_exact
                            and teacher_key in decoded
                        ),
                        "states_with_all_exact_matches_retained": int(
                            len(group_exact & exact_final) == len(group_exact)
                        ),
                    }
                )
                add_totals(group_totals[name], group_row)
                state_groups[name] = {
                    "exact_matches": len(group_exact),
                    "exact_final_retained": len(group_exact & exact_final),
                }

            states.append(
                {
                    "state": state_index,
                    "prefix_length": int(cpu_batch["prefix_length"][batch_index]),
                    **row_totals,
                    "teacher": teacher_details,
                    "groups": state_groups,
                }
            )
            state_index += 1
        if log_every_batches and batch_number % log_every_batches == 0:
            print(
                f"audit_batch={batch_number}/{len(loader)} states={state_index}",
                flush=True,
            )

    result = summarize_totals(totals)
    result["groups"] = {
        name: summarize_totals(rows) for name, rows in group_totals.items()
    }
    result["source_gate_lengths"] = {
        str(length): summarize_exact_totals(rows)
        for length, rows in sorted(length_totals.items())
    }
    frequency_order = ("unseen", "1", "2-7", "8-63", "64-511", "512+")
    result["training_source_frequency_buckets"] = {
        name: summarize_exact_totals(frequency_totals[name])
        for name in frequency_order
        if name in frequency_totals
    }
    return result, states


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure recall of every exact Quartz source/binding on held-out "
            "trajectory states; action ranking is intentionally not evaluated."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-frequency-checkpoint",
        type=Path,
        help=(
            "reuse source-state counts from another checkpoint trained on the "
            "same data; useful when auditing an unbalanced control checkpoint"
        ),
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-recalls", type=float, nargs="+", default=(0.95, 0.99)
    )
    parser.add_argument("--max-source-matches", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=0,
        help="print audit progress every N batches; 0 disables it",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="dataset split to audit; test isolates the held-out trajectories",
    )
    parser.add_argument(
        "--exclude-calibration-trajectories",
        action="store_true",
        help=(
            "exclude the trajectory modulo/remainder partition recorded by "
            "the calibration file"
        ),
    )
    args = parser.parse_args()
    if not args.target_recalls or any(
        not 0.0 < recall <= 1.0 for recall in args.target_recalls
    ):
        parser.error("target recalls must be within (0, 1]")
    if args.max_source_matches < 1 or args.batch_size < 1:
        parser.error("source cap and batch size must be positive")
    if args.log_every_batches < 0:
        parser.error("--log-every-batches must be nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    if args.split == "train":
        trajectories = payload["train_trajectories"]
    elif args.split == "test":
        trajectories = payload["test_trajectories"]
    else:
        trajectories = payload["train_trajectories"] + payload["test_trajectories"]
    excluded_calibration_partition = None
    if args.exclude_calibration_trajectories:
        calibration_payload = json.loads(args.calibration.read_text())
        modulo = int(calibration_payload["trajectory_modulo"])
        remainder = int(calibration_payload["trajectory_remainder"])
        trajectories, excluded = exclude_trajectory_partition(
            trajectories, modulo=modulo, remainder=remainder
        )
        excluded_calibration_partition = {
            "trajectory_modulo": modulo,
            "trajectory_remainder": remainder,
            "excluded_trajectories": excluded,
        }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    architecture = train_args.get("architecture", "legacy")
    state_only = architecture == "legacy" and bool(train_args.get("state_only"))
    if architecture != "paged_action" and not state_only:
        raise ValueError(
            "generalization audit requires paged_action or legacy state-only"
        )
    dataset = PrefixDataset(trajectories, rules)
    collate = collate_current_graphs if state_only else collate_prefixes
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate(samples, rules),
        num_workers=0,
    )
    model = build_model(rules, len(rules.xfer_to_source), train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if hasattr(model, "readout_attention_backend"):
        model.readout_attention_backend = "sdpa"
    source_frequency_checkpoint = checkpoint
    if args.source_frequency_checkpoint is not None:
        source_frequency_checkpoint = torch.load(
            args.source_frequency_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        frequency_args = source_frequency_checkpoint.get("args", {})
        if str(frequency_args.get("data")) != str(train_args.get("data")):
            raise ValueError(
                "source-frequency checkpoint was trained on different data"
            )
    stored_source_balance = source_frequency_checkpoint.get("source_balance") or {}
    stored_source_counts = stored_source_balance.get(
        "positive_source_state_counts"
    )
    if stored_source_counts is not None and (
        len(stored_source_counts) == model.num_sources
    ):
        training_source_counts = list(map(int, stored_source_counts))
        training_states_for_source_frequency = stored_source_balance.get(
            "training_states"
        )
        if training_states_for_source_frequency is None and (
            str(args.data) == str(train_args.get("data"))
        ):
            training_states_for_source_frequency = prefix_training_state_count(
                payload["train_trajectories"],
                include_terminal=bool(
                    train_args.get("include_train_terminal", False)
                ),
                terminal_only_repeat=int(
                    train_args.get("train_terminal_repeat", 1)
                ),
            )
    else:
        training_dataset = PrefixDataset(
            payload["train_trajectories"],
            rules,
            include_terminal=bool(
                train_args.get("include_train_terminal", False)
            ),
            terminal_only_repeat=int(train_args.get("train_terminal_repeat", 1)),
        )
        training_source_counts = source_state_counts(
            training_dataset, model.num_sources
        ).tolist()
        training_states_for_source_frequency = len(training_dataset)
    training_source_frequency_class_counts = {
        name: sum(
            source_frequency_bucket(count) == name
            for count in training_source_counts
        )
        for name in ("unseen", "1", "2-7", "8-63", "64-511", "512+")
    }

    results = {}
    state_details = {}
    for recall in args.target_recalls:
        key = f"{recall:.4f}"
        threshold_config = load_threshold_config(args.calibration, recall)
        summary, states = audit_recall(
            model=model,
            loader=loader,
            device=device,
            threshold_config=threshold_config,
            max_source_matches=args.max_source_matches,
            training_source_counts=training_source_counts,
            log_every_batches=args.log_every_batches,
        )
        results[key] = summary
        state_details[key] = states
        print(
            f"recall={recall:.4f} exact={summary['exact_final_retained']}/"
            f"{summary['exact_matches']} teacher="
            f"{summary['teacher_final_retained']}/{summary['teacher_actions']} "
            f"states_all={summary['states_with_all_exact_matches_retained']}/"
            f"{summary['states']}",
            flush=True,
        )

    output = {
        "config": {
            "data": str(args.data),
            "checkpoint": str(args.checkpoint),
            "source_frequency_checkpoint": (
                str(args.source_frequency_checkpoint)
                if args.source_frequency_checkpoint is not None
                else str(args.checkpoint)
            ),
            "checkpoint_training_data": (
                str(train_args["data"]) if train_args.get("data") is not None else None
            ),
            "checkpoint_init": (
                str(train_args["init_checkpoint"])
                if train_args.get("init_checkpoint") is not None
                else None
            ),
            "calibration": str(args.calibration),
            "target_recalls": args.target_recalls,
            "max_source_matches": args.max_source_matches,
            "batch_size": args.batch_size,
            "device": str(device),
            "split": args.split,
            "excluded_calibration_partition": excluded_calibration_partition,
            "trajectories": len(trajectories),
            "states": len(dataset),
            "training_states_for_source_frequency": (
                training_states_for_source_frequency
            ),
            "training_source_frequency_class_counts": (
                training_source_frequency_class_counts
            ),
        },
        "results": results,
        "states": state_details,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
