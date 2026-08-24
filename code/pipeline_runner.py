                                                                     

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from all_tasks_global_parameters import (
    Contract, content_hash, contract_by_id, load_global_parameters,
    load_task_spec, registered_contracts,
)
from base_cascade_builder import fit_cascade
from base_cascade_calibration import calibrate_cascade, fit_terminal_reference, terminal_actions
from dasr_lp_solver import fail_safe_policy, route_policy, solve_representation
from dasr_policy_index import PolicyIndex
from dasr_structure_resolution import compile_structural_representations, score_anchor_representation
from data_loader import load_task_frame, validate_dataset_schema, resolve_data_path
from policy_gate import select_policy
from report_manager import write_contract_report, write_json, write_run_index
from sealed_audit import SealedAuditReader


def run_pipeline(args: argparse.Namespace) -> list[dict[str, Any]]:
    task = load_task_spec(args.task_spec)
    global_config = load_global_parameters(args.global_spec)
    frame = load_task_frame(
        task, global_config, data_root=args.data_root, data_path=args.data_path)
    data_provenance = {
        "mode": "prepared_dataset",
        "path": str(resolve_data_path(task, args.data_root, args.data_path)),
        "rows": len(frame),
    }
    cascade = fit_cascade(frame, task, global_config)
    calibrated = calibrate_cascade(frame, cascade, global_config)
    contracts = registered_contracts(global_config) if args.contract == "all" else [
        contract_by_id(global_config, args.contract)]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "implementation": "standalone_reference",
        "task_key": task.task_key,
        "task_spec_hash": content_hash(task.raw),
        "global_spec_hash": content_hash(global_config),
        "data": data_provenance,
    }
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(run_contract, frame, calibrated, task.task_key, contract,
                        global_config, output, common): contract.contract_id
            for contract in contracts
        }
        for future in as_completed(futures):
            summaries.append(future.result())
    summaries.sort(key=lambda row: row["contract_id"])
    write_run_index(output / "RUN_REPORT.md", task.task_key, summaries)
    write_json(output / "RUN_SUMMARY.json", {"common": common, "contracts": summaries})
    return summaries


def run_contract(frame, calibrated, task_key: str, contract: Contract,
                 global_config: dict[str, Any], output: Path,
                 common: dict[str, Any]) -> dict[str, Any]:
    reference = fit_terminal_reference(frame, calibrated, contract, global_config)
    terminal = terminal_actions(calibrated, reference)
    construction = frame["split"].eq("mdp_build").to_numpy()
    score_representation = score_anchor_representation(calibrated, construction)
    anchor = solve_representation(
        frame, calibrated, terminal, score_representation, contract, global_config)
    if anchor is None:
        raise RuntimeError(f"score anchor LP is infeasible for {contract.contract_id}")
    representations = [score_representation]
    solve_cache = {score_representation.fingerprint: anchor}
    construction_signature = lambda policy: construction_route_hash(
        frame, calibrated, policy, construction)

    def exact_solver(representation):
        cached = solve_cache.get(representation.fingerprint)
        if cached is None:
            cached = solve_representation(
                frame, calibrated, terminal, representation, contract, global_config)
            if cached is not None:
                solve_cache[representation.fingerprint] = cached
        return cached

    representations.extend(compile_structural_representations(
        frame=frame,
        cascade=calibrated.cascade,
        calibrated=calibrated,
        terminal_reference=reference,
        terminal_action=terminal,
        contract=contract,
        global_config=global_config,
        exact_solver=exact_solver,
        route_signature=construction_signature,
    ))
    policies = []
    for representation in representations:
        if representation.representation_id == "score_anchor":
            policy = anchor
        else:
            policy = exact_solver(representation)
        if policy is not None:
            policies.append(policy)
    policies.append(fail_safe_policy(score_representation))
    index = PolicyIndex()
    for policy in policies:
        index.add(policy)
    frozen = index.freeze()
    selected_policy, selection, _ = select_policy(
        frame, calibrated, terminal, frozen, contract, global_config)
    audit_reader = SealedAuditReader(
        frame=frame,
        calibrated=calibrated,
        terminal_action=terminal,
        contract=contract,
        global_config=global_config,
        selected_policy_hash=selected_policy.policy_hash,
    )
    audit = audit_reader.audit(selected_policy)
    audit_reader.assert_single_access()
    contract_output = output / contract.contract_id
    index.write_manifest(contract_output / "FROZEN_POLICY_INDEX.json")
    summary = {
        **common,
        "contract_id": contract.contract_id,
        "contract": {
            "evidence_schedule": contract.evidence_schedule,
            "action_profile": contract.action_profile,
            "tier_costs": contract.tier_costs,
            "false_positive_price": contract.false_positive_price,
            "review_price": contract.review_price,
            "positive_threshold": contract.positive_threshold,
        },
        "terminal_reference": dict(reference.__dict__),
        "selection": selection.public(),
        "sealed_audit": audit.public(),
        "sealed_audit_access_log": audit_reader.access_log,
    }
    write_json(contract_output / "CONTRACT_RESULT.json", summary)
    write_contract_report(contract_output / "REPORT.md", summary)
    return summary


def schema_command(args: argparse.Namespace) -> int:
    task = load_task_spec(args.task_spec)
    source = resolve_data_path(task, args.data_root, args.data_path)
    report = validate_dataset_schema(source, task)
    print(report)
    return 0 if report["status"] == "pass" else 1


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description="A standalone reference implementation of the DASR construction, selection, audit, and governance protocol.")
    sub = out.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one task through the standalone DASR reference pipeline")
    add_common_arguments(run)
    run.add_argument("--output", required=True)
    run.add_argument("--contract", default="all",
                     help="registered contract id or 'all'")
    run.add_argument("--workers", type=int, default=1)
    schema = sub.add_parser("validate-schema", help="check a prepared dataset without loading it fully")
    add_common_arguments(schema)
    return out


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-spec", required=True)
    parser.add_argument("--global-spec", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--data-path")


def construction_route_hash(frame, calibrated, policy, construction: np.ndarray) -> str:
                                                                                
    routed = route_policy(frame, calibrated, policy, construction)
    values = routed.sort_values("row_id", kind="mergesort")[["row_id", "tier", "action"]]
    return content_hash(values.astype(str).to_dict(orient="records"))


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate-schema":
        return schema_command(args)
    summaries = run_pipeline(args)
    return 0 if all(row["sealed_audit"]["safety_pass"] for row in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
