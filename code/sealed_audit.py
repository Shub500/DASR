                                                                            

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from all_tasks_global_parameters import Contract
from base_cascade_calibration import CalibratedCascade
from dasr_lp_solver import Policy
from policy_gate import PolicyEvaluation, evaluate_policy


@dataclass
class SealedAuditReader:
    frame: pd.DataFrame
    calibrated: CalibratedCascade
    terminal_action: np.ndarray
    contract: Contract
    global_config: dict[str, Any]
    selected_policy_hash: str
    access_log: list[dict[str, Any]] = field(default_factory=list)

    def audit(self, policy: Policy) -> PolicyEvaluation:
        if policy.policy_hash != self.selected_policy_hash:
            raise PermissionError("sealed audit may be opened only for the selected policy")
        if self.access_log:
            raise PermissionError("sealed audit has already been opened for this contract")
        evaluation = evaluate_policy(
            self.frame, self.calibrated, self.terminal_action, policy,
            self.contract, self.global_config, split="report_test",
            cap=float(self.global_config["audit"]["report_cap"]))
        self.access_log.append({
            "policy_hash": policy.policy_hash,
            "split": "report_test",
            "checked_cap": evaluation.checked_cap,
            "safety_pass": evaluation.safety_pass})
        return evaluation

    def assert_single_access(self) -> None:
        if len(self.access_log) != 1:
            raise RuntimeError(
                f"expected exactly one sealed-audit access; found {len(self.access_log)}")
