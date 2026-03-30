from __future__ import annotations

from typing import List

from ..ehr.models import DataProfile
from ..ehr.tasking import default_tool_bundle


class ToolBroker:
    """
    Keep tool exposure small and task-specific instead of dumping a large tool list
    into the executor prompt.
    """

    def select_bundle(self, task_family: str, data_profile: DataProfile) -> List[str]:
        bundle = list(default_tool_bundle(task_family))
        if "structured_tabular" in data_profile.modalities:
            bundle.append("tabular inspection")
        if "clinical_text" in data_profile.modalities:
            bundle.append("text parsing")
        if data_profile.patient_id_columns:
            bundle.append("patient-level split audit")
        if data_profile.target_columns or task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling"}:
            bundle.append("leakage + temporal audit")
        if task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling"}:
            bundle.append("metrics + validation artifacts")
        if task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling", "cohort_extraction"}:
            bundle.append("cohort definition artifact")
        # Preserve order while de-duplicating.
        return list(dict.fromkeys(bundle))[:7]
