from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CriterionResult = Literal["pass", "fail", "not_verifiable"]


class Analysis(BaseModel):
    subjects: list[str] = Field(default_factory=list)
    pose: str = "preserve original"
    expression: str = "preserve original"
    required_elements: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    important_regions: list[str] = Field(default_factory=list)
    edit_instructions: str = ""


class Defect(BaseModel):
    criterion: Literal["identity", "expression", "pose", "details", "text", "lines", "colors", "background"]
    severity: Literal["low", "medium", "high"]
    region: str = ""
    evidence: str = ""
    correction: str = ""


class EvaluationCriteria(BaseModel):
    identity: CriterionResult
    expression: CriterionResult
    pose: CriterionResult
    details: CriterionResult
    text: CriterionResult
    lines: CriterionResult
    colors: CriterionResult
    background: CriterionResult


class Evaluation(BaseModel):
    category: Literal["no_defects", "possible_correction", "requires_attention"]
    criteria: EvaluationCriteria
    defects: list[Defect] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
