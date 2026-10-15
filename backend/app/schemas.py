from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from . import config

ModelName = Literal["bert", "logreg", "rf"]


class PredictRequest(BaseModel):
    title: str = ""
    description: str = ""
    model: Literal["bert", "logreg", "rf", "all"] = "bert"

    @field_validator("title", "description", mode="before")
    @classmethod
    def _strip(cls, v):
        if v is None:
            return ""
        if not isinstance(v, str):
            raise ValueError("must be text")
        return v.strip()

    @model_validator(mode="after")
    def _check_length(self):
        total = len(self.title) + len(self.description)
        if total == 0:
            raise ValueError("Please enter a title or a description.")
        if total > config.MAX_CHARS:
            raise ValueError(f"Text is too long ({total} characters). The limit is {config.MAX_CHARS}.")
        return self

    @property
    def text(self) -> str:
        return f"{self.title} {self.description}".strip()


class ModelResult(BaseModel):
    model: ModelName
    model_name: str
    topic: str
    confidence: float
    probabilities: dict[str, float]
    hints: list[str]


class PredictResponse(BaseModel):
    primary: ModelResult
    results: dict[str, ModelResult]
    agreement: bool | None          # None when fewer than two models answered
    unavailable: dict[str, str]     # model -> reason, for models that could not answer
