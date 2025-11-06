from dataclasses import dataclass
from typing import Iterable

from token_self_repair.config import ProjectConfig, Thresholds
from token_self_repair.llm.base import LLMClient, TokenLogit
from token_self_repair.messaging.status import StatusMessenger
from token_self_repair.pipelines.base import UncertaintyAwarePipeline
from token_self_repair.repair.constitutional import ConstitutionalRepair
from token_self_repair.uncertainty.logtoku import LogTokUEstimator


@dataclass
class UniformLLM(LLMClient):
    tokens: Iterable[str]

    def generate(self, prompt: str, *, max_tokens: int = 256):
        logits = [0.0, 0.0, 0.0, 0.0]
        for token in self.tokens:
            yield TokenLogit(token=token, logits=logits)


def test_pipeline_triggers_repair_when_uncertain():
    config = ProjectConfig(
        max_self_repairs=1,
        thresholds=Thresholds(
            high_confidence=0.9,
            moderate_confidence=0.7,
            low_confidence=0.5,
            repair_activation_uncertainty=0.2,
        ),
    )
    llm = UniformLLM(tokens=["draft", "response"])
    estimator = LogTokUEstimator(config=config)
    messenger = StatusMessenger()
    strategy = ConstitutionalRepair()
    pipeline = UncertaintyAwarePipeline(
        llm=llm,
        estimator=estimator,
        strategies=[strategy],
        messenger=messenger,
        config=config,
    )

    result = pipeline.run("Solve the task")

    assert result.step.final is True
    assert result.step.repair_attempt >= 1
    titles = [message.title for message in messenger.history]
    assert any("Low confidence" in title or "Moderate uncertainty" in title for title in titles)
