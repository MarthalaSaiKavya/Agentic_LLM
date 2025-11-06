from token_self_repair.config import ProjectConfig
from token_self_repair.uncertainty.logtoku import LogTokUEstimator


def test_logtoku_estimator_produces_scores():
    config = ProjectConfig()
    estimator = LogTokUEstimator(config=config)
    tokens = ["a", "b", "c"]
    logits = [
        [0.0, 1.0, -0.5],
        [1.5, 0.0, -1.0],
        [-0.5, -0.2, 0.0],
    ]

    scores = list(estimator.score(tokens, logits))

    assert len(scores) == len(tokens)
    for score in scores:
        assert 0.0 <= score.total_uncertainty <= 1.0
        assert score.probability <= 1.0
