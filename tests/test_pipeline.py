from anomaly_detection.models import ModelOutput
from anomaly_detection.pipeline import SeriesPipeline


class Fixed:
    """Always expects 100 and reports score 1."""

    def __init__(self, expected=100.0, ready=True):
        self.expected = expected
        self.ready = ready

    def update(self, ts, value):
        if not self.ready:
            return None
        return ModelOutput(1.0, self.expected, self.expected - 10, self.expected + 10)


def test_warmup_none_then_scored():
    p = SeriesPipeline(Fixed(), warmup_samples=3)
    assert p.update(1, 50) is None
    assert p.update(2, 50) is None
    assert p.update(3, 50).score == 1.0


def test_model_not_ready_gives_no_result():
    p = SeriesPipeline(Fixed(ready=False))
    assert p.update(1, 5) is None and not p.ready


def test_direction():
    above = SeriesPipeline(Fixed(), direction="above")
    below = SeriesPipeline(Fixed(), direction="below")
    assert above.update(1, 50).score == 0.0 and above.update(2, 150).score == 1.0
    assert below.update(1, 150).score == 0.0 and below.update(2, 50).score == 1.0


def test_tolerance_amount_and_ratio():
    p = SeriesPipeline(Fixed(), tolerance_amount=5)
    assert p.update(1, 96).score == 0.0 and p.update(2, 90).score == 1.0
    q = SeriesPipeline(Fixed(), tolerance_ratio=0.1)
    assert q.update(1, 91).score == 0.0 and q.update(2, 80).score == 1.0


def test_min_expected_suppresses_only_downward_and_only_for_small_expected():
    low = SeriesPipeline(Fixed(expected=3.0), min_expected=10)
    assert low.update(1, 0.0).score == 0.0  # low-traffic noise
    assert low.update(2, 9.0).score == 1.0  # upward move is still reported
    big = SeriesPipeline(Fixed(expected=500.0), min_expected=10)
    assert big.update(1, 0.0).score == 1.0  # collapse from a large level is reported


def test_repeated_or_old_timestamps_are_ignored():
    p = SeriesPipeline(Fixed())
    assert p.update(10, 1) is not None
    assert p.update(10, 1) is None
    assert p.update(5, 1) is None
