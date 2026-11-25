import pytest

from anomaly_detection.config import DatasourceConfig
from anomaly_detection.prom import PromClient, QueryError

from .conftest import matrix, vector


def client(fake, **kw):
    return PromClient(DatasourceConfig(url=fake.url, **kw))


def test_instant_query_parses_and_drops_nan(fake_prom):
    fake_prom.handler = lambda p, q: (
        200,
        vector(({"__name__": "m", "a": "1"}, 100, "5"), ({"a": "2"}, 100, "NaN")),
    )
    out = client(fake_prom).query_instant("m", 100)
    assert [(dict(s.labels), s.value) for s in out] == [({"a": "1"}, 5.0)]


def test_empty_result_is_empty_list_not_zero(fake_prom):
    assert client(fake_prom).query_instant("m", 100) == []


def test_http_error_raises(fake_prom):
    fake_prom.handler = lambda p, q: (503, {"status": "error", "error": "overloaded"})
    with pytest.raises(QueryError, match="overloaded"):
        client(fake_prom).query_instant("m", 1)


def test_connection_error_raises():
    c = PromClient(DatasourceConfig(url="http://127.0.0.1:1", timeout=1))
    with pytest.raises(QueryError):
        c.query_instant("m", 1)


def test_auth_headers_and_tenant_path(fake_prom):
    c = client(
        fake_prom,
        path_prefix="/select/0/prometheus",
        auth={"type": "bearer", "token": "tok"},
        headers={"X-Scope-OrgID": "t1"},
    )
    c.query_instant("m", 1)
    req = fake_prom.requests[0]
    assert req["path"] == "/select/0/prometheus/api/v1/query"
    assert req["headers"]["Authorization"] == "Bearer tok"
    assert req["headers"]["X-Scope-OrgID"] == "t1"


def test_basic_auth(fake_prom):
    c = client(fake_prom, auth={"type": "basic", "username": "u", "password": "p"})
    c.query_instant("m", 1)
    assert fake_prom.requests[0]["headers"]["Authorization"].startswith("Basic ")


def test_query_range_is_chunked_and_merged(fake_prom):
    def handler(path, q):
        start, end, step = float(q["start"]), float(q["end"]), float(q["step"])
        vals = []
        t = start
        while t <= end:
            vals.append((t, "1"))
            t += step
        return 200, matrix(({"a": "x"}, vals))

    fake_prom.handler = handler
    c = client(fake_prom, max_points_per_request=10)
    out = c.query_range("m", 0, 99 * 60, 60)
    assert len(fake_prom.requests) == 10
    (points,) = out.values()
    assert [p[0] for p in points] == [i * 60.0 for i in range(100)]
