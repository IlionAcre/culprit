from culprit.worker import run_worker


class _FakeQueue:
    def __init__(self):
        self.connection = "fake-connection"


class _FakeWorker:
    """Captures its construction and work() args instead of touching a
    real Worker: RQ's Worker requires heartbeat/process-registration
    behavior that fakeredis does not fully emulate, and this module's own
    job is only to prove run_worker builds the right queue and calls
    work(burst=...) on it, not to re-test RQ's Worker itself."""

    instances = []

    def __init__(self, queues, connection=None):
        self.queues = queues
        self.connection = connection
        self.burst = None
        _FakeWorker.instances.append(self)

    def work(self, burst=False):
        self.burst = burst


def test_run_worker_builds_a_worker_for_the_culprit_queue_and_calls_work(monkeypatch):
    fake_queue = _FakeQueue()
    _FakeWorker.instances.clear()
    monkeypatch.setattr("culprit.worker.get_queue", lambda: fake_queue)
    monkeypatch.setattr("culprit.worker.Worker", _FakeWorker)

    run_worker(burst=True)

    assert len(_FakeWorker.instances) == 1
    worker = _FakeWorker.instances[0]
    assert worker.queues == [fake_queue]
    assert worker.connection == fake_queue.connection
    assert worker.burst is True


def test_run_worker_defaults_to_non_burst_for_production_polling(monkeypatch):
    fake_queue = _FakeQueue()
    _FakeWorker.instances.clear()
    monkeypatch.setattr("culprit.worker.get_queue", lambda: fake_queue)
    monkeypatch.setattr("culprit.worker.Worker", _FakeWorker)

    run_worker()

    assert _FakeWorker.instances[0].burst is False
