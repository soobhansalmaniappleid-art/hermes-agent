"""proxy.network_lock: sandboxes reach the network only through the egress proxy."""

import subprocess

import pytest

from tools.environments import docker as docker_env

PROXY_ENV = {
    "HTTPS_PROXY": "http://host.docker.internal:9090",
    "https_proxy": "http://host.docker.internal:9090",
    "HTTP_PROXY": "http://host.docker.internal:9091",
    "http_proxy": "http://host.docker.internal:9091",
    "NO_PROXY": "127.0.0.1,localhost,::1",
    "OPENROUTER_API_KEY": "proxy-token",
}
HOST_ARGS = ["--add-host", "host.docker.internal:host-gateway"]


class FakeDocker:
    """Records docker invocations; networks/containers exist once created."""

    def __init__(self, *, network=False, gateway=None, fail=None):
        self.network = network
        self.gateway = gateway  # None, "true" (running) or "false" (stopped)
        self.fail = fail
        self.calls = []

    def __call__(self, docker_exe, *args, timeout=60):
        self.calls.append(args)
        code, err, out = 0, "", ""
        if args[:2] == ("network", "inspect"):
            code = 0 if self.network else 1
        elif args[:2] == ("network", "create"):
            self.network = True
        elif args[0] == "inspect":
            code, out = (1, "") if self.gateway is None else (0, self.gateway)
        elif args[0] == "run" and self.fail == "run":
            code, err = 125, "pull failed"
        elif args[0] == "run":
            self.gateway = "true"
        return subprocess.CompletedProcess(args, code, out, err)


@pytest.fixture
def lock_on(monkeypatch):
    monkeypatch.setattr(docker_env, "_egress_network_lock_enabled", lambda: True)


def test_lock_off_leaves_everything_unchanged(monkeypatch):
    monkeypatch.setattr(docker_env, "_egress_network_lock_enabled", lambda: False)
    fake = FakeDocker()
    monkeypatch.setattr(docker_env, "_docker_ok", fake)

    env, hosts, net = docker_env._apply_egress_network_lock("docker", dict(PROXY_ENV), list(HOST_ARGS))

    assert (env, hosts, net) == (PROXY_ENV, HOST_ARGS, [])
    assert fake.calls == []


def test_lock_without_proxy_does_nothing(monkeypatch, lock_on):
    """No proxy means nothing to route through: never strand the sandbox offline."""
    fake = FakeDocker()
    monkeypatch.setattr(docker_env, "_docker_ok", fake)

    assert docker_env._apply_egress_network_lock("docker", {}, []) == ({}, [], [])
    assert fake.calls == []


def test_lock_routes_proxy_through_gateway_on_internal_network(monkeypatch, lock_on):
    fake = FakeDocker()
    monkeypatch.setattr(docker_env, "_docker_ok", fake)

    env, hosts, net = docker_env._apply_egress_network_lock("docker", dict(PROXY_ENV), list(HOST_ARGS))

    assert net == ["--network", "hermes-egress"]
    assert hosts == []
    assert env["HTTPS_PROXY"] == "http://hermes-egress-gw-9090:9090"
    assert env["http_proxy"] == "http://hermes-egress-gw-9090:9091"
    assert env["NO_PROXY"] == PROXY_ENV["NO_PROXY"]
    assert env["OPENROUTER_API_KEY"] == "proxy-token"

    create = next(c for c in fake.calls if c[:2] == ("network", "create"))
    assert "--internal" in create
    run = next(c for c in fake.calls if c[0] == "run")
    assert {"--cap-drop", "ALL", "--read-only"} <= set(run)
    script = run[-1]
    assert "TCP:host.docker.internal:9090" in script and "TCP:host.docker.internal:9091" in script
    assert ("network", "connect", "hermes-egress", "hermes-egress-gw-9090") in fake.calls


def test_running_gateway_is_reused(monkeypatch, lock_on):
    fake = FakeDocker(network=True, gateway="true")
    monkeypatch.setattr(docker_env, "_docker_ok", fake)

    docker_env._apply_egress_network_lock("docker", dict(PROXY_ENV), list(HOST_ARGS))

    assert not any(c[0] in {"run", "rm"} or c[:2] == ("network", "create") for c in fake.calls)


def test_stopped_gateway_is_replaced(monkeypatch, lock_on):
    fake = FakeDocker(network=True, gateway="false")
    monkeypatch.setattr(docker_env, "_docker_ok", fake)

    docker_env._apply_egress_network_lock("docker", dict(PROXY_ENV), list(HOST_ARGS))

    assert ("rm", "-f", "hermes-egress-gw-9090") in fake.calls
    assert any(c[0] == "run" for c in fake.calls)


def test_gateway_failure_refuses_to_start_the_sandbox(monkeypatch, lock_on):
    """Fail closed: a sandbox must never fall back to open networking."""
    monkeypatch.setattr(docker_env, "_docker_ok", FakeDocker(fail="run"))

    with pytest.raises(RuntimeError, match="egress gateway"):
        docker_env._apply_egress_network_lock("docker", dict(PROXY_ENV), list(HOST_ARGS))
