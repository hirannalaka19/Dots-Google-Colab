import pytest

import setup_colab


@pytest.mark.parametrize(
    ("version", "expected"),
    [("2.11.0+cu130", (2, 11)), ("2.8.0", (2, 8)), ("2.10.1+cpu", (2, 10))],
)
def test_minor(version, expected):
    assert setup_colab.minor(version) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [("2.11.0+cu130", "cu130"), ("2.11.0+cu126", "cu126"), ("2.11.0+cpu", "cu130"), ("2.11.0", "cu130")],
)
def test_cuda_tag(version, expected):
    assert setup_colab.cuda_tag(version) == expected


def test_run_reports_a_missing_program():
    assert setup_colab.run("definitely-not-a-real-program-xyz") == 127


@pytest.fixture
def fake_env(monkeypatch):
    versions: dict[str, str | None] = {}
    installs: list[tuple] = []
    monkeypatch.setattr(setup_colab, "installed", lambda package: versions.get(package))
    monkeypatch.setattr(setup_colab, "pip_install", lambda *args: installs.append(args) or 0)
    return versions, installs


def test_matching_torchaudio_needs_nothing(fake_env):
    versions, installs = fake_env
    versions.update(torch="2.11.0+cu130", torchaudio="2.11.0+cu130")
    assert setup_colab.ensure_matching_torchaudio() and installs == []


def test_mismatched_torchaudio_is_replaced_from_the_right_index(fake_env):
    versions, installs = fake_env
    versions.update(torch="2.10.0+cu128", torchaudio="2.9.0+cu128")
    assert setup_colab.ensure_matching_torchaudio()
    assert installs == [("torchaudio==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128")]


def test_torch_newer_than_last_torchaudio_installs_the_211_pair(fake_env):
    versions, installs = fake_env
    versions.update(torch="2.13.0+cu130")
    assert setup_colab.ensure_matching_torchaudio()
    assert installs == [("torch==2.11.0", "torchaudio==2.11.0", "--index-url",
                         "https://download.pytorch.org/whl/cu130")]


def test_missing_torch_fails(fake_env):
    assert not setup_colab.ensure_matching_torchaudio()
