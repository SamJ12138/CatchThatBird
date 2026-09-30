from __future__ import annotations

import pytest

import main
from camera import classify_device, find_pocket_index
from config_schema import AppConfig, CameraConfig, DetectionConfig, LoggingConfig, StorageConfig
from tests.conftest import read_log


@pytest.mark.parametrize(
    "name, kind",
    [
        ("DJI Osmo Pocket 3", "pocket"),
        ("OsmoPocket3", "pocket"),
        ("dji webcam", "pocket"),
        ("Integrated Camera", "builtin"),
        ("Integrated Webcam", "builtin"),
        ("HP HD Camera", "builtin"),
        ("Lenovo EasyCamera", "builtin"),
        ("Surface Camera Front", "builtin"),
        ("USB2.0 HD UVC WebCam", "unknown"),  # this PC's own webcam
        ("OBS Virtual Camera", "unknown"),
        ("<device 0>", "unknown"),            # probe fallback placeholder
        ("", "unknown"),
    ],
)
def test_classify_device(name: str, kind: str) -> None:
    assert classify_device(name) == kind


def test_find_pocket_index() -> None:
    devices = [(0, "USB2.0 HD UVC WebCam"), (1, "OBS Virtual Camera"), (2, "DJI Osmo Pocket 3")]
    assert find_pocket_index(devices) == 2
    assert find_pocket_index(devices[:2]) is None


def _config(index: int) -> AppConfig:
    return AppConfig(
        camera=CameraConfig(device_index=index), detection=DetectionConfig(),
        logging=LoggingConfig(), storage=StorageConfig(),
    )


def _device_select_terminal(obs) -> dict:
    obs.close()
    lines = [l for l in read_log(obs.path) if l["stage"] == "device_select"]
    return [l for l in lines if l["event"] != "start"][-1]


def test_device_select_warns_on_unknown_device(obs, log_messages) -> None:
    """obs #13: an unrecognised camera warns (decision: warn, do not refuse)."""
    name = main.resolve_device_selection(_config(0), [(0, "USB2.0 HD UVC WebCam")], obs)

    assert name == "USB2.0 HD UVC WebCam"  # still selected: warn only
    warnings = [m for level, m in log_messages if level == "WARNING"]
    assert any("USB2.0 HD UVC WebCam" in m for m in warnings), warnings
    terminal = _device_select_terminal(obs)
    assert terminal["event"] == "fail"
    assert terminal["error_type"] == "input_invalid"


def test_device_select_pocket_does_not_warn(obs, log_messages) -> None:
    main.resolve_device_selection(_config(1), [(0, "Integrated Camera"), (1, "DJI Osmo Pocket 3")], obs)

    assert [m for level, m in log_messages if level == "WARNING"] == []
    assert _device_select_terminal(obs)["event"] == "success"


def test_device_select_builtin_still_warns(obs, log_messages) -> None:
    main.resolve_device_selection(_config(0), [(0, "Integrated Camera"), (1, "DJI Osmo Pocket 3")], obs)

    warnings = [m for level, m in log_messages if level == "WARNING"]
    assert any("--device 1" in m for m in warnings), warnings
