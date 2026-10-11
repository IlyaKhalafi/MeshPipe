import json

from meshpipe.cli import main

pytest_plugins = ["test_motion"]


def test_demo_benchmark_without_gui(tmp_path, capsys):
    path = tmp_path / "nested" / "benchmark.json"
    assert (
        main(["benchmark", "--demo", "--frames", "5", "--warmup", "1", "--output", str(path)]) == 0
    )
    result = json.loads(path.read_text())
    assert result["frames"] == 5
    assert result["faces_detected"] == 5
    assert result["throughput_fps"] > 0


def test_invalid_config_exit_code(capsys):
    assert main(["benchmark", "--demo", "--width", "10"]) == 1
    assert "at least 160" in capsys.readouterr().err


def test_headless_motion_exports(performance, tmp_path):
    assert (
        main(["export-animation", str(performance), "--output", str(tmp_path / "motion.json")]) == 0
    )
    assert main(["export-web", str(performance), "--output", str(tmp_path / "web")]) == 0
    assert (tmp_path / "web" / "index.html").is_file()
