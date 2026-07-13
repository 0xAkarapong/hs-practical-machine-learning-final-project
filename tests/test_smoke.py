from pathlib import Path


def test_main_exists():
    assert Path("main.py").exists()
