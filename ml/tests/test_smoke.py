from pathlib import Path


def test_main_exists():
    root_dir = Path(__file__).resolve().parent.parent
    assert (root_dir / "main.py").exists()
