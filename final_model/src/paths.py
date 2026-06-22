"""Path helpers: dataset root vs code root."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent


def default_checkpoint_dir(stage: str = "stage1") -> Path:
    return CODE_ROOT / "checkpoints" / stage
