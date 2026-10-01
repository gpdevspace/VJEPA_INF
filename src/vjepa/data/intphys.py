"""IntPhys 2019 dev set: 3 blocks x 30 scenes x 4 movies (2 physically possible, 2 impossible).

Download: https://download-intphys.cognitive-ml.fr/dev.tar.gz (md5 6a72715007f2f3b0e9546bfa8d8fc39b), extract to
data/intphys/ so movies live at data/intphys/dev/<block>/<scene>/<1..4>/scene/scene_XXX.png.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vjepa.video import read_frame_dir

ROOT = Path("data/intphys/dev")
BLOCKS = {"O1": "object permanence", "O2": "shape constancy", "O3": "spatio-temporal continuity"}


@dataclass(frozen=True)
class Movie:
    block: str
    scene: str
    index: int  # 1..4
    frames_dir: Path
    possible: bool


def list_scenes(block: str, root: Path = ROOT) -> dict[str, list[Movie]]:
    """All scenes of a block, each with its 4 movies."""
    scenes = {}
    scene_dirs = [p for p in (root / block).iterdir() if p.is_dir() and p.name.isdigit()]  # skips .DS_Store etc.
    for scene_dir in sorted(scene_dirs, key=lambda p: int(p.name)):
        movies = []
        for i in (1, 2, 3, 4):
            status = json.loads((scene_dir / str(i) / "status.json").read_text())
            movies.append(Movie(block, scene_dir.name, i, scene_dir / str(i) / "scene", status["header"]["is_possible"]))
        scenes[scene_dir.name] = movies
    return scenes


def load_frames(movie: Movie, frame_step: int = 2) -> np.ndarray:
    """Every `frame_step`-th frame, `99 // frame_step` frames in total, as in the reference eval."""
    return read_frame_dir(movie.frames_dir, frame_step)[: 99 // frame_step]


def match_pairs(clips: list[np.ndarray]) -> list[tuple[int, int]]:
    """Pair the movies that stay identical longest (they differ only by the physical event).

    Same rule as `get_breaking_points` + `get_matches` in jepa-intuitive-physics: movie 0 is paired with the movie
    it diverges from last, and the remaining two form the other pair.
    """

    def first_difference(a: np.ndarray, b: np.ndarray) -> int:
        diff = np.flatnonzero((a != b).reshape(len(a), -1).any(axis=1))
        return int(diff[0]) if len(diff) else len(a)

    breaking_points = [first_difference(clips[0], clips[k]) for k in (1, 2, 3)]
    partner = int(np.argmax(breaking_points)) + 1
    rest = [k for k in (1, 2, 3) if k != partner]
    return [(0, partner), (rest[0], rest[1])]
