"""Fixture tree for the Grep/Glob semantic-parity tests (see test_workflow_search_parity).

Every semantic dimension the workflow Grep/Glob steps must reproduce has a
file here: hidden files and directories, a gitignored directory and file
pattern, a non-ignored ``node_modules``, a NUL-byte binary, a Latin-1 file, a
line longer than Grep's column limit, case variants, VCS metadata and
symlinks that leave the workspace. Modification times are fixed so ordering
is deterministic.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

BASE_MTIME = 1_700_000_000

#: (relative path, bytes) in modification-time order (oldest first).
FILES: tuple[tuple[str, bytes], ...] = (
    (".gitignore", b"ignored_dir/\n*.log\n"),
    ("src/a.py", b"alpha Needle one\nneedle two\n"),
    ("src/sub/b.py", b"NEEDLE three\n"),
    (".hid/h.txt", b"needle hidden\n"),
    (".dotfile", b"needle dot\n"),
    ("ignored_dir/i.py", b"needle ignored\n"),
    ("x.log", b"needle log\n"),
    ("node_modules/pkg/n.js", b"needle nm\n"),
    ("bin.dat", b"needle\x00binary\n"),
    ("latin.txt", b"needle caf\xe9 latin1\n"),
    ("long.txt", b"needle " + b"x" * 700 + b"\n"),
    ("UP.PY", b"needle upper\n"),
    ("la.txt", b"foobar\nfoobaz\n"),
    ("src/c.txt", b"needle newest\n"),
)


def build(root: Path) -> dict[str, bool]:
    """Create the tree under ``root`` (a fresh Git repo); returns which links exist."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    for i, (rel, data) in enumerate(FILES):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        stamp = BASE_MTIME + 10 * i
        os.utime(path, (stamp, stamp))
    (root / ".git" / "needle_in_git").write_bytes(b"needle .git internal\n")
    outside = root.parent / f"{root.name}_outside"
    outside.mkdir(exist_ok=True)
    (outside / "o.txt").write_bytes(b"needle outside\n")
    links = {"linkdir": False, "linkfile.txt": False}
    for name, target, is_dir in (
        ("linkdir", outside, True),
        ("linkfile.txt", outside / "o.txt", False),
    ):
        try:
            os.symlink(target, root / name, target_is_directory=is_dir)
            links[name] = True
        except OSError:
            pass  # Windows without symlink privilege
    return links


if __name__ == "__main__":
    import sys

    print(build(Path(sys.argv[1])))
