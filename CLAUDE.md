# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Blender add-on importing Star Wars: The Old Republic 64-bit `.jba` animations, retargeted onto the
Jedi Academy `_humanoid` skeleton, as NLA strips. It is meant to be used with the Blender Jedi Academy
Plugin Suite (`jediacademy` add-on, `../Blender-Jedi-Academy-Tools`), which owns `.gla`/`animation.cfg`
I/O; this add-on only mirrors its scene conventions (armature `skeleton_root`, NLA layout,
`action.g2_sequence_prop`), like `../Blender-JA-XSI-Tools`.

The installed module name is `Blender_JA_SWTOR_Tools`: the folder name inside the release zip
(`PACKAGE` in the Makefile and in `tests/testutil.py`).

## Commands

- `make` — builds `build/Blender_JA_SWTOR_Tools.zip` (installable add-on).
- `make pep8` / `make format` — pycodestyle check (what CI runs) / autopep8 fix, config in `.pep8`.
- Tests: `blender --background --factory-startup --python-exit-code 1 --python tests/run_tests.py`.
- CI (`.github/workflows/ci.yml`): `smoke-test` (Blender 4.1 and 5.2) and `pep8` on PRs, tags and the
  daily schedule; `nightly` and `release` publish the zip. No pyright job yet: the code isn't clean
  against `fake-bpy-module` stubs.

## Writing comments, commit messages, PR descriptions, and issues

Keep all four succinct: state the fact/change, skip restating what the diff already shows.

## Architecture

- `tor2jka/` — numpy-only, testable outside Blender:
  - `swtor.py` — `.jba` ("JAWB" v2), `.gr2` skeleton and `.mph` (bone order) readers; skeleton/rig picking.
  - `gla.py` — Jedi Academy `.gla` ("2LGA" v6) reader/writer.
  - `retarget.py` — `BONE_MAP` and the SWTOR → `_humanoid` retargeter (rotations only, JKA bone lengths).
  - `qmath.py` — quaternion (w, x, y, z) / matrix helpers.
- `blender_importer.py` — bakes retargeted poses into actions and NLA strips; additive `ad_*` files on a
  COMBINE layer above the selected strip.
- `__init__.py` — operator `import_anim.swtor_jba`, dialog, menu entry.

### Blender version support

`bl_info["blender"]` is 4.1. Blender 4.4 introduced slotted actions and 5.0 removed `Action.fcurves`:
FCurve access goes through `get_fcurves`. CI tests the 4.1 and 5.2 boundaries.

## Testing

Real game files are proprietary and not committed (keep local copies in the git-ignored
`.proprietary/`); `tests/run_tests.py` covers registration and the pure readers on synthetic data.

## Releases

See the `release` skill: SemVer in `bl_info["version"]`, user-facing changes under `## next version` in
`CHANGELOG.md`, the release commit message is the GitHub Release body, tag it with
`.claude/skills/release/tag_release.sh`.
