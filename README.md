# Blender JA SWTOR Tools

Blender add-on importing Star Wars: The Old Republic (64-bit client) `.jba` animations, retargeted
onto the Jedi Academy `_humanoid` skeleton, as NLA strips. It complements the
[Blender Jedi Academy Plugin Suite](https://github.com/BE-Arbiter/Blender-Jedi-Academy-Tools), which
reads and writes the game's `.gla` and `animation.cfg`.

## Installation

Download `Blender_JA_SWTOR_Tools.zip` from the [releases](https://github.com/BE-Arbiter/Blender-JA-SWTOR-Tools/releases),
install it with "Install from Disk" in Blender's add-on preferences and enable
"JA SWTOR Anim Import (.jba)". Blender 4.1 or newer.

## Usage

1. Import `_humanoid.gla` with the Jedi Academy plugin suite (armature `skeleton_root`).
2. **File > Import > JA SWTOR Anim Import (.jba)**, pick one or more `.jba` files from an extracted
   SWTOR `resources/` tree. Each becomes an action and an NLA strip with `animation.cfg` metadata.
   - The SWTOR body skeleton (`resources/art/dynamic/spec/<body>_skeleton.gr2`) is picked
     automatically from the animation's proportions, or forced with "SWTOR Skeleton".
   - Several sequence names separated by dots (`BOTH_STAND2.BOTH_STAND1`) create stacked actions.
   - Additive `ad_*` files (aim yaw / pitch, head looks, faces...) are layered with COMBINE blending
     on the selected NLA strip instead of creating a strip of their own.
3. Export the `.gla` and `animation.cfg` with the Jedi Academy plugin suite.

Options cover looping, resampling, pelvis horizontal offset, root motion, finger animation and
grounding the feet; hover them in the import dialog for details.

## Development

- `make` builds `build/Blender_JA_SWTOR_Tools.zip` (needs `zip`).
- `make pep8` / `make format` check / fix the code style.
- Tests run headless in Blender:
  `blender --background --factory-startup --python-exit-code 1 --python tests/run_tests.py`.
- Releases: bump `bl_info["version"]`, rename `## next version` in `CHANGELOG.md`, then push a
  `vX.Y.Z` tag on `master` (`.claude/skills/release/tag_release.sh`). CI builds the zip and
  publishes the GitHub Release, using the tagged commit's message as release notes.
