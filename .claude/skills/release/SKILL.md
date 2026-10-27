---
name: release
description: Cut a versioned release of this addon end-to-end -- decide the version bump, update the changelog, open the release PR, and tag the merged commit once it's ready. Use when the user asks to cut/push/ship a release.
---

# Cutting a release

1. **Version**: compare `bl_info["version"]` in `__init__.py` with the latest `vX.Y.Z` tag and the
   bullets under `## next version` in `CHANGELOG.md`. Bump (SemVer) if not already done; dropping a
   Blender version is a major bump. Drop bullets for bugs that never reached `master`.
2. **Changelog**: rename `## next version` to `## X.Y.Z`.
3. **Commit**: the commit message becomes the GitHub Release body verbatim (`ci.yml`'s `release`
   job reads `git log -1 --format=%B` of the tagged commit), so write user-facing release notes and
   don't add `Co-Authored-By`/session trailers.
4. **PR**: open it and stop; continue only once the user confirms it merged (no polling).
5. **Tag** the version-bump commit, not a merge commit:

   ```
   .claude/skills/release/tag_release.sh [-y|--yes] <version> <#pr-number|sha>
   ```

   It checks the commit is on `origin/master`, is not a merge commit, has the right
   `bl_info["version"]` and `## X.Y.Z` changelog heading, and that the tag doesn't exist yet, then
   tags and pushes. Agents have no TTY for its prompt: get the user's go-ahead, then pass `-y`.
   The tag push triggers the `release` job, which publishes `Blender_JA_SWTOR_Tools.zip`.
