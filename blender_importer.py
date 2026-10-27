"""Bakes retargeted SWTOR animations onto a Jedi Academy armature as NLA strips.

The armature is the jediacademy addon's skeleton_root (a .gla imported in the scene), or the
active armature. The retargeter works in Ghoul2 model space and returns, per frame and bone,
the rigid "absolute offset" that takes the bone from its base pose to its animated pose - the
same quantity the dotXSI importer builds from its XSI world matrices - so baking and the NLA
layout follow the dotXSI 3.0 importer: one action per sequence, strips on
"Sequences Layer n" tracks, extrapolation NOTHING, REPLACE blending, animation.cfg metadata in
action.g2_sequence_prop.

Additive SWTOR animations (ad_*.jba: aim yaw / pitch, head looks, faces, mount leans...) are
deltas, so they are not strips of their own: they are baked as pose bone basis deltas (identity
where the additive does nothing) into an action without sequence metadata and added with
COMBINE blending on an "Additive Layer n" track right above the selected NLA strip's track.
"""

import os
import re

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

from .tor2jka.gla import GLA
from .tor2jka.retarget import Retargeter
from .tor2jka.swtor import (JBA, Skeleton, Pose, AdditiveTracks, find_rigs, pick_skeleton_and_rig,
                            pick_additive_rig)


class TorImportError(Exception):
    pass


def find_skeleton(context):
    """skeleton_root (as named by the jediacademy addon), else the active armature."""
    scene_objects = context.scene.objects
    obj = scene_objects.get("skeleton_root")
    if obj is not None and obj.type == "ARMATURE":
        return obj
    obj = context.active_object
    if obj is not None and obj.type == "ARMATURE" and obj.name in scene_objects:
        return obj
    return None


def tor_resources_root(jba_path):
    """.../resources/anim/humanoid/<body>/x.jba -> .../resources"""
    d = os.path.dirname(os.path.abspath(jba_path))
    return os.path.dirname(os.path.dirname(os.path.dirname(d)))


def default_skeleton_path(jba_path):
    body = os.path.basename(os.path.dirname(os.path.abspath(jba_path)))
    return os.path.join(tor_resources_root(jba_path), "art", "dynamic", "spec", body + "_skeleton.gr2")


class ArmatureSkeleton:
    """Ghoul2 rest pose read back from a Blender armature (fallback when no .gla is given).
    Ghoul2 bones point along X, Blender bones along Y."""

    def __init__(self, obj):
        L = obj.matrix_local
        bones = list(obj.data.bones)
        self.names = [b.name for b in bones]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parents = [self.index[b.parent.name] if b.parent else -1 for b in bones]
        self.num_bones = len(bones)
        self._rest = np.zeros((len(bones), 4, 4))
        for i, b in enumerate(bones):
            m = L @ b.matrix_local
            bx, by, bz = (m.col[k].to_3d().normalized() for k in range(3))
            # inverse of the jediacademy addon's GLABoneRotToBlender: Ghoul2 axes = (Y, -Z, -X)
            self._rest[i, :3, 0] = by
            self._rest[i, :3, 1] = -bz
            self._rest[i, :3, 2] = -bx
            self._rest[i, :3, 3] = m.translation
            self._rest[i, 3, 3] = 1

    def rest_positions(self):
        return self._rest[:, :3, 3].copy()

    def rest_matrices(self):
        return self._rest.copy()


def get_fcurves(obj, action):
    """FCurve collection of `action` for `obj`, on Blender 4.4+ (slotted actions) and older."""
    anim = obj.animation_data_create()
    anim.action = action
    if not hasattr(action, "slots"):
        return action.fcurves
    slot = anim.action_slot
    if slot is None:
        slot = action.slots.new(id_type="OBJECT", name=obj.name)
        anim.action_slot = slot
    from bpy_extras import anim_utils
    return anim_utils.action_ensure_channelbag_for_slot(action, slot).fcurves


def new_fcurve(fcurves, data_path, index, group):
    for kwargs in ({"group_name": group}, {"action_group": group}, {}):
        try:
            return fcurves.new(data_path, index=index, **kwargs)
        except TypeError:
            continue


LOOP_NAME = re.compile(r"idle|loop|_run|run_|walk|jog|sprint|strafe|swim|stand|stance|_mid|breath|dance|hover|fly|crouch|sit")

LINEAR = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value


class Importer:
    _cache = {}  # skeleton / rig / gla readers, shared between runs

    def __init__(self, operator, context, opt):
        self.operator = operator
        self.context = context
        self.opt = opt
        self.messages = []
        self.warnings = []
        self.errors = []

    # ------------------------------------------------------------------ inputs
    def cached(self, kind, path, loader):
        key = (kind, os.path.normcase(os.path.abspath(path)), os.path.getmtime(path))
        if key not in self._cache:
            self._cache[key] = loader(path)
        return self._cache[key]

    def jka_skeleton(self, obj):
        gla_path = bpy.path.abspath(self.opt["gla_path"]) if self.opt["gla_path"] else ""
        if gla_path and os.path.isfile(gla_path):
            gla = self.cached("gla", gla_path, GLA)
            missing = [b.name for b in obj.data.bones if b.name not in gla.index]
            if len(missing) < len(obj.data.bones):
                return gla
            self.warnings.append("%s shares no bone with %r, using the armature's rest pose"
                                 % (os.path.basename(gla_path), obj.name))
        elif gla_path:
            self.warnings.append("Reference .gla not found (%s), using the armature's rest pose" % gla_path)
        return ArmatureSkeleton(obj)

    def humanoid_skeletons(self, path):
        """{body: Skeleton} for every body folder next to the .jba that has a skeleton."""
        humanoid = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        spec = os.path.join(tor_resources_root(path), "art", "dynamic", "spec")
        out = {}
        for body in sorted(os.listdir(humanoid)):
            skel_path = os.path.join(spec, body + "_skeleton.gr2")
            if os.path.isdir(os.path.join(humanoid, body)) and os.path.isfile(skel_path):
                out[body] = self.cached("skel", skel_path, Skeleton)
        return out

    def humanoid_rigs(self, path):
        """Rig bone orders from the .mph files of every body folder: an animation may use the
        rig of another body type (shared animations)."""
        humanoid = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        key = ("rigs", os.path.normcase(humanoid))
        if key not in self._cache:
            dirs = [humanoid] + [os.path.join(humanoid, d) for d in sorted(os.listdir(humanoid))
                                 if os.path.isdir(os.path.join(humanoid, d))]
            rigs = [tuple(r) for d in dirs for r in find_rigs(d)]
            self._cache[key] = [list(r) for r in dict.fromkeys(rigs)]
        return self._cache[key]

    def folder_rigs(self, path):
        """Rig bone orders of the .mph files next to the .jba (its own body type)."""
        folder = os.path.dirname(os.path.abspath(path))
        key = ("folder rigs", os.path.normcase(folder))
        if key not in self._cache:
            self._cache[key] = find_rigs(folder)
        return self._cache[key]

    def load_swtor(self, path):
        body = os.path.basename(os.path.dirname(os.path.abspath(path)))
        skel_path = bpy.path.abspath(self.opt["skeleton_path"]) if self.opt["skeleton_path"] else ""
        if skel_path:
            if not os.path.isfile(skel_path):
                raise TorImportError("SWTOR skeleton not found: %s" % skel_path)
            skeletons = {"custom": self.cached("skel", skel_path, Skeleton)}
        else:
            skeletons = self.humanoid_skeletons(path)
            if not skeletons:
                raise TorImportError("No SWTOR skeleton found for %s (expected %s); set it in the import options"
                                     % (os.path.basename(path), default_skeleton_path(path)))
        rigs = self.humanoid_rigs(path)

        try:
            jba = JBA(path)
        except (ValueError, OSError) as e:
            raise TorImportError(str(e))
        if jba.name.lower().startswith("ad_"):
            # An additive's translations are deltas, so the translation fit cannot pick the
            # skeleton or the rig: use the body's own skeleton and the rig its length / bones fit.
            skel = skeletons.get(body) or next(iter(skeletons.values()))
            names, count, margin = pick_additive_rig(jba, self.folder_rigs(path), rigs)
            if names is None:
                raise TorImportError("%s: no Morpheme rig with %d bones found (the .mph files are needed)"
                                     % (os.path.basename(path), jba.num_bones))
            if count > 1 and margin < 1.0:
                self.warnings.append("%s: the bone order is a close call between %d rigs, check the result"
                                     % (jba.name, count))
            return jba, names, skel
        used, skel, names, err = pick_skeleton_and_rig(jba, skeletons, rigs, preferred=body)
        if names is None:
            raise TorImportError("%s: no Morpheme rig with %d bones or more found (the .mph files are needed)"
                                 % (os.path.basename(path), jba.num_bones))
        if err > 3.0:
            self.warnings.append("%s: bone order only matched to %.1f cm, check the result" % (jba.name, err))
        if used not in (body, "custom"):
            print("SWTOR jba: %s uses the %s skeleton (shared animation)" % (jba.name, used))
        return jba, names, skel

    # ------------------------------------------------------------------ main
    def run(self, paths):
        obj = find_skeleton(self.context)
        if obj is None:
            raise TorImportError("No armature: import a .gla first (File > Import > JA Ghoul 2 Skeleton), "
                                 "or select an armature")
        jka = self.jka_skeleton(obj)

        names_opt = [n.strip() for n in self.opt.get("sequence_name", "").split(".") if n.strip()]
        start = self.opt["strip_start"]
        actions = []
        additive = [os.path.basename(p).lower().startswith("ad_") for p in paths]
        target, target_error = None, None
        if any(additive):
            # once, before the run adds (and selects) strips of its own
            target, target_error = self.find_target_strip(obj)
        for n, path in enumerate(paths):
            if additive[n]:
                if target is None:
                    self.errors.append("%s: %s" % (os.path.basename(path), target_error))
                else:
                    try:
                        actions.extend(self.add_additive(obj, jka, path, target,
                                                         names_opt[0] if names_opt and len(paths) == 1 else None))
                    except TorImportError as e:
                        self.errors.append(str(e))
                continue
            jba, rig, skel = self.load_swtor(path)
            if self.opt["fps"] > 0:
                jba.resample(self.opt["fps"])
            loop = self.opt["loop"]
            if loop == "AUTO":
                # attacks and transitions also end on their first pose, so require a looping name
                is_loop = jba.is_loop() and LOOP_NAME.search(jba.name) is not None
            else:
                is_loop = loop == "LOOP"
            pose = Pose(jba, rig, skel, root_motion=self.opt["root_motion"])
            rt = Retargeter(jka, skel, horizontal=self.opt["horizontal"].lower(),
                            fingers=self.opt["fingers"], ground=self.opt["ground"])
            ab = rt.convert_abs(pose)
            if is_loop and jba.is_loop():
                ab = ab[:-1]  # Ghoul2 wraps from the last frame back to the first itself

            # one file with a dotted name list: one strip per name; several files: file names
            names = names_opt if (names_opt and len(paths) == 1) else [jba.name]
            placed = self.add_sequence(obj, jka, ab, names, start, jba.fps, 0 if is_loop else -1)
            actions.extend(placed)
            start = -1  # following files go after the previous one
        return obj, actions

    # ------------------------------------------------------------------ baking
    def basis_keys(self, obj, jka, ab):
        """Pose bone basis (location, quaternion) per bone and frame for the absolute Ghoul2
        offsets `ab`. Returns (bones parent first, {bone: [loc per frame]}, {bone: [quat per frame]})."""
        arm = obj.data
        # The jediacademy exporter reads poses as skeleton_root.matrix_local @ pose_bone.matrix, so
        # Ghoul2 space is the armature's parent space.
        L = obj.matrix_local.copy()
        L_inv = L.inverted()

        ordered = []

        def walk(bone):
            ordered.append(bone)
            for child in bone.children:
                walk(child)
        for bone in arm.bones:
            if bone.parent is None:
                walk(bone)

        idx = {b.name: jka.index.get(b.name) for b in ordered}
        if not any(i is not None for i in idx.values()):
            raise TorImportError("No bone of armature %r is a Jedi Academy _humanoid bone" % obj.name)

        rest = {b.name: b.matrix_local.copy() for b in ordered}
        rest_inv = {n: m.inverted() for n, m in rest.items()}

        num = ab.shape[0]
        locs = {b.name: [] for b in ordered}
        quats = {b.name: [] for b in ordered}
        offsets = {n: [Matrix(ab[f, i].tolist()) for f in range(num)] for n, i in idx.items() if i is not None}

        for fi in range(num):
            posed = {}
            for bone in ordered:
                name = bone.name
                parent = bone.parent
                if name in offsets:
                    posed[name] = L_inv @ offsets[name][fi] @ L @ rest[name]
                elif parent:
                    posed[name] = posed[parent.name] @ rest_inv[parent.name] @ rest[name]
                else:
                    posed[name] = rest[name]

                if parent:
                    basis = rest_inv[name] @ rest[parent.name] @ posed[parent.name].inverted() @ posed[name]
                else:
                    basis = rest_inv[name] @ posed[name]

                loc, quat, _ = basis.decompose()
                prev = quats[name][-1] if quats[name] else None
                if prev is not None:
                    quat.make_compatible(prev)
                locs[name].append(loc)
                quats[name].append(quat)
        return ordered, locs, quats

    def write_action(self, obj, ordered, locs, quats, frames, action_name):
        """Writes basis keys (one LINEAR key per entry of `frames`) into a new action."""
        action = bpy.data.actions.new(action_name)
        action.use_fake_user = True
        fcurves = get_fcurves(obj, action)

        num = len(frames)
        frames = [float(f) for f in frames]
        for bone in ordered:
            name = bone.name
            obj.pose.bones[name].rotation_mode = "QUATERNION"
            escaped = bpy.utils.escape_identifier(name)
            for path, values, size in (
                    ('pose.bones["%s"].location' % escaped, locs[name], 3),
                    ('pose.bones["%s"].rotation_quaternion' % escaped, quats[name], 4)):
                for component in range(size):
                    fc = new_fcurve(fcurves, path, component, name)
                    fc.keyframe_points.add(num)
                    co = [0.0] * (2 * num)
                    co[0::2] = frames
                    co[1::2] = [v[component] for v in values]
                    fc.keyframe_points.foreach_set("co", co)
                    fc.keyframe_points.foreach_set("interpolation", [LINEAR] * num)
                    fc.update()
        return action

    def bake(self, obj, jka, ab, action_name):
        """Writes the animation into a new action, keys 0..N-1."""
        ordered, locs, quats = self.basis_keys(obj, jka, ab)
        return self.write_action(obj, ordered, locs, quats, range(ab.shape[0]), action_name)

    def add_sequence(self, obj, jka, ab, names, start, fps, loop_frame):
        """Bakes one action and adds it as NLA strip(s) - same layout as the dotXSI importer."""
        anim = obj.animation_data_create()
        prev_action = anim.action
        prev_slot = getattr(anim, "action_slot", None)

        num = ab.shape[0]
        if start < 0:  # after the last existing strip
            ends = [s.frame_end for t in anim.nla_tracks for s in t.strips]
            start = int(max(ends)) + 1 if ends else 0

        action = self.bake(obj, jka, ab, names[0])
        slot = getattr(anim, "action_slot", None)

        if hasattr(action, "g2_sequence_prop"):  # jediacademy addon: animation.cfg metadata
            props = action.g2_sequence_prop
            props.loop_start_frame = loop_frame
            props.num_frames = num
            props.fps = int(round(fps))

        # Blender would clip the strip to the action range of the previous assignment
        if hasattr(anim, "action_slot"):
            anim.action_slot = None
        anim.action = None

        actions = [action]
        for name in names[1:]:
            copy = action.copy()  # keeps keys, slots and g2_sequence_prop
            copy.name = name
            copy.use_fake_user = True
            actions.append(copy)

        placed = []
        for act in actions:
            placed.append(self.add_strip(anim, act, slot, start, num))
        anim.use_nla = True

        anim.action = prev_action
        if prev_action is not None and hasattr(anim, "action_slot") and prev_slot is not None:
            try:
                anim.action_slot = prev_slot
            except (RuntimeError, TypeError):
                pass

        end = start + max(0, num - 1)
        scene = self.context.scene
        if self.opt["set_scene_range"]:
            # The .gla export samples the scene range, so it has to include the new sequence
            only_ours = sum(len(t.strips) for t in anim.nla_tracks) == len(placed)
            if only_ours:
                scene.frame_start, scene.frame_end = start, end
                scene.render.fps = int(round(fps))
                scene.render.fps_base = 1.0
            else:
                scene.frame_start = min(scene.frame_start, start)
                scene.frame_end = max(scene.frame_end, end)
            scene.frame_current = start

        our_tracks = {track for _, track in placed}
        solo = [t.name for t in anim.nla_tracks if t.is_solo and t not in our_tracks]
        msg = "Added %s, frames %d-%d%s" % (
            ", ".join("%r on %r" % (s.name, t.name) for s, t in placed), start, end,
            " (loop)" if loop_frame >= 0 else "")
        if solo:
            msg += " (track %r is soloed, un-solo it to see the new strips)" % solo[0]
        self.messages.append(msg)
        print("SWTOR jba:", msg)
        return [s.action for s, _ in placed]

    def add_strip(self, anim, action, slot, start, num):
        """Adds a strip for `action` on the first "Sequences Layer n" track that is free there."""
        if slot is not None and hasattr(action, "slots"):
            # a copied action has its own slots, with the same identifiers
            slot = next((s for s in action.slots if s.identifier == slot.identifier), None)

        strip = None
        layer = 1
        while strip is None:
            if layer > 64:
                raise TorImportError("Could not find a free NLA track at frame %d" % start)
            track_name = "Sequences Layer %d" % layer
            track = anim.nla_tracks.get(track_name) or anim.nla_tracks.new()
            track.name = track_name
            free = all(s.frame_end < start or s.frame_start > start + num - 1 for s in track.strips)
            if free:
                try:
                    strip = track.strips.new(action.name, start, action)
                except RuntimeError:
                    strip = None
            layer += 1

        if hasattr(strip, "action_slot") and slot is not None:
            strip.action_slot = slot
        strip.action_frame_start = 0
        strip.action_frame_end = max(0, num - 1)
        strip.frame_start = start
        strip.frame_end = start + max(0, num - 1)
        strip.scale = 1.0
        strip.repeat = 1.0
        strip.extrapolation = "NOTHING"
        strip.blend_type = "REPLACE"
        strip.use_auto_blend = False
        strip.blend_in = strip.blend_out = 0.0
        strip.select = True
        return strip, track

    # ------------------------------------------------------------------ additive layers
    def find_target_strip(self, obj):
        """The NLA strip an additive goes on: the selected one (the active one when several are
        selected). Returns ((track name, strip name), None) or (None, error message)."""
        anim = obj.animation_data
        picked = [(t, s) for t in anim.nla_tracks for s in t.strips if s.select] if anim else []
        if len(picked) > 1:
            active = [(t, s) for t, s in picked if getattr(s, "active", False)]
            if len(active) == 1:
                picked = active
        if not picked:
            return None, "no NLA strip is selected: select the strip the additive animation goes on"
        if len(picked) > 1:
            shown = ", ".join("%r" % s.name for _, s in picked[:4]) + (", ..." if len(picked) > 4 else "")
            return None, ("%d NLA strips are selected (%s): select only the one the additive animation "
                          "goes on" % (len(picked), shown))
        track, strip = picked[0]
        if strip.type != "CLIP" or strip.action is None:
            return None, "the selected NLA strip %r is not an action strip" % strip.name
        return (track.name, strip.name), None

    def add_additive(self, obj, jka, path, target, name):
        """Bakes an ad_* file as pose bone basis deltas and adds it, COMBINE blended, on a new
        track right above the target strip's track, starting at the target's first frame.

        The delta of a SWTOR additive is applied in the bone's own frame after the base
        animation (local = base_local * delta), so the retargeter is run on the rest pose with
        every bone's rest rotation turned by its delta, and on the plain rest pose. Their JKA basis
        difference (rest^-1 * additive) is the delta to COMBINE, whose quaternion product base * delta
        (and summed location) is the same composition. The action carries no g2_sequence_prop, so
        it is not a sequence of its own."""
        anim = obj.animation_data
        base_track = anim.nla_tracks.get(target[0])
        base = base_track.strips.get(target[1]) if base_track is not None else None
        if base is None:
            raise TorImportError("The selected NLA strip %r is gone" % target[1])

        jba, rig, skel = self.load_swtor(path)
        pose_frame = self.opt["additive_frame"]
        if pose_frame >= jba.num_frames:
            raise TorImportError("%s has %d frames, there is no frame %d" % (jba.name, jba.num_frames, pose_frame))
        base_props = getattr(base.action, "g2_sequence_prop", None)
        fps = self.opt["fps"] or int(getattr(base_props, "fps", 0) or 0)  # same time base as the strip
        if fps > 0 and pose_frame < 0:
            jba.resample(fps)

        rt = Retargeter(jka, skel, horizontal="none", fingers=self.opt["fingers"], ground=False)
        frames = None if pose_frame < 0 else [pose_frame]
        ab = rt.convert_abs(Pose(AdditiveTracks(jba, rig, skel, frames), rig, skel))
        ab0 = rt.convert_abs(Pose(AdditiveTracks(jba, rig, skel, deltas=False), rig, skel))
        ordered, locs0, quats0 = self.basis_keys(obj, jka, ab0)
        _, locs_a, quats_a = self.basis_keys(obj, jka, ab)

        num = ab.shape[0]
        locs, quats = {}, {}
        for bone in ordered:
            n = bone.name
            inv0, loc0 = quats0[n][0].inverted(), locs0[n][0]
            prev, deltas = Quaternion(), []
            for q in quats_a[n]:
                d = inv0 @ q
                d.normalize()
                d.make_compatible(prev)
                deltas.append(d)
                prev = d
            quats[n] = deltas
            locs[n] = [loc - loc0 for loc in locs_a[n]]

        # Length: the additive plays once from the first frame of the target strip (HOLD then
        # keeps its last pose until the end of the target, LOOP repeats it to cover the target),
        # and is cut at the end of the target when longer. A one pose additive is held.
        start = base.frame_start
        cover = max(1.0, base.frame_end - base.frame_start)
        span = num - 1
        mode = self.opt["additive_length"]
        keys = list(range(num))
        repeat = 1.0
        if span == 0 or (mode == "HOLD" and cover > span):
            keys.append(cover)
            for n in locs:
                locs[n].append(locs[n][-1].copy())
                quats[n].append(quats[n][-1].copy())
            action_end = length = cover
        elif mode == "LOOP":
            action_end, length, repeat = span, cover, cover / span
        else:
            action_end = length = min(span, cover)

        prev_action = anim.action
        prev_slot = getattr(anim, "action_slot", None)
        action = self.write_action(obj, ordered, locs, quats, keys, name or jba.name)
        slot = getattr(anim, "action_slot", None)
        if hasattr(anim, "action_slot"):
            anim.action_slot = None
        anim.action = None

        track = anim.nla_tracks.new(prev=base_track)
        layer = 1
        while "Additive Layer %d" % layer in anim.nla_tracks:
            layer += 1
        track.name = "Additive Layer %d" % layer
        strip = track.strips.new(action.name, int(start), action)
        if hasattr(strip, "action_slot") and slot is not None:
            strip.action_slot = slot
        strip.action_frame_start = 0
        strip.action_frame_end = action_end
        strip.scale = 1.0
        strip.repeat = repeat
        strip.frame_start = start
        if abs(strip.frame_end - (start + length)) > 1e-4:
            strip.frame_end = start + length
        strip.extrapolation = "NOTHING"
        strip.blend_type = "COMBINE"
        strip.use_auto_blend = False
        strip.blend_in = strip.blend_out = 0.0
        strip.influence = 1.0
        strip.select = False
        anim.use_nla = True

        anim.action = prev_action
        if prev_action is not None and hasattr(anim, "action_slot") and prev_slot is not None:
            try:
                anim.action_slot = prev_slot
            except (RuntimeError, TypeError):
                pass

        msg = "Added additive %r on %r above %r, frames %d-%d (%s)" % (
            strip.name, track.name, base.name, round(strip.frame_start), round(strip.frame_end),
            "pose %d" % pose_frame if pose_frame >= 0 else mode.lower())
        if base_track.is_solo:
            msg += " (track %r is soloed, un-solo it to see the additive)" % base_track.name
        self.messages.append(msg)
        print("SWTOR jba:", msg)
        return [action]


def load(operator, context, paths, **opt):
    importer = Importer(operator, context, opt)
    try:
        obj, actions = importer.run(paths)
    except (TorImportError, OSError) as e:
        operator.report({"ERROR"}, str(e))
        return {"CANCELLED"}

    for error in importer.errors:
        operator.report({"ERROR"}, error)
    if importer.errors and not importer.messages:
        return {"CANCELLED"}

    message = " | ".join(importer.messages) or "Nothing imported"
    if importer.warnings:
        operator.report({"WARNING"}, " - ".join(importer.warnings + [message]))
    else:
        operator.report({"INFO"}, message)
    return {"FINISHED"}
