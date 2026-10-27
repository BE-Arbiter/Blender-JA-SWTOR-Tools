import importlib
import os
import sys

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, CollectionProperty
from bpy_extras.io_utils import ImportHelper

bl_info = {
    "name": "JA SWTOR Anim Import (.jba)",
    "author": "BE-Arbiter",
    "version": (0, 1, 0),
    "blender": (4, 1, 0),
    "location": "File > Import > JA SWTOR Anim Import (.jba)",
    "description": "Imports Star Wars: The Old Republic 64-bit .jba animations, retargeted onto the Jedi Academy _humanoid skeleton, as NLA strips (additive ad_* files are layered on the selected strip)",
    "category": "Import-Export"
}

# F8 "Reload Scripts": the operator imports the submodules lazily, so reload the ones already loaded
for _name in ("tor2jka.qmath", "tor2jka.swtor", "tor2jka.gla", "tor2jka.retarget", "blender_importer"):
    _module = sys.modules.get("%s.%s" % (__name__, _name))
    if _module is not None:
        importlib.reload(_module)

DEFAULT_GLA = r"C:\jka_animations\GameData\base\models\players\_humanoid\_humanoid.gla"


class ImportSwtorJBA(bpy.types.Operator, ImportHelper):
    """Import SWTOR .jba animations onto a Jedi Academy skeleton as NLA strips. Additive ad_* files are applied on top of the selected NLA strip"""
    bl_idname = "import_anim.swtor_jba"
    bl_label = "Import SWTOR .jba"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".jba"
    filter_glob: StringProperty(default="*.jba", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    gla_path: StringProperty(
        name="Reference .gla",
        description="Jedi Academy .gla whose base pose the animation is retargeted to (normally the one "
                    "imported in the scene). Empty or missing = the armature's own rest pose",
        default=DEFAULT_GLA, subtype="FILE_PATH"
    )

    skeleton_path: StringProperty(
        name="SWTOR Skeleton",
        description="SWTOR <body>_skeleton.gr2 to force. Empty = automatic: every humanoid body skeleton "
                    "(resources/art/dynamic/spec/<body>_skeleton.gr2) is tried and the one matching the "
                    "animation's proportions is used, since animations are shared between body types",
        default="", subtype="FILE_PATH"
    )

    strip_start: IntProperty(
        name="Strip Start",
        description="Frame where the (first) strip starts, -1 = right after the last existing strip",
        default=-1, min=-1
    )

    sequence_name: StringProperty(
        name="Sequence Name",
        description="Name of the action / animation.cfg sequence (e.g. BOTH_STAND2). Several names separated "
                    "by dots create one action per name, stacked at the same frames. Only used when importing "
                    "a single file. Empty = file name",
        default=""
    )

    loop: EnumProperty(
        name="Loop",
        description="animation.cfg loop frame",
        items=(
            ("AUTO", "Auto", "Loop when the last frame repeats the first and the name looks like a cycle (idle, run, walk, loop...); the duplicate end frame is dropped"),
            ("LOOP", "Loop", "Loop from the start (loop frame 0)"),
            ("NONE", "No Loop", "Play once (loop frame -1)"),
        ),
        default="AUTO"
    )

    fps: IntProperty(
        name="Resample FPS",
        description="Resample the animation to this frame rate, 0 = keep SWTOR's (usually 30)",
        default=0, min=0, max=120
    )

    horizontal: EnumProperty(
        name="Pelvis Offset",
        description="What to do with the pelvis horizontal movement",
        items=(
            ("KEEP", "Keep", "Keep SWTOR's pelvis movement relative to the origin"),
            ("CENTER", "Center", "Keep the movement but center it on the origin"),
            ("NONE", "Lock", "Lock the pelvis above the origin horizontally"),
        ),
        default="KEEP"
    )

    root_motion: BoolProperty(
        name="Root Motion",
        description="Apply SWTOR's world-space (CharacterWorldSpaceTM) track: the character travels. "
                    "Jedi Academy moves players itself, so leave off for in-place animations",
        default=False
    )

    fingers: BoolProperty(
        name="Animate Fingers",
        description="Transfer finger animation. Off = fingers stay rigid on the hand, like every stock _humanoid animation",
        default=False
    )

    ground: BoolProperty(
        name="Ground Feet",
        description="Lower the body so standing feet touch the floor like the stock animations "
                    "(the _humanoid base pose floats ~3 units above it)",
        default=True
    )

    additive_length: EnumProperty(
        name="Additive Length",
        description="Additive (ad_*) files: how long the additive layer lasts on the selected NLA strip. "
                    "It always starts on the strip's first frame and is cut at its last frame",
        items=(
            ("HOLD", "Hold", "Play once, then hold the last pose until the end of the selected strip"),
            ("OWN", "Own Length", "Play once and stop, whatever the length of the selected strip"),
            ("LOOP", "Loop", "Repeat until the end of the selected strip"),
        ),
        default="HOLD"
    )

    additive_frame: IntProperty(
        name="Additive Pose Frame",
        description="Additive (ad_*) files: use only this frame of the file, held over the selected strip. "
                    "Aim / lean files (ad_*_yaw, ad_*_pitch, ...) are not time animations but a few poses "
                    "(e.g. left, centre, right), pick one here. -1 = use the whole animation",
        default=-1, min=-1
    )

    set_scene_range: BoolProperty(
        name="Set Scene Range && FPS",
        description="Extend the scene frame range to include the new strips",
        default=True
    )

    def draw(self, context):
        from .blender_importer import find_skeleton
        layout = self.layout

        skeleton = find_skeleton(context)
        if skeleton is not None:
            layout.label(text="Armature: %s" % skeleton.name, icon="ARMATURE_DATA")
        else:
            layout.label(text="No armature: import a .gla first", icon="ERROR")

        box = layout.box()
        box.label(text="Skeletons")
        box.prop(self, "gla_path")
        box.prop(self, "skeleton_path")

        box = layout.box()
        box.label(text="NLA Strip")
        box.prop(self, "strip_start")
        sub = box.column()
        sub.enabled = len(self.files) <= 1
        sub.label(text="Sequence Name:")
        sub.prop(self, "sequence_name", text="")
        box.prop(self, "loop")
        box.prop(self, "fps")

        box = layout.box()
        box.label(text="Retargeting")
        box.prop(self, "horizontal")
        box.prop(self, "root_motion")
        box.prop(self, "fingers")
        box.prop(self, "ground")
        layout.prop(self, "set_scene_range")

        box = layout.box()
        box.label(text="Additive (ad_*) files: layered on the selected NLA strip")
        box.prop(self, "additive_length")
        box.prop(self, "additive_frame")

    def execute(self, context):
        from . import blender_importer
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        paths = [os.path.join(self.directory, f.name) for f in self.files
                 if f.name.lower().endswith(".jba")]
        if not paths:
            paths = [self.filepath]
        keywords = self.as_keywords(ignore=("filter_glob", "filepath", "files", "directory"))
        return blender_importer.load(self, context, paths, **keywords)


def menu_func_import(self, context):
    self.layout.operator(ImportSwtorJBA.bl_idname, text="JA SWTOR Anim Import (.jba)")


def register():
    bpy.utils.register_class(ImportSwtorJBA)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(ImportSwtorJBA)
