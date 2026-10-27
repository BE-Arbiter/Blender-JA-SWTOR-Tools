"""Retarget SWTOR humanoid poses onto the Jedi Academy _humanoid skeleton.

Both worlds are Z-up and face -Y with the character's left on +X once SWTOR's GOD bone
is left at identity (as every animation does), so no axis change is needed, only a scale.

For each mapped JKA bone j driven by SWTOR bone s:
    world delta   D_s(t) = R_s(t) * R_s(rest)^-1
    rest fix      A_j    = shortest arc turning JKA's rest bone direction onto SWTOR's
                           rest bone direction (limbs only - corrects A-pose differences)
    absOffset_j rotation = D_s(t) * A_j
Joint positions are rebuilt down the JKA hierarchy with JKA bone lengths, so the result
never stretches the JKA mesh; only the pelvis position comes from SWTOR (scaled).
"""
import numpy as np

from .qmath import qmul, qconj, qnorm, q2m, m2q, q_between
from .swtor import Pose

# JKA bone -> (SWTOR rotation bone, SWTOR bone giving the direction, or None for "no rest fix")
# Unlisted JKA bones stay rigidly attached to their parent.
BONE_MAP = {
    'pelvis': ('Pelvis', None),
    'lfemurYZ': ('LeftHip', 'LeftKnee'),
    'lfemurX': ('LeftHipTwist1', 'LeftKnee'),
    'ltibia': ('LeftKnee', 'LeftAnkle'),
    'ltalus': ('LeftAnkle', 'LeftToe'),
    'rfemurYZ': ('RightHip', 'RightKnee'),
    'rfemurX': ('RightHipTwist1', 'RightKnee'),
    'rtibia': ('RightKnee', 'RightAnkle'),
    'rtalus': ('RightAnkle', 'RightToe'),
    'lower_lumbar': ('LowerBack', None),
    'upper_lumbar': ('Chest', None),
    'thoracic': ('Chest2', None),
    'cervical': ('Neck', None),
    'cranium': ('Head', None),
    'face': ('Head', None),
    'jaw': ('fc_jaw', None),
    'leye': ('fc_eye_left', None),
    'reye': ('fc_eye_right', None),
    'lclavical': ('LeftCollar', 'LeftShoulder'),
    'lhumerus': ('LeftShoulder', 'LeftElbow'),
    'lhumerusX': ('LeftShoulderTwist1', 'LeftElbow'),
    'lradius': ('LeftElbow', 'LeftWrist'),
    'lradiusX': ('LeftUlna', 'LeftWrist'),
    'lhand': ('LeftWrist', 'LeftMiddleFinger'),
    'l_d1_j1': ('LeftThumbFinger', 'LeftThumbFinger1'),
    'l_d1_j2': ('LeftThumbFinger1', 'LeftThumbFinger2'),
    'l_d2_j1': ('LeftIndexFinger', 'LeftIndexFinger1'),
    'l_d2_j2': ('LeftIndexFinger1', 'LeftIndexFinger2'),
    'l_d4_j1': ('LeftRingFinger', 'LeftRingFinger1'),
    'l_d4_j2': ('LeftRingFinger1', 'LeftRingFinger2'),
    'rclavical': ('RightCollar', 'RightShoulder'),
    'rhumerus': ('RightShoulder', 'RightElbow'),
    'rhumerusX': ('RightShoulderTwist1', 'RightElbow'),
    'rradius': ('RightElbow', 'RightWrist'),
    'rradiusX': ('RightUlna', 'RightWrist'),
    'rhand': ('RightWrist', 'RightMiddleFinger'),
    'r_d1_j1': ('RightThumbFinger', 'RightThumbFinger1'),
    'r_d1_j2': ('RightThumbFinger1', 'RightThumbFinger2'),
    'r_d2_j1': ('RightIndexFinger', 'RightIndexFinger1'),
    'r_d2_j2': ('RightIndexFinger1', 'RightIndexFinger2'),
    'r_d4_j1': ('RightRingFinger', 'RightRingFinger1'),
    'r_d4_j2': ('RightRingFinger1', 'RightRingFinger2'),
    # weapon attachment points follow the SWTOR weapon bones axis for axis (see WEAPON_AXES)
    'rhang_tag_bone': ('RightWeapon', '@weapon'),
    'lhang_tag_bone': ('LeftWeapon', '@weapon'),
}

# JKA bones parented to the forearm in the GLA that must move with the hand instead.
HAND_CHILDREN = {
    'rhand': ('r_d1_j1', 'r_d1_j2', 'r_d2_j1', 'r_d2_j2', 'r_d4_j1', 'r_d4_j2', 'rhang_tag_bone'),
    'lhand': ('l_d1_j1', 'l_d1_j2', 'l_d2_j1', 'l_d2_j2', 'l_d4_j1', 'l_d4_j2', 'lhang_tag_bone'),
}
FINGERS = {b for kids in HAND_CHILDREN.values() for b in kids if 'tag' not in b}
POSITION_PARENT = {c: h for h, kids in HAND_CHILDREN.items() for c in kids}
# The GLA parents the humerus to the thoracic, but every stock animation keeps it at the tip of
# the clavicle (and the jediacademy addon's skeleton fixes connect it there).
POSITION_PARENT.update({'rhumerus': 'rclavical', 'lhumerus': 'lclavical'})
POSITION_PARENT.update({'r_d1_j2': 'r_d1_j1', 'r_d2_j2': 'r_d2_j1', 'r_d4_j2': 'r_d4_j1',
                        'l_d1_j2': 'l_d1_j1', 'l_d2_j2': 'l_d2_j1', 'l_d4_j2': 'l_d4_j1'})

# SWTOR weapon bone axes -> JKA hang tag axes, measured on firing / stance animations of
# both games: SWTOR X, +Y (top of a gun, saber blade), +Z (barrel) are JKA tag X, +Z, -Y.
# Columns are the tag axes expressed in the weapon bone's frame.
WEAPON_AXES = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])

MODEL_ROOT_OFFSET = np.array([0.0, 0.0, -24.0])  # every stock _humanoid frame uses this
# Height of the ankle (talus) above the floor in stock standing anims (BOTH_STAND1: -19.8
# with the floor at -24); the GLA base pose itself floats ~3.3 units higher.
JKA_ANKLE_HEIGHT = 4.2


def swtor_rest(skel):
    """SWTOR rest pose with GOD at identity (as in the animations)."""
    saved = skel.rest_rot.copy()
    for n in ('Master', 'GOD'):
        if n in skel.index:
            skel.rest_rot[skel.index[n]] = [1.0, 0, 0, 0]
    rot, pos = Pose.rest(skel)
    skel.rest_rot[:] = saved
    return rot, pos


class Retargeter:
    def __init__(self, gla, skel, bone_map=None, horizontal='keep', fingers=False,
                 ground=True):
        self.gla = gla
        self.skel = skel
        self.map = dict(BONE_MAP if bone_map is None else bone_map)
        self.horizontal = horizontal
        if not fingers:
            for b in FINGERS:
                self.map.pop(b, None)
        self.s_rest_rot, self.s_rest_pos = swtor_rest(skel)
        self.j_rest_pos = gla.rest_positions()
        self.j_rest = gla.rest_matrices()
        # drop mappings whose bones do not exist on either side
        ok = lambda c: c is None or c[0] in '=@' or c in skel.index
        self.map = {j: (s, c) for j, (s, c) in self.map.items()
                    if j in gla.index and s in skel.index and ok(c)}
        self.scale = self._leg_ratio()
        self.ground_fix = 0.0
        if ground and 'ltalus' in gla.index:
            self.ground_fix = JKA_ANKLE_HEIGHT - self.j_rest_pos[gla.index['ltalus']][2]
        self.fix = {}
        for j, (s, c) in self.map.items():
            if c == '@weapon':
                # tag world rotation = weapon world rotation @ WEAPON_AXES, so
                # absOffset = D(t) @ (weapon_rest @ WEAPON_AXES @ tag_rest^-1)
                tag = self.j_rest[gla.index[j]][:3, :3]
                u, _, vt = np.linalg.svd(tag)
                tag = u @ vt
                w = q2m(self.s_rest_rot[skel.index[s]])
                self.fix[j] = m2q(w @ WEAPON_AXES @ tag.T)
                continue
            if c is None or c.startswith('='):
                self.fix[j] = np.array([1.0, 0, 0, 0])
                continue
            ji = gla.index[j]
            d_j = self.j_rest[ji][:3, 0]  # GLA bones point along their local X axis
            d_s = self.s_rest_pos[skel.index[c]] - self.s_rest_pos[skel.index[s]]
            self.fix[j] = q_between(d_j, d_s)
        for j, (s, c) in self.map.items():
            if c is not None and c.startswith('=') and c[1:] in self.fix:
                self.fix[j] = self.fix[c[1:]]

    def _leg_ratio(self):
        g, s = self.gla.index, self.skel.index
        lj = sum(np.linalg.norm(self.j_rest_pos[g[b]] - self.j_rest_pos[g[a]])
                 for a, b in (('lfemurYZ', 'ltibia'), ('ltibia', 'ltalus')))
        ls = sum(np.linalg.norm(self.s_rest_pos[s[b]] - self.s_rest_pos[s[a]])
                 for a, b in (('LeftHip', 'LeftKnee'), ('LeftKnee', 'LeftAnkle')))
        return lj / ls

    def convert(self, pose):
        """pose: swtor.Pose -> parent-relative offsets [frames, jkaBones, 4, 4] (GLA frames)."""
        ab = self.convert_abs(pose)
        local = np.zeros_like(ab)
        for i, par in enumerate(self.gla.parents):
            local[:, i] = ab[:, i] if par < 0 else np.linalg.inv(ab[:, par]) @ ab[:, i]
        return local

    def convert_abs(self, pose):
        """pose: swtor.Pose -> absolute Ghoul2 offsets [frames, jkaBones, 4, 4]: the rigid
        model-space transform taking each bone from its base pose to its animated pose
        (model_root's (0, 0, -24) origin included)."""
        gla, skel = self.gla, self.skel
        nf = pose.rot.shape[0]
        nb = gla.num_bones
        parents = gla.parents
        pel_s = skel.index['Pelvis']
        pel_j = gla.index['pelvis']
        s_pel0 = self.s_rest_pos[pel_s]
        delta_pel = (pose.pos[:, pel_s] - s_pel0) * self.scale
        if self.horizontal == 'center':
            delta_pel[:, :2] -= delta_pel[:, :2].mean(0)
        elif self.horizontal == 'none':
            delta_pel[:, :2] = 0
        delta_pel[:, 2] += self.ground_fix

        # per-bone world deltas (JKA world == SWTOR world here)
        rot_off = {}
        for j, (s, _c) in self.map.items():
            si = skel.index[s]
            d = qmul(pose.rot[:, si], qconj(self.s_rest_rot[si]))
            rot_off[j] = q2m(qnorm(qmul(d, self.fix[j])))

        ab = np.zeros((nf, nb, 4, 4))
        ab[:, :, 3, 3] = 1
        for i in _position_order(gla.names, parents):
            name = gla.names[i]
            par = parents[i]
            p0 = self.j_rest_pos[i]
            if par < 0:
                ab[:, i, :3, :3] = np.eye(3)
                ab[:, i, :3, 3] = MODEL_ROOT_OFFSET
                continue
            ppar = gla.index.get(POSITION_PARENT.get(name), par)
            target = np.einsum('fij,j->fi', ab[:, ppar, :3, :3], p0) + ab[:, ppar, :3, 3]
            if name == 'pelvis':
                target = target + delta_pel
            if name == 'Motion':
                r = np.tile(np.eye(3), (nf, 1, 1))
                pp = self.j_rest_pos[pel_j]
                pel_now = np.einsum('fij,j->fi', ab[:, pel_j, :3, :3], pp) + ab[:, pel_j, :3, 3]
                target = pel_now + (p0 - pp)
            elif name in rot_off:
                r = rot_off[name]
            else:
                ab[:, i] = ab[:, ppar]
                continue
            ab[:, i, :3, :3] = r
            ab[:, i, :3, 3] = target - np.einsum('fij,j->fi', r, p0)
        return ab


def _position_order(names, parents):
    """Parent-first order that also puts each bone after its POSITION_PARENT."""
    index = {n: i for i, n in enumerate(names)}
    deps = []
    for i, n in enumerate(names):
        d = {parents[i]} if parents[i] >= 0 else set()
        if n in POSITION_PARENT and POSITION_PARENT[n] in index:
            d.add(index[POSITION_PARENT[n]])
        if n == 'Motion' and 'pelvis' in index:
            d.add(index['pelvis'])
        deps.append(d)
    order, done = [], set()
    while len(order) < len(names):
        for i in range(len(names)):
            if i not in done and deps[i] <= done:
                order.append(i)
                done.add(i)
    return order
