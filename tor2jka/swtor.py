"""Readers for Star Wars: The Old Republic (64-bit client, game update 7.2.1+) assets:

* .jba  - "JAWB" v2 joint/bone animations (reverse engineered here; the public
           SWTOR Slicers importer only handles the older 32-bit layout)
* .gr2  - "GAWB" v5 skeletons (only the bone table is read)
* .mph  - Morpheme networks, only used to recover the rig bone order, because
           64-bit .jba files no longer store bone names.

JBA "JAWB" v2 layout (all offsets absolute, every "align" is 128 bytes relative to
offset 8, i.e. the 8-byte magic/version prefix is not part of the aligned image):

    0x00  "JAWB"  u32 version(2)  u64 0
    0x10  f32 length (s)   f32 fps   u32 numBlocks  ...
    0x30  u32 numBones
    0x50  numBlocks x (u32 startFrame, u32 blockSize)
    align(0x50 + 16*numBlocks):  numBones x 48-byte quantisation records
          (translation stride, translation base, rotation stride, rotation base)
    align(...) per block:
          u32 numBones, u32 0, u64 0
          numBones x 32-byte entries (u32 numRot, u32 0, u64 0, u32 numTrans, u32 0, u64 0)
          per bone: numRot x 6-byte rotations, pad to 4, numTrans x 4-byte translations
    align(...) world-space (CharacterWorldSpaceTM) track:
          u64 0, f32 fps, ts[3], tb[3], rs[3], rb[3], pad,
          u32 numRot, u32 0, u64 0, u32 numTrans, u32 0, u64 0, rotations, pad, translations

Rotation key: three u16; x uses the low 15 bits, bit 15 is the sign of w.
Translation key: one u32 packed 11/11/10 bits (x, y, z).
Consecutive blocks share their boundary frame.
All animation data lives in "anim space": Z up, centimetres.
"""
import glob
import os
import re
import struct

import numpy as np

from .qmath import qmul, qrot, qnorm, m2q

# skeleton space (Y up, 1/1000 of anim units) -> anim space (Z up, cm)
SKEL_TO_ANIM = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
SKEL_SCALE = 1000.0


def _align(x, a=128):
    return 8 + ((x - 8 + a - 1) & ~(a - 1))


class JBA:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'rb') as f:
            d = f.read()
        if d[:4] != b'JAWB':
            if struct.unpack_from('<I', d, 0)[0] == 0:
                raise ValueError(f"{path}: old 32-bit JBA format is not supported by this reader")
            raise ValueError(f"{path}: not a SWTOR JBA file")
        U = lambda o: struct.unpack_from('<I', d, o)[0]
        F = lambda o: struct.unpack_from('<f', d, o)[0]
        self.version = U(4)
        self.length = F(0x10)
        self.fps = F(0x14)
        nblk = U(0x18)
        nb = U(0x30)
        self.num_bones = nb
        nf = int(round(self.length * self.fps)) + 1
        self.num_frames = nf
        blocks = [(U(0x50 + 8 * i), U(0x54 + 8 * i)) for i in range(nblk)]

        rec = _align(0x50 + 16 * nblk)
        q = np.array(struct.unpack_from('<%df' % (12 * nb), d, rec), dtype=np.float64).reshape(nb, 4, 3)
        t_stride, t_base, r_stride, r_base = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        self.t_base = t_base

        rot = np.zeros((nb, nf, 4))
        trans = np.zeros((nb, nf, 3))
        has_trans = np.zeros(nb, dtype=bool)
        pos = _align(rec + 48 * nb)
        for bi, (start, _size) in enumerate(blocks):
            if U(pos) != nb:
                raise ValueError(f"{path}: block {bi} header mismatch at 0x{pos:x}")
            bf = (blocks[bi + 1][0] + 1 if bi + 1 < nblk else nf) - start
            ent = np.frombuffer(d, dtype='<u4', count=8 * nb, offset=pos + 0x10).reshape(nb, 8)
            p = pos + 0x10 + 32 * nb
            for k in range(nb):
                nr, nt = int(ent[k, 0]), int(ent[k, 4])
                n = min(nr, bf)
                raw = np.frombuffer(d, dtype='<u2', count=3 * nr, offset=p).reshape(nr, 3)[:n].astype(np.float64)
                rot[k, start:start + n] = _decode_rot(raw, r_base[k], r_stride[k])
                p += nr * 6
                p = (p + 3) & ~3
                if nt:
                    has_trans[k] = True
                    raw = np.frombuffer(d, dtype='<u4', count=nt, offset=p)[:min(nt, bf)]
                    trans[k, start:start + len(raw)] = _decode_trans(raw, t_base[k], t_stride[k])
                    p += nt * 4
                else:
                    trans[k, start:start + bf] = t_base[k] + np.array([2047, 2047, 1023]) * t_stride[k]
            pos = _align(p)
        self.rot = rot
        self.trans = trans
        self.has_trans = has_trans

        # world-space (CharacterWorldSpaceTM) track = root motion
        self.world_rot = np.tile([1.0, 0, 0, 0], (nf, 1))
        self.world_trans = np.zeros((nf, 3))
        try:
            w = pos
            wq = np.array(struct.unpack_from('<12f', d, w + 0x0c)).reshape(4, 3)
            nr, nt = U(w + 0x40), U(w + 0x50)
            if 0 < nr <= nf and nt <= nf:
                p = w + 0x60
                raw = np.frombuffer(d, dtype='<u2', count=3 * nr, offset=p).reshape(nr, 3).astype(np.float64)
                self.world_rot[:nr] = _decode_rot(raw, wq[3], wq[2])
                p = (p + nr * 6 + 3) & ~3
                if nt:
                    raw = np.frombuffer(d, dtype='<u4', count=nt, offset=p)
                    self.world_trans[:nt] = _decode_trans(raw, wq[1], wq[0])
        except struct.error:
            pass

    def resample(self, fps):
        """Resample every track to a new frame rate (nlerp / lerp)."""
        if not fps or abs(fps - self.fps) < 1e-6:
            return
        nf = max(2, int(round(self.length * fps)) + 1)
        src = np.linspace(0, self.num_frames - 1, nf)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, self.num_frames - 1)
        w = (src - i0)[:, None]

        def rot(q):
            q = _hemisphere(q)
            return qnorm(q[..., i0, :] * (1 - w) + q[..., i1, :] * w)

        def lin(v):
            return v[..., i0, :] * (1 - w) + v[..., i1, :] * w
        self.rot = rot(self.rot)
        self.trans = lin(self.trans)
        self.world_rot = rot(self.world_rot)
        self.world_trans = lin(self.world_trans)
        self.num_frames = nf
        self.fps = float(fps)

    def is_loop(self, max_deg=3.0, max_cm=1.0):
        """True when the last frame repeats the first (SWTOR loops store it twice)."""
        if self.num_frames < 3:
            return False
        a, b = self.rot[:, 0], self.rot[:, -1]
        dot = np.abs(np.sum(a * b, -1)).clip(0, 1)
        ang = np.degrees(2 * np.arccos(dot)).max()
        dt = np.linalg.norm(self.trans[:, 0] - self.trans[:, -1], axis=-1).max()
        return ang < max_deg and dt < max_cm


def _hemisphere(q):
    q = np.array(q)
    d = np.sum(q[..., 1:, :] * q[..., :-1, :], -1)
    sign = np.cumprod(np.where(d < 0, -1.0, 1.0), axis=-1)
    q[..., 1:, :] *= sign[..., None]
    return q


def _decode_rot(raw, base, stride):
    rx = raw[:, 0]
    neg = rx >= 32768
    x = base[0] + np.where(neg, rx - 32768, rx) * stride[0]
    y = base[1] + raw[:, 1] * stride[1]
    z = base[2] + raw[:, 2] * stride[2]
    dot = x * x + y * y + z * z
    w = np.sqrt(np.clip(1.0 - dot, 0.0, None))
    w = np.where(neg, -w, w)
    return qnorm(np.stack([w, x, y, z], -1))


def _decode_trans(raw, base, stride):
    raw = raw.astype(np.int64)
    x = base[0] + (raw >> 21) * stride[0]
    y = base[1] + ((raw >> 10) & 2047) * stride[1]
    z = base[2] + (raw & 1023) * stride[2]
    return np.stack([x, y, z], -1)


class Skeleton:
    """SWTOR .gr2 skeleton, rest pose converted to anim space."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            d = f.read()
        if d[:4] != b'GAWB':
            raise ValueError(f"{path}: not a SWTOR gr2 file")
        ver = struct.unpack_from('<I', d, 4)[0]
        nb = struct.unpack_from('<H', d, 0x1c)[0]
        if ver == 5:
            off = struct.unpack_from('<Q', d, 0x68)[0]
            size = 144
        else:
            off = struct.unpack_from('<I', d, 0x5c)[0]
            size = 136
        self.names, self.parents = [], []
        self.rest_rot, self.rest_trans = [], []
        for i in range(nb):
            p = off + size * i
            if ver == 5:
                no = struct.unpack_from('<Q', d, p)[0]
                p += 8
            else:
                no = struct.unpack_from('<I', d, p)[0]
                p += 4
            name = d[no:d.index(b'\0', no)].decode('latin1')
            par = struct.unpack_from('<i', d, p)[0]
            p += 8 if ver == 5 else 4
            m = np.array(struct.unpack_from('<16f', d, p)).reshape(4, 4)  # row-vector convention
            r = SKEL_TO_ANIM @ m[:3, :3].T @ SKEL_TO_ANIM.T
            t = SKEL_TO_ANIM @ m[3, :3] * SKEL_SCALE
            if name == 'Bip01':
                name = 'GOD'
            self.names.append(name)
            self.parents.append(par)
            self.rest_rot.append(m2q(r))
            self.rest_trans.append(t)
        self.rest_rot = np.array(self.rest_rot)
        self.rest_trans = np.array(self.rest_trans)
        self.index = {n: i for i, n in enumerate(self.names)}

    def parent_name(self, name):
        p = self.parents[self.index[name]]
        return self.names[p] if p >= 0 else None


def find_rigs(anim_dir):
    """All distinct Morpheme rig bone orders found in the .mph files of a folder."""
    rigs = []
    seen = set()
    files = glob.glob(os.path.join(anim_dir, '*.mph'))
    files.sort(key=os.path.getsize)
    for fn in files:
        with open(fn, 'rb') as f:
            d = f.read()
        for m in re.finditer(rb'CharacterWorldSpaceTM\0', d):
            o = m.end()
            names = []
            while True:
                e = d.index(b'\0', o)
                s = d[o:e]
                if not s or not re.fullmatch(rb'[A-Za-z0-9_]+', s):
                    break
                names.append(s.decode())
                o = e + 1
            t = tuple(names)
            if t and t not in seen:
                seen.add(t)
                rigs.append(list(t))
    return rigs


SKIP_PENALTY = 0.05  # cm per rig bone left out, so an exact-length rig wins ties


def align_rig(jba, rig, skel, cap=50.0, max_skip=8):
    """Order-preserving alignment of the animation's tracks onto a rig that may be longer:
    some animations are exported from a bigger rig with a few bones left out (e.g. the
    105-bone rig without fc_wrinkle1-3). Cost = distance between each track's mean
    translation and the skeleton's rest offset of the bone it is given.
    Returns (mean cost in cm, bone name per track) or (None, None)."""
    n, m = jba.num_bones, len(rig)
    if m < n or m > n + max_skip:
        return None, None
    if not hasattr(jba, '_mean_trans'):
        jba._mean_trans = jba.trans.mean(1)
    t = jba._mean_trans
    cost = np.full((n, m), cap)
    for b, name in enumerate(rig):
        i = skel.index.get(name)
        if i is not None:
            cost[:, b] = np.minimum(np.linalg.norm(t - skel.rest_trans[i], axis=1), cap)
    # D[k, b]: best cost of the first k tracks within the first b rig bones, i.e.
    # min(D[k, b-1] + p, D[k-1, b-1] + cost) - solved per row with a running minimum.
    p = SKIP_PENALTY
    ramp = np.arange(m + 1) * p
    D = np.full((n + 1, m + 1), np.inf)
    take = np.zeros((n + 1, m + 1), dtype=bool)  # True: track k-1 is rig bone b-1
    D[0] = ramp
    for k in range(1, n + 1):
        cand = np.full(m + 1, np.inf)
        cand[1:] = D[k - 1, :-1] + cost[k - 1]
        best = np.minimum.accumulate(cand - ramp) + ramp
        D[k] = best
        take[k] = cand <= best + 1e-9
    names = [None] * n
    k, b = n, m
    while k > 0:
        if take[k, b]:
            names[k - 1] = rig[b - 1]
            k -= 1
        b -= 1
    return D[n, m] / n, names


def pick_rig(jba, rigs, skel):
    """Bone name per track: the rig (or rig minus some bones) that best explains the
    animation's translations. Returns (names, mean error in cm)."""
    best, best_err = None, None
    for rig in rigs:
        err, names = align_rig(jba, rig, skel)
        if err is not None and (best_err is None or err < best_err):
            best, best_err = names, err
    return best, best_err


def pick_skeleton_and_rig(jba, skeletons, rigs, preferred=None, prefer_margin=0.1):
    """Animations are shared between body types (a bmfnew .jba can carry bmnnew proportions),
    so pick the (skeleton, rig) pair that explains the animation best.
    skeletons: {body name: Skeleton}. Returns (body, skeleton, names, error)."""
    best, best_score = (None, None, None, None), None
    for body, skel in skeletons.items():
        names, err = pick_rig(jba, rigs, skel)
        if err is None:
            continue
        score = err - (prefer_margin if body == preferred else 0.0)
        if best_score is None or score < best_score:
            best, best_score = (body, skel, names, err), score
    return best


# --- additive ("ad_*") animations ----------------------------------------------------------
# An additive stores parent-relative *deltas*, so the translation-based rig alignment above
# cannot be used: their translations are ~0 and no longer match the skeleton's bone offsets.
# Rigs are chosen by length first (a 102 track additive fits the 102 bone rig, or the 105 bone
# rig without its three fc_wrinkle bones), and when several rigs of the same length exist
# (they differ in the order of the neck / head / left arm / Ulna tracks) by how plausible the
# set of moving bones is, using the fraction of ad_* files in which each kind of bone moves.
ACTIVE_DEG = 0.05  # a track "moves" when it turns more than this
BONE_ACTIVITY = {
    'Master': 0.054, 'GOD': 0.014, 'Root': 0.096, 'Pelvis': 0.057, 'Hip': 0.178,
    'Knee': 0.135, 'Ankle': 0.253, 'Toe': 0.245, 'HipTwist': 0.169, 'Skirt': 0.161,
    'RootSpine': 0.078, 'LowerBack': 0.556, 'Chest': 0.604, 'Collar': 0.265,
    'Shoulder': 0.458, 'Elbow': 0.295, 'Ulna': 0.384, 'Wrist': 0.465, 'ThumbFinger': 0.159,
    'PinkFinger': 0.161, 'IndexFinger': 0.168, 'MiddleFinger': 0.161, 'RingFinger': 0.166,
    'Weapon': 0.212, 'ShoulderTwist': 0.448, 'BlendShoulder': 0.455, 'Neck': 0.468,
    'Head': 0.579, 'fc_cheek': 0.001, 'fc_lid_bottom': 0.062, 'fc_lid_top': 0.055,
    'fc_brow_mid': 0.069, 'fc_brow_1': 0.131, 'fc_brow_2': 0.09, 'fc_eye': 0.057,
    'fc_jaw': 0.14, 'fc_lip_bottom': 0.165, 'fc_tongue': 0.007,
    'fc_lip_center_bottom': 0.148, 'fc_lip_corner': 0.161, 'fc_lip_top': 0.162,
    'fc_puffer': 0.061, 'fc_cheek_inner': 0.004, 'fc_nose': 0.126,
    'fc_lip_center_top': 0.148, 'Camera': 0.03, 'NamePlate': 0.03, 'FxRoot': 0.001,
}


def _activity_key(name):
    """Bone name without side and trailing digits: 'LeftIndexFinger1' -> 'IndexFinger'."""
    if name.startswith('fc_'):
        return re.sub(r'_(left|right)', '', name)
    return re.sub(r'\d+$', '', name.replace('Left', '').replace('Right', ''))


def additive_candidates(n, rigs):
    """Bone orders with exactly `n` bones that an n track additive can have: a rig of that
    length, or a longer one without its fc_wrinkle bones."""
    out = []
    for rig in rigs:
        if len(rig) != n:
            rig = [b for b in rig if not b.startswith('fc_wrinkle')]
            if len(rig) != n:
                continue
        if rig not in out:
            out.append(list(rig))
    return out


def additive_moving(jba, min_deg=ACTIVE_DEG):
    """Bool per track: the delta rotation leaves identity by more than min_deg at some frame."""
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(jba.rot[:, :, 0]), 0, 1)))
    return ang.max(1) > min_deg


def pick_additive_rig(jba, folder_rigs, all_rigs):
    """Bone name per track of an additive. The rigs of the animation's own body folder are
    preferred, then every known rig. Returns (names, n candidates, log-likelihood margin
    over the runner-up, inf when the runner-up moves the same bones) or (None, 0, 0.0)."""
    n = jba.num_bones
    cands = additive_candidates(n, folder_rigs) or additive_candidates(n, all_rigs)
    if not cands:
        return None, 0, 0.0
    if len(cands) == 1:
        return cands[0], 1, float('inf')
    moving = additive_moving(jba)
    scores = []
    for rig in cands:
        p = np.array([min(max(BONE_ACTIVITY.get(_activity_key(b), 0.05), 0.02), 0.98) for b in rig])
        scores.append(float(np.where(moving, np.log(p), np.log(1 - p)).sum()))
    order = np.argsort(scores)[::-1]
    best, second = cands[order[0]], cands[order[1]]
    if not any(moving[k] and best[k] != second[k] for k in range(n)):
        return best, len(cands), float('inf')  # the runner-up would move the very same bones
    return best, len(cands), scores[order[0]] - scores[order[1]]


ROOT_BONES = ('Master', 'GOD', 'Root')  # no meaning for a Jedi Academy pose, never moved by an additive


class AdditiveTracks:
    """JBA-like tracks (rot / trans / num_frames / world_*) that hold the skeleton's rest pose
    with an additive's deltas applied, ready for Pose. Semantics (measured on the ad_* files:
    the bone axis is the axis of spine twists and head pitch): the delta of a track is applied in the bone's own frame, after
    the base animation: local = base_local * delta. On the rest pose that is
    rest_local * delta, and translation deltas are dropped (bones do not stretch in Ghoul2).
    frames=None keeps every frame of the additive, a list picks those frames, and deltas=False
    gives the plain rest pose (one frame), the reference the deltas are measured from."""

    def __init__(self, jba, rig, skel, frames=None, deltas=True):
        if not deltas:
            frames = [0]
        elif frames is None:
            frames = list(range(jba.num_frames))
        nf = len(frames)
        self.num_frames = nf
        track = {name: k for k, name in enumerate(rig)}
        self.rot = np.zeros((len(rig), nf, 4))
        self.trans = np.zeros((len(rig), nf, 3))
        for k, name in enumerate(rig):
            i = skel.index.get(name)
            if i is None:
                self.rot[k] = [1.0, 0, 0, 0]
                continue
            rest = np.tile(skel.rest_rot[i], (nf, 1))
            self.trans[k] = skel.rest_trans[i]
            if name in ('Master', 'GOD'):
                self.rot[k] = [1.0, 0, 0, 0]  # like every animation
            elif deltas and name not in ROOT_BONES:
                self.rot[k] = qnorm(qmul(rest, jba.rot[track[name]][frames]))
            else:
                self.rot[k] = rest
        self.world_rot = np.tile([1.0, 0, 0, 0], (nf, 1))
        self.world_trans = np.zeros((nf, 3))


class Pose:
    """Global (anim space) transforms of every skeleton bone for every frame."""

    def __init__(self, jba, rig, skel, root_motion=False):
        nf = jba.num_frames
        nb = len(skel.names)
        track = {name: k for k, name in enumerate(rig)}
        self.names = skel.names
        self.index = skel.index
        self.rot = np.zeros((nf, nb, 4))
        self.pos = np.zeros((nf, nb, 3))
        order = _parent_first(skel.parents)
        for i in order:
            name = skel.names[i]
            k = track.get(name)
            if k is not None:
                lr, lt = jba.rot[k], jba.trans[k]
            else:
                lr = np.tile(skel.rest_rot[i], (nf, 1))
                lt = np.tile(skel.rest_trans[i], (nf, 1))
            par = skel.parents[i]
            if par < 0:
                if root_motion:
                    pr, pt = jba.world_rot, jba.world_trans
                else:
                    pr, pt = np.tile([1.0, 0, 0, 0], (nf, 1)), np.zeros((nf, 3))
            else:
                pr, pt = self.rot[:, par], self.pos[:, par]
            self.rot[:, i] = qnorm(qmul(pr, lr))
            self.pos[:, i] = pt + qrot(pr, lt)

    @staticmethod
    def rest(skel):
        nb = len(skel.names)
        rot = np.zeros((nb, 4))
        pos = np.zeros((nb, 3))
        for i in _parent_first(skel.parents):
            par = skel.parents[i]
            if par < 0:
                rot[i], pos[i] = skel.rest_rot[i], skel.rest_trans[i]
            else:
                rot[i] = qnorm(qmul(rot[par], skel.rest_rot[i]))
                pos[i] = pos[par] + qrot(rot[par], skel.rest_trans[i])
        return rot, pos


def _parent_first(parents):
    order, done = [], set()
    while len(order) < len(parents):
        for i, p in enumerate(parents):
            if i not in done and (p < 0 or p in done):
                order.append(i)
                done.add(i)
    return order
