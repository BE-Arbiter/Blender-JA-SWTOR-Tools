"""Jedi Academy Ghoul2 animation (.gla, "2LGA" v6) reader / writer.

Frame layout: numFrames x numBones 3-byte indices into a pool of 14-byte compressed
bones (quaternion w,x,y,z and translation as u16). A compressed bone is the bone's
offset *relative to its parent's offset*; the posed model-space transform of a bone is
absOffset @ basePose, with absOffset = absOffset(parent) @ offset.
"""
import struct

import numpy as np

from .qmath import q2m, m2q

QUAT_SCALE, QUAT_BIAS = 16383.0, 2.0
XLAT_STEPS, XLAT_BIAS = 64.0, 512.0
HEADER_SIZE = 4 + 4 + 64 + 4 * 7


class GLA:
    def __init__(self, path):
        with open(path, 'rb') as f:
            d = f.read()
        ident, ver = struct.unpack_from('<4si', d, 0)
        if ident != b'2LGA' or ver != 6:
            raise ValueError(f"{path}: not a Jedi Academy GLA (v6)")
        self.name = d[8:72].split(b'\0')[0].decode('latin1')
        (self.scale, self.num_frames, ofs_frames, nb, ofs_pool,
         self.ofs_skel, ofs_end) = struct.unpack_from('<f6i', d, 72)
        self.num_bones = nb
        offs = struct.unpack_from('<%di' % nb, d, HEADER_SIZE)
        self.skel_bytes = d[HEADER_SIZE:ofs_frames]
        self.bones = []
        for i in range(nb):
            p = HEADER_SIZE + offs[i]
            name = d[p:p + 64].split(b'\0')[0].decode('latin1')
            flags, parent = struct.unpack_from('<Ii', d, p + 64)
            base = np.array(struct.unpack_from('<12f', d, p + 72)).reshape(3, 4)
            nc = struct.unpack_from('<i', d, p + 168)[0]
            children = list(struct.unpack_from('<%di' % nc, d, p + 172))
            self.bones.append(dict(name=name, flags=flags, parent=parent, base=base, children=children))
        self.names = [b['name'] for b in self.bones]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parents = [b['parent'] for b in self.bones]
        raw = np.frombuffer(d, dtype=np.uint8, count=self.num_frames * nb * 3, offset=ofs_frames)
        raw = raw.reshape(-1, 3).astype(np.uint32)
        self.frame_indices = (raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)).reshape(self.num_frames, nb)
        npool = (ofs_end - ofs_pool) // 14
        self.pool = np.frombuffer(d, dtype='<u2', count=npool * 7, offset=ofs_pool).reshape(npool, 7).copy()

    # ---- rest pose -------------------------------------------------------------------
    def rest_positions(self):
        return np.array([b['base'][:, 3] for b in self.bones])

    def rest_matrices(self):
        out = []
        for b in self.bones:
            m = np.eye(4)
            m[:3, :] = b['base']
            out.append(m)
        return np.array(out)

    # ---- decoding --------------------------------------------------------------------
    @staticmethod
    def decode_pool(entries):
        e = np.asarray(entries, dtype=np.float64)
        q = e[..., :4] / QUAT_SCALE - QUAT_BIAS
        t = e[..., 4:] / XLAT_STEPS - XLAT_BIAS
        m = np.zeros(e.shape[:-1] + (4, 4))
        m[..., :3, :3] = q2m(q)
        m[..., :3, 3] = t
        m[..., 3, 3] = 1
        return m

    def frame_abs_offsets(self, frame):
        local = self.decode_pool(self.pool[self.frame_indices[frame]])
        return abs_from_local(local, self.parents)

    def frame_positions(self, frame):
        ab = self.frame_abs_offsets(frame)
        rest = self.rest_matrices()
        return np.einsum('bij,bjk->bik', ab, rest)[:, :3, 3]

    # ---- writing ---------------------------------------------------------------------
    def save_with_appended(self, path, new_local_offsets):
        """Write a copy of this GLA with extra frames appended.

        new_local_offsets: array [numNewFrames, numBones, 4, 4] of parent-relative offsets.
        Returns (firstNewFrame, clampedCount)."""
        nb = self.num_bones
        pool = [bytes(r.tobytes()) for r in self.pool]
        lookup = {}
        for i, b in enumerate(pool):
            lookup.setdefault(b, i)
        new_idx = np.zeros(new_local_offsets.shape[:2], dtype=np.uint32)
        clamped = 0
        for f in range(new_local_offsets.shape[0]):
            for b in range(nb):
                raw, c = compress(new_local_offsets[f, b])
                clamped += c
                i = lookup.get(raw)
                if i is None:
                    i = len(pool)
                    pool.append(raw)
                    lookup[raw] = i
                new_idx[f, b] = i
        if len(pool) >= (1 << 24):
            raise ValueError("compressed bone pool overflow (more than 16M unique bones)")
        first = self.num_frames
        idx = np.concatenate([self.frame_indices, new_idx]).astype(np.uint32)
        nf = idx.shape[0]
        packed = np.stack([idx & 255, (idx >> 8) & 255, (idx >> 16) & 255], -1).astype(np.uint8).tobytes()
        ofs_frames = HEADER_SIZE + len(self.skel_bytes)
        ofs_pool = ofs_frames + len(packed)
        pad = (-ofs_pool) % 4
        ofs_pool += pad
        pool_bytes = b''.join(pool)
        ofs_end = ofs_pool + len(pool_bytes)
        with open(path, 'wb') as f:
            f.write(struct.pack('<4si64sf6i', b'2LGA', 6, self.name.encode('latin1'), self.scale,
                                nf, ofs_frames, nb, ofs_pool, self.ofs_skel, ofs_end))
            f.write(self.skel_bytes)
            f.write(packed)
            f.write(b'\0' * pad)
            f.write(pool_bytes)
        return first, clamped


def abs_from_local(local, parents):
    ab = np.zeros_like(local)
    done = [False] * len(parents)
    while not all(done):
        for i, p in enumerate(parents):
            if done[i] or (p >= 0 and not done[p]):
                continue
            ab[i] = local[i] if p < 0 else ab[p] @ local[i]
            done[i] = True
    return ab


def compress(m):
    q = m2q(m[:3, :3])
    t = m[:3, 3]
    clamped = 0
    vals = []
    for c in q:
        vals.append(int(round((c + QUAT_BIAS) * QUAT_SCALE)))
    for c in t:
        if not -511.0 <= c <= 511.0:
            clamped += 1
            c = max(-511.0, min(511.0, c))
        vals.append(int(round((c + XLAT_BIAS) * XLAT_STEPS)))
    vals = [min(65535, max(0, v)) for v in vals]
    return struct.pack('<7H', *vals), clamped
