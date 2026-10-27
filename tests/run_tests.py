"""Headless test suite: blender --background --factory-startup --python-exit-code 1 --python tests/run_tests.py

Real SWTOR / Jedi Academy files are proprietary and not committed, so the cases cover registration
and the pure readers / math on synthetic data.
"""
import importlib
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testutil import PACKAGE, TestRunner, import_addon  # noqa: E402


def case_smoke() -> None:
    import bpy
    addon = import_addon()
    addon.register()
    try:
        assert hasattr(bpy.ops.import_anim, "swtor_jba"), "import_anim.swtor_jba operator not registered"
    finally:
        addon.unregister()


def case_qmath_roundtrip() -> None:
    qmath = sys.modules[PACKAGE + ".tor2jka.qmath"]
    rng = np.random.default_rng(0)
    q = qmath.qnorm(rng.normal(size=(32, 4)))
    q[q[:, 0] < 0] *= -1  # m2q returns w >= 0
    for qi in q:
        back = qmath.m2q(qmath.q2m(qi))
        assert np.allclose(back, qi, atol=1e-9), "m2q(q2m(q)) != q: %s vs %s" % (back, qi)
    v = rng.normal(size=3)
    for qi in q:
        assert np.allclose(qmath.qrot(qi, v), qmath.q2m(qi) @ v, atol=1e-9), "qrot disagrees with q2m"


def case_jba_rejects_bad_files() -> None:
    swtor = sys.modules[PACKAGE + ".tor2jka.swtor"]
    with tempfile.TemporaryDirectory() as tmp:
        for name, data, expected in (
            ("old.jba", struct.pack("<I", 0) + bytes(60), "old 32-bit"),
            ("junk.jba", b"NOPE" + bytes(60), "not a SWTOR JBA"),
        ):
            path = os.path.join(tmp, name)
            with open(path, "wb") as f:
                f.write(data)
            try:
                swtor.JBA(path)
            except ValueError as e:
                assert expected in str(e), "unexpected error for %s: %s" % (name, e)
            else:
                raise AssertionError("%s was accepted" % name)


def main() -> None:
    import_addon()
    # the operators import the submodules lazily
    for sub in ("tor2jka.qmath", "tor2jka.swtor", "tor2jka.gla", "tor2jka.retarget", "blender_importer"):
        importlib.import_module(PACKAGE + "." + sub)
    runner = TestRunner()
    runner.run("smoke", case_smoke)
    runner.run("qmath_roundtrip", case_qmath_roundtrip)
    runner.run("jba_rejects_bad_files", case_jba_rejects_bad_files)
    runner.report()


main()
