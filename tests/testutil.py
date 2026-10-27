import importlib.util
import os
import sys
from types import ModuleType
from typing import Callable, List, Optional, Tuple

PACKAGE = "Blender_JA_SWTOR_Tools"


def import_addon() -> ModuleType:
    """Import the repo as the add-on package regardless of its on-disk directory name."""
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        PACKAGE, os.path.join(repo_root, "__init__.py"),
        submodule_search_locations=[repo_root],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


class TestRunner:
    """Runs each case, logs pass/fail immediately, defers raising until every case has run."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, Optional[Exception]]] = []

    def run(self, name: str, fn: Callable[[], None]) -> None:
        print(f"[test] Running {name}...")
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[test] {name}: FAIL - {e}")
            self.results.append((name, e))
        else:
            print(f"[test] {name}: PASS")
            self.results.append((name, None))

    def report(self) -> None:
        print("[test] === Summary ===")
        failed = []
        for name, err in self.results:
            print(f"[test]   {name}: {'FAIL' if err else 'PASS'}")
            if err is not None:
                failed.append(name)
        if failed:
            raise RuntimeError(f"{len(failed)}/{len(self.results)} test case(s) failed: {', '.join(failed)}")
