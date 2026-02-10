"""Quick test runner that executes a subset of tests without using pytest's collection.
This avoids importing project-level __init__.py which references missing modules.

It imports our test modules directly and calls their test functions.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    'tests.test_io_and_maps',
    'tests.test_reprojection_and_utils',
    'tests.test_single_map_process',
    'tests.test_synthetic_raster_process',
]

results = []
for mod_name in MODULES:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f"IMPORT FAIL: {mod_name}: {e}")
        results.append((mod_name, False, str(e)))
        continue
    # execute callables starting with test_
    for name in dir(mod):
        if name.startswith('test_'):
            fn = getattr(mod, name)
            try:
                print(f"RUN {mod_name}.{name}() ... ", end='')
                fn()
                print("OK")
                results.append((f"{mod_name}.{name}", True, ''))
            except Exception as e:
                print(f"FAIL: {e}")
                results.append((f"{mod_name}.{name}", False, str(e)))

print('\nSummary:')
for r in results:
    print(r)

# Exit with non-zero if any failed
if not all(r[1] for r in results):
    raise SystemExit(1)
