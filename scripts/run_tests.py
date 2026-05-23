from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


def main() -> int:
	sys.path.insert(0, str(SRC_DIR))

	passed = 0
	failed = 0

	for test_file in sorted(TESTS_DIR.glob("test_*.py")):
		spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
		module = importlib.util.module_from_spec(spec)
		assert spec is not None and spec.loader is not None
		spec.loader.exec_module(module)

		for name in dir(module):
			if not name.startswith("test_"):
				continue

			test = getattr(module, name)
			if not callable(test):
				continue

			try:
				test()
				print(f"PASS {test_file.name}::{name}")
				passed += 1
			except Exception as error:
				print(f"FAIL {test_file.name}::{name}")
				print(f"  {type(error).__name__}: {error}")
				failed += 1

	print(f"\nPassed: {passed}")
	print(f"Failed: {failed}")

	return 0 if failed == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())