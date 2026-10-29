"""Use the installed pinned runner with screenshot capture from the first frame.

Preserves its checksum guard, empty temporary cwd, credential-free environment,
no-download policy, owned process cleanup and exclusive output publication.
Screenshot capture is explicit. The production entry adds a capture margin, then
finish_export.py removes it to correct the observed lower-caption mismatch.
The actual encoded frames are checked.
No installed runtime file is modified.
"""
from pathlib import Path
import importlib.util
import sys

runner = Path('/Users/nazeeh/.local/share/codex-tools/hyperframes-0.8.33-local/local.py')
spec = importlib.util.spec_from_file_location('causalos_pinned_hyperframes',runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
base_environment = module.environment

def screenshot_environment():
    env = base_environment()
    env['PRODUCER_FORCE_SCREENSHOT'] = 'true'
    env['PRODUCER_EXPERIMENTAL_FAST_CAPTURE'] = 'false'
    return env

module.environment = screenshot_environment
if __name__ == '__main__':
    sys.exit(module.main())
