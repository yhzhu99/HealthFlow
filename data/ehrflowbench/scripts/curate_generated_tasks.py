from importlib import import_module as _import_module
import sys as _sys


_impl = _import_module("data.ehrflowbench.scripts.prepare_tasks.curate_generated_tasks")
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
