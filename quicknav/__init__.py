import os

def _get_version():
    version_file = os.path.join(os.path.dirname(__file__), '..', 'VERSION.txt')
    try:
        with open(version_file, 'r') as vf:
            return vf.read().strip()
    except Exception:
        return "unknown"

__version__ = _get_version()