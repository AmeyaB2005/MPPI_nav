"""Where recorded paths and results are read and written (no hard-coded clone location).

With `colcon build --symlink-install` the installed package files are links back to the
cloned source folder, so we follow one of them to find the clone, wherever it is:

    <clone>/paths/     recorded and test paths (CSV)
    <clone>/results/   driven-path CSVs written at the goal

If the package was built without --symlink-install (installed files are plain copies), the
source folder cannot be found and ~/.ros/mppi_nav/{paths,results} is used instead.
"""
import os

from ament_index_python.packages import get_package_share_directory

FALLBACK_DIR = os.path.join('~', '.ros', 'mppi_nav')


def package_source_dir():
    """The cloned mppi_nav source folder, or None if the install is not symlinked."""
    share = get_package_share_directory('mppi_nav')
    source = os.path.dirname(os.path.realpath(os.path.join(share, 'package.xml')))
    return source if source != os.path.realpath(share) else None


def data_dir(sub: str) -> str:
    """<clone>/<sub> (e.g. 'paths', 'results'), or the ~/.ros/mppi_nav/<sub> fallback."""
    source = package_source_dir()
    return os.path.join(source, sub) if source else os.path.join(os.path.expanduser(FALLBACK_DIR), sub)
