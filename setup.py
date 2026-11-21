"""Build script for the mppi_nav ament_python package."""
from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'mppi_nav'


def share(subdir: str, pattern: str) -> tuple:
    """Install every file in <subdir> matching <pattern> to share/mppi_nav/<subdir>."""
    return (os.path.join('share', package_name, subdir), glob(os.path.join(subdir, pattern)))


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'sim2d', 'sim2d.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        share('launch', '*.launch.py'),
        share('config', '*.yaml'),
        share('config', '*.rviz'),
        share('worlds', '*.sdf'),
        share('paths', '*.csv'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ataraxy',
    maintainer_email='ameyabhagwat2005@gmail.com',
    description='Path recorder + MPPI path tracker with obstacle avoidance (TurtleBot3, Python).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_recorder = mppi_nav.path_recorder:main',
            'mppi_tracker = mppi_nav.mppi_tracker:main',
            'obstacle_spawner = mppi_nav.obstacle_spawner:main',
        ],
    },
)
