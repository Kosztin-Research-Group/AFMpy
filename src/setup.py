from setuptools import setup, find_packages

setup(
    name = 'AFMpy',
    version = '0.2.0',
    author = 'Creighton M. Lisowski',
    author_email = 'clisowski@missouri.edu',
    packages = find_packages(exclude=['tests*']),
    license = 'GPL-3.0',
    description = 'Python package for Simulated AFM and AFM image analysis',
    extras_require = {
        'gpu': ['cupy-cuda12x'],
    },
    classifiers=[
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
]
)