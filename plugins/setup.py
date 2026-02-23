from setuptools import setup, find_packages

setup(
    name="oneat-event-visualizer",
    version="0.1.0",
    description="Napari plugin for ONEAT event visualization and annotation",
    author="KapoorLabs",
    author_email="varunkapoor@kapoorlabs.org",
    license="BSD-3",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "napari[all]>=0.4.15",
        "magicgui>=0.5.0",
        "numpy",
        "pandas",
        "tifffile",
        "qtpy",
    ],
    package_data={
        "": ["napari.yaml", "resources/*"],
    },
    entry_points={
        "napari.manifest": [
            "oneat-event-visualizer = oneat_event_visualizer:napari.yaml",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Framework :: napari",
    ],
)
