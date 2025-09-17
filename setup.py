from setuptools import setup, find_packages

setup(
    name="qalice-core",
    version="0.2.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "qalice-exec=qalice_core.execution_opt.cli:app",
        ],
    },
)