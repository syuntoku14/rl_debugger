from setuptools import setup, Extension, find_packages

install_requires = [
    "numpy",
    "scipy",
    "pillow"
]

setup(
    name="rl_debugger",
    description="Debugging tools for Reinforcement Learning",
    install_requires=install_requires,
    version="v0.0.1")