from setuptools import setup, find_namespace_packages

setup(
    name="pia_ptsd_prediction",
    version="0.0.1",
    url="https://github.com/JanCBrammer/PoliceInAction_PTSD_prediction",
    author="Jan C. Brammer",
    author_email="jan.c.brammer@gmail.com",
    packages=find_namespace_packages(exclude=["data", "literature", "preregistration"]),
    python_requires=">=3.7",
    license="GPL-3.0",
)
