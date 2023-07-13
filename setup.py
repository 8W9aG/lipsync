from setuptools import setup, find_packages
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lipsync',
    version='0.0.1',
    description='Lip Synchronization (Wav2Lip).',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/mowshon/flarepy',
    keywords='lipsync, lip, wav2lip, lip synchronization',
    author='Rudrabha Mukhopadhyay, Mowshon',
    author_email='mowshon@yandex.ru',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'librosa',
        'numpy',
        'opencv-contrib-python',
        'opencv-python',
        'tensorflow',
        'torch==1.1.0',
        'torchvision==0.3.0',
        'tqdm==4.45.0',
        'numba==0.48',
        'moviepy'
    ],
    zip_safe=False
)
