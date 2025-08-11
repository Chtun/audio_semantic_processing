from setuptools import setup, find_packages

setup(
    name='audio_semantic_processing',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'timm==0.4.12',
        'torchaudio',
        'wandb',
        'scipy',
        'scikit-learn',
    ],
    author='Chitsein Htun',
    author_email='chtun@live.com',
    description='Re-Implementation of EquiAV, finetuned on specific subset of AudioSet.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Chtun/audio_semantic_processing',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)