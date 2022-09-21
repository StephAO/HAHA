#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding='UTF8') as fh:
    long_description = fh.read()

setup(name='oai_agents',
      version='0.1.0',
      description='Cooperative multi-agent environment based on Overcooked',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Stéphane Aroca-Ouellette',
      author_email='stephane.aroca-ouellette@colorado.edu',
      url='https://github.com/StephAO/oai_agents',
      download_url='https://github.com/StephAO/oai_agents',
      packages=find_packages('oai_agents'),
      keywords=['Overcooked', 'AI', 'Reinforcement Learning', 'Human Agent Collaboration'],
      package_dir={"": "oai_agents"},
      package_data={
        'oai_agents' : [
          'data/*.pickle'
        ],
      },
      install_requires=[
        'numpy',
        'stable_baselines3',
        'sb3_contrib',
        'tqdm',
        'wandb',
        'gym',
        'pygame',
      ]
    )