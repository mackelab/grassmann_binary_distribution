from setuptools import setup

setup(name='GrassmannBinaryDistribution', # project name
      version='0.1',
      description='Implementation of binary distributions in the Grassmann formalism, including conditional distributions and estimating methods.',
      url='https://github.com/mackelab/grassmann_binary_distribution.git',
      author='Cornelius Schroeder',
      author_email='cornelius.schroeder@uni-tuebingen.de',
      license='MIT',
      packages=['grassmann_distribution'], # actual package name (to import package)
      zip_safe=False)
