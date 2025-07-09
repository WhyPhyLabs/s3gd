from setuptools import setup, find_packages

setup(
  name = 'smoothed-sign-sgd-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.3',
  license='MIT',
  description = 'Smoothed SignSGD Optimizer - Pytorch',
  author = 'Aria Bagheri',
  author_email = 'aria@whyphy.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/WhyPhyLabs/smoothed-sign-sgd',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=2.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
