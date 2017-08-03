from setuptools import setup, find_packages


setup(name='schema_games',
      version='1.0.0',
      author='Vicarious FPC Inc',
      author_email='info@vicarious.com',
      description="Breakout Environments from Schema Network ICML '17 Paper",
      url='https://github.com/vicariousinc/schema_games',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('schema_games')],
      install_requires=['gym>=0.9.1[all]',
                        'pygame',
                        'matplotlib',
                        'future'],
      )
