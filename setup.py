from setuptools import setup

setup(name='ties_analysis',
      version='0.1',
      description='TIES analysis scripts',
      url='https://github.com/adw62/TIES_analysis',
      author='Alexander David Wade',
      author_email='None',
      license='LGPL',
      packages=['ties_analysis', 'ties_analysis.engines', 'ties_analysis.methods'],
      python_requires='>=3.6.0',
      install_requires=['numpy', 'pymbar', 'scikit-learn',
                         'six', 'nose'],
      entry_points = {'console_scripts':['TIES_ana = ties_analysis.ties_analysis:main']})
