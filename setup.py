from setuptools import setup

setup(name='ties_analysis',
      version='0.1',
      description='TIES analysis scripts',
      url='https://github.com/adw62/TIES_analysis',
      author='AW',
      author_email='None',
      license='None',
      packages=['ties_analysis', 'ties_analysis.engines', 'ties_analysis.methods'],
      python_requires='>=3.6.0',
      install_requires=['numpy==1.19.5', 'pymbar==3.0.3', 'scikit-learn==0.24.0',
                         'six'],
      entry_points = {'console_scripts':['TIES_ana = ties_analysis.ties_analysis:main']})
