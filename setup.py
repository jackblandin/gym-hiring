from setuptools import setup, find_packages

setup(
    name='gym_hiring',
    version='0.0.1',
    description='A Reinforcement Learning OpenAI Gym environment for hiring \
        policies.',
    license='MIT',
    author='Jack Blandin',
    author_email='blandin1@uic.edu',
    url='https://github.com/jackblandin/gym_hiring',
    keywords=['reinforcement-learning', 'machine-learning', 'fairness',
              'research', 'irl', 'python', 'inverse-reinforcement-learning'],
    packages=find_packages(),
    install_requires=['gym', 'numpy==1.15.4'])
