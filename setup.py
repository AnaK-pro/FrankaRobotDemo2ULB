from setuptools import setup, find_packages

setup(
    name='demo2_robot_simulation',
    version='0.1.0',
    author='Ana',
    description='Simulation robotique Franka Panda avec IsaacSim',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.23.0',
        'torch>=2.0.0',
        'gymnasium>=0.29.0',
        'pyyaml>=6.0.0',
        'python-dotenv>=1.0.0',
        
    ],
)