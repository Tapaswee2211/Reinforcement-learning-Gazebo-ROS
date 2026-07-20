from setuptools import setup

package_name = 'gazebo_rl_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tapaswee',
    maintainer_email='dasaritapaswee2018@gmail.com',
    description='Reinforcement Learning environment interface for Gazebo using ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
#            'env_node = gazebo_rl_env.env_node:main',
            'train_env = gazebo_rl_env.train_env:main',
        ],
    },
)

