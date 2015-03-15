from distutils.core import setup

setup(
    name='FISTASpecializer',
    version='0.95a',

    packages=[
        'FISTASpecializer',
    ],

    package_data={
        'FISTASpecializer': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
    ]
)

