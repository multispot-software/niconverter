from setuptools import setup

__version__ = '0.1'

setup(
    name = 'niconverter',
    version=__version__,
    author = 'Antonino Ingargiola',
    author_email = 'tritemio@gmail.com',
    url = 'https://github.com/tritemio/niconverter',
    download_url = 'https://github.com/tritemio/niconverter',
    install_requires = ['phconvert'],
    license = 'MIT',
    description = ("Convert 48-spot smFRET data from NI-FGPA to Photon-HDF5."),
    platforms = ('Windows', 'Linux', 'Mac OS X'),
    classifiers=['Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering',
                 ],
    py_modules = ['niconverter'],
    keywords = ('single-molecule FRET smFRET biophysics file-format HDF5 '
                'Photon-HDF5')
)
