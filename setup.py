from setuptools import setup
import versioneer


setup(
    name = 'niconverter',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author = 'Antonino Ingargiola',
    author_email = 'tritemio@gmail.com',
    url = 'https://github.com/tritemio/niconverter',
    download_url = 'https://github.com/tritemio/niconverter',
    install_requires = ['phconvert', 'tqdm', 'numba'],
    license = 'MIT',
    description = ("Convert 48-spot smFRET data from NI-FGPA to Photon-HDF5."),
    platforms = ('Windows', 'Linux', 'Mac OS X'),
    classifiers=['Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering',
                 ],
    packages = ['niconverter'],
    keywords = ('single-molecule FRET smFRET biophysics file-format HDF5 '
                'Photon-HDF5')
)
