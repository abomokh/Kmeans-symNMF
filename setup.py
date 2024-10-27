from setuptools import setup, Extension

module = Extension("mysymnmfsp", sources=['symnmfmodule.c', 'symnmf.c'])
setup(name='mysymnmfsp',
     version='1.0',
     description='Python wrapper for custom C extension',
     ext_modules=[module]
)
