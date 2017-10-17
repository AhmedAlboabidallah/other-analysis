from distutils.core import setup
from Cython.Build import cythonize
setup(
  name='test3 app',
  ext_modles=cythonize("test3.pyx"))