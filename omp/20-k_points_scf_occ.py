#!/usr/bin/env python

'''
Mean field with k-points sampling using occ-fft to compute faster

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.
'''
from pyscf.pbc import gto, scf, dft, df
import numpy
from pyscf import lib
lib.logger.TIMER_LEVEL = 0
#from pyscf.pbc.dft import multigrid

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    #mesh = [5,5,5]
    #verbose = 4,
)

nk = [1,1,2]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
kmf = scf.KRHF(cell, kpts)
kmf.with_df.occ = True
kmf.verbose = 4
#kmf.max_cycle = 0
kmf.kernel()
