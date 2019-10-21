#!/usr/bin/env python

'''
Mean field with k-points sampling

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.
'''
import time
from pyscf.pbc import gto, scf, dft, df
import numpy

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
Jtime=time.time()
nk = [1,1,1]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
kmf = scf.KRHF(cell, kpts)
kmf.verbose = 4
kmf.kernel()
print "Took this long for total: ", time.time()-Jtime

mf = scf.RHF(cell)
ehf = mf.kernel()
print("HF energy (per unit cell) = %.17g" % ehf)



exit()

kmf = dft.KRKS(cell, kpts)
# Turn to the atomic grids if you like
kmf.grids = dft.gen_grid.BeckeGrids(cell)
kmf.xc = 'm06,m06'
kmf.kernel()


#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation
#
mf = scf.KRHF(cell, kpts).newton()
mf.kernel()

