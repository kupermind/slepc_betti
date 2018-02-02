#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
LOCDIR     = src/svd/examples/tutorials/
EXAMPLESC  = ex8.c ex14.c ex15.c svd.c convert.cc convert_sym.cc convert_att.cc eps.cc convert_serial.cc nullspace.c svd_lobpcg.c laplacian.c
EXAMPLESF  = ex15f.F
MANSEC     = SVD

TESTEXAMPLES_C           = ex8.PETSc runex8_1 ex8.rm
TESTEXAMPLES_C_DATAFILE  = ex14.PETSc runex14_1 runex14_2 ex14.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o ${SLEPC_SVD_LIB}
	${RM} ex8.o

convert: convert.o chkopts
	-${CLINKER} -o convert convert.o ${SLEPC_SVD_LIB}
	${RM} convert.o

convert_sym: convert_sym.o chkopts
	-${CLINKER} -o convert_sym convert_sym.o ${SLEPC_SVD_LIB}
	${RM} convert_sym.o

convert_serial: convert_serial.o chkopts
	-${CLINKER} -o convert_serial convert_serial.o ${SLEPC_SVD_LIB}
	${RM} convert_serial.o

convert_att: convert_att.o chkopts
	-${CLINKER} -o convert_att convert_att.o ${SLEPC_SVD_LIB}
	${RM} convert_att.o

eps: eps.o chkopts
	-${CLINKER} -o eps eps.o ${SLEPC_SVD_LIB}
	${RM} eps.o

laplacian: laplacian.o chkopts
	-${CLINKER} -o laplacian laplacian.o ${SLEPC_SVD_LIB}
	${RM} laplacian.o

svd: svd.o chkopts
	-${CLINKER} -o svd svd.o ${SLEPC_SVD_LIB}
	${RM} svd.o

nullspace: nullspace.o chkopts
	-${CLINKER} -o nullspace nullspace.o ${SLEPC_SVD_LIB}
	${RM} nullspace.o

svd_lobpcg: svd_lobpcg.o chkopts
	-${CLINKER} -o svd_lobpcg svd_lobpcg.o ${SLEPC_SVD_LIB}
	${RM} svd_lobpcg.o

ex14: ex14.o chkopts
	-${CLINKER} -o ex14 ex14.o ${SLEPC_SVD_LIB}
	${RM} ex14.o

ex15: ex15.o chkopts
	-${CLINKER} -o ex15 ex15.o ${SLEPC_SVD_LIB}
	${RM} ex15.o

ex15f: ex15f.o chkopts
	-${FLINKER} -o ex15f ex15f.o ${SLEPC_SVD_LIB}
	${RM} ex15f.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runex8_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex8 > $${test}.tmp 2>&1; \
	${TESTCODE}

runex14_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex14 -file ${DATAPATH}/rdb200.petsc -svd_nsv 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex14_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex14 -file ${DATAPATH}/rdb200.petsc -svd_nsv 2 -svd_type cyclic -svd_cyclic_explicitmatrix -svd_cyclic_st_type sinvert -svd_cyclic_eps_target 12.0 -svd_cyclic_st_ksp_type preonly -svd_cyclic_st_pc_type lu -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex15_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex15 > $${test}.tmp 2>&1; \
	${TESTCODE}

runex15f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex15f > $${test}.tmp 2>&1; \
	${TESTCODE}

