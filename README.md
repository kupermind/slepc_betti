# slepc_betti

All the routines are parallel except when explicitly stated that they are not.

convert.cc
Converting the .sms format into .petsc one. Command line example:
srun convert file.sms file.petsc
The routine converts file.sms file into file.petsc petsc file format ready for the parallel use with PETSc/SLEPc routines.

laplacian.c
Calculation of betti numbers for Laplacian matrix. The matrix must be square. Command line example:
srun laplacian -file laplace_l4_lvl2.petsc -eps_nev 12000 -bv_type vecs
where laplace_l4_lvl2.petsc is the matrix file, -eps_nev is the approximate number of zero eigenvalues,
-bv_type is the parameter for the memory not to go off limits

nullspace.c
Calculation of the matrix nullity.

svd.c
Calculation of svd-s. Command line example:
srun svd -file mc2_lvl4.petsc -svd_nsv 2000 -svd_max_it 7000 -svd_smallest -svd_type trlanczos -svd_error_absolute ::ascii_info_detail
This example would calculate 2000 singilar values with a maximum 7000 iterations using the trlanczos as its core method and printing the absolute error result.
All the parameters are standard and described on the slepsc documentation page related to SVD.

svd_lobpcg.c
Calculation of nullity using the LOBPCG core method. The matrix can be rectangular. Command line example:
srun svd_lobpcg -file mc2_lvl4.petsc -svd_nsv 1010
where mc2_lvl4.petsc is the matrix file, -svd_nsv is the approximate number of zero singular values.
