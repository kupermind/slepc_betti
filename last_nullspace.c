/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a singular value problem with the matrix loaded from a file.\n"
  "This example works for both real and complex numbers.\n\n"
  "The command line options are:\n"
  "  -file <filename>, where <filename> = matrix file in PETSc binary form.\n\n";

#include <petscksp.h>
#include <slepcbv.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B,F;
  KSP            ksp;
  PC             pc;
  BV             V;
  Vec            x,b,v,w;
  PetscInt       i,m,n,dim;
  PetscReal      nrm;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,ckerr=PETSC_FALSE;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

#if !defined(PETSC_HAVE_MUMPS)
#error "Need to configure PETSc with --download-mumps --download-parmetis --download-scalapack --download-metis"
#else

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nNullspace problem.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-file",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file option");

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading COMPLEX matrix from a binary file...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading REAL matrix from a binary file...\n");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatGetSize(A,&n,&m);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The matrix %ld x %ld is loaded\n\n",n,m);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-ckerr",&ckerr,NULL);CHKERRQ(ierr);

  /* explicitly compute the cross-product matrix B = A*A' */
  ierr = MatMatTransposeMult(A,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);

  /* create work vectors */
  ierr = MatCreateVecs(B,&x,&b);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&w,NULL);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,B,B);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  ierr = PetscOptionsInsertString(NULL,"-mat_mumps_icntl_24 1");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* get dimension of null space */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
  ierr = MatMumpsGetInfog(F,28,&dim);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Dimension of the null space = %d\n",dim);CHKERRQ(ierr);

  /* V will hold the basis of the null space */
  ierr = BVCreate(PETSC_COMM_WORLD,&V);CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(V,x,dim);CHKERRQ(ierr);
  ierr = BVSetFromOptions(V);CHKERRQ(ierr);

  for (i=1;i<=dim;i++) {
    ierr = MatMumpsSetIcntl(F,25,i);CHKERRQ(ierr);
    ierr = BVGetColumn(V,i-1,&v);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,v);CHKERRQ(ierr);
    ierr = VecNormalize(v,NULL);CHKERRQ(ierr);
    if (ckerr) {
      ierr = MatMultTranspose(A,v,w);CHKERRQ(ierr);
      ierr = VecNorm(w,NORM_2,&nrm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"norm of A'*x_%d = %g\n",i,nrm);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(V,i-1,&v);CHKERRQ(ierr);
  }

  /* orthogonalize basis - not necessary since it seems that it is already orthogonal */
/*
  Mat R;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,dim,dim,NULL,&R);CHKERRQ(ierr);
  ierr = BVOrthogonalize(V,R);CHKERRQ(ierr);
  ierr = MatView(R,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
*/

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = BVDestroy(&V);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
#endif
  ierr = SlepcFinalize();
  return ierr;
}
