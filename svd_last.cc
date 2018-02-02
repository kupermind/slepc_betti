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

#include <slepcsvd.h>
#include <vector>
#include <algorithm>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  SVD            svd;             /* singular value problem solver context */
  SVDType        type;
  PetscReal      tol;
  PetscInt       nsv,maxit,its;
  char           filename[PETSC_MAX_PATH_LEN];
//  PetscViewer    viewer;
  PetscBool      flg,terse;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the operator matrix that defines the singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSingular value problem stored in file.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-file",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file option");

  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading matrix from a file...\n");CHKERRQ(ierr);
//  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  int rank, size;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  /// Read and create the matrix based on arrays
  PetscInt nrows, ncols;
  char c;

  FILE *fp = fopen(filename, "r");
  fscanf(fp, "%ld %ld %c", &nrows, &ncols, &c);

  std::vector<PetscInt> icumarr(nrows, 0);
  std::vector<PetscInt> iarr, jarr;
  std::vector<PetscScalar> varr;

  PetscInt i, lasti, j;
  PetscScalar val;
  fscanf(fp, "%ld %ld %lf\n", &i, &j, &val);
  lasti = i - 1;
  while(val != 0)
  {
      --i; --j;
      if(lasti != i)
      {
          lasti = i;
          icumarr[i] = icumarr[i-1];
      }
      icumarr[i] = icumarr[i] + 1;

      iarr.push_back(i); 
      jarr.push_back(j);
      varr.push_back(val);

//printf("i=%ld, j=%ld, iarr[i]=%ld, lasti=%ld\n", i, j, iarr[i], lasti);

      fscanf(fp, "%ld %ld %lf\n", &i, &j, &val);
  }
  fclose(fp);

  /// Need to sort column indices
  for (i = 1; i < nrows; ++i)
      std::sort(jarr.begin() + icumarr[i-1], jarr.begin() + icumarr[i]);
/*
  for (i = 1; i < nrows; ++i)
  {
      PetscScalar one = -1;   
      for (j = icumarr[i-1]; j < icumarr[i]; ++j)
      {
          one *= -1;
          jarr[j] = one;
      }
  }
*/
  if (rank == 0)
      printf("The matrix size is: %ld x %ld\n", nrows, ncols);

  /// Distributing the matrix
  Vec x;
  PetscInt rstart, rend;
  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, PETSC_DECIDE, nrows);
  VecSetFromOptions(x);
  VecGetOwnershipRange(x, &rstart, &rend);

//  printf("\n  rstart %ld, rend %ld, icumlocarr size %ld\n", rstart, rend);
 
  /// Create local array of row offsets based on the global array of row offsets
  std::vector<PetscInt> icumlocarr;
  icumlocarr.push_back(0);
  if (rstart == 0)
      lasti = icumarr[rstart];
  else
      lasti = icumarr[rstart] - icumarr[rstart-1];
  icumlocarr.push_back(lasti);
  for (i = rstart+1; i < rend; ++i)
  {
      lasti = lasti + icumarr[i] - icumarr[i-1];
      icumlocarr.push_back(lasti);
  }

/*
  for (int irank = 0; irank < size; ++irank)
  {
      if (irank == rank)
          for (i = 0; i < icumlocarr.size(); ++i)
              printf("[%ld]: %ld, ", rank, icumlocarr[i]);
      printf("\n\n");
  }
  MPI_Barrier(PETSC_COMM_WORLD);
//      printf("%ld, ", icumarr[i]);
//printf("\nLast i %ld, last j %ld, size of jarr %ld, size of varr %ld\n", i, j, jarr.size(), varr.size());
*/

  /// Offset
  if (rstart == 0)
      lasti = 0;
  else
      lasti = icumarr[rstart-1];
  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, rend - rstart, PETSC_DECIDE, nrows, ncols, &icumlocarr[0], &jarr[0] + lasti, &varr[0] + lasti, &A);
//  MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, nrows, ncols, &iarr[0], &jarr[0], &varr[0], &A);

  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);

  /*
     Set operator
  */
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDGetIterationNumber(svd,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
  ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = SVDReasonView(svd,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
