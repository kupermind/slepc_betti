#include <petscmat.h>
#include <stdio.h>

int main(int argc, char ** argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  PetscErrorCode err;
  Mat A;
  err = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(err);
  PetscInt m, n, i, j;
  PetscReal x;
  
  FILE * f = fopen(argv[1], "r");
  fscanf(f, "%ld %ld\n", &m, &n);
  err = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n); CHKERRQ(err);
  err = MatSetUp(A); CHKERRQ(err);

  while (fscanf(f, "%ld %ld %lf", &i, &j, &x) != EOF)
  {
    err = MatSetValue(A, i-1, j-1, x, ADD_VALUES); CHKERRQ(err);
  }
  err = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
  err = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
  
  fclose(f);

  MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  Mat AAT;
  err = MatMatTransposeMult(A, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AAT);
  
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[2], FILE_MODE_WRITE, &viewer);
  MatView(AAT, viewer);

  PetscFinalize();

  return 0;
}
