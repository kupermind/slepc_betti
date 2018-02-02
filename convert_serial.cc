#include <petscmat.h>
#include <stdio.h>
#include <fstream>
#include <vector>

int main(int argc, char ** argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  PetscErrorCode err;
  Mat A;
  err = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(err);
  PetscInt m, n, i, j, lasti;
  PetscReal x;

  err = PetscPrintf(PETSC_COMM_WORLD,"Getting ready to read\n");CHKERRQ(err);
  
  std::ifstream f(argv[1]);
  f >> n >> m;

  std::vector<PetscInt> iarr(n + 1, 0.);
  std::vector<PetscInt> jarr;
  std::vector<PetscScalar> varr;

  err = PetscPrintf(PETSC_COMM_WORLD,"Matrix size is %ld x %ld\n", n, m);CHKERRQ(err);

  lasti = 0;
  while (!f.eof())
  {
      f >> i >> j >> x;

      if(i == 0)
          break;

      if(lasti != i)
      {
          lasti = i;
          iarr[i] = iarr[i - 1];
      }

      iarr[i] = iarr[i] + 1; 
      jarr.push_back(j - 1);
      varr.push_back(x);
  }
  f.close();

  MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, n, m, &iarr[0], &jarr[0], &varr[0], &A);
  err = PetscPrintf(PETSC_COMM_WORLD, "Created matrix\n");CHKERRQ(err);

//  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  err = PetscPrintf(PETSC_COMM_WORLD, "Writing binary matrix\n");CHKERRQ(err);
  
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[2], FILE_MODE_WRITE, &viewer);
  MatView(A, viewer);

  err = PetscPrintf(PETSC_COMM_WORLD, "Binary matrix has written\n");CHKERRQ(err);

  PetscFinalize();

  return 0;
}
