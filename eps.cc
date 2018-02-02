#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <slepceps.h>
#include <petscmat.h>

int main(int argc, char ** argv)
{
  SlepcInitialize(&argc, &argv, NULL, NULL);

  std::string infile(argv[1]);
  std::string outfile(argv[2]);

  PetscMPIInt mpi_rank;
  PetscMPIInt mpi_size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  
  PetscErrorCode err;
  Mat L;
  err = MatCreate(PETSC_COMM_WORLD, &L); CHKERRQ(err);
  err = MatSetFromOptions(L); CHKERRQ(err);

  PetscViewer viewer;
  err = PetscViewerBinaryOpen(PETSC_COMM_WORLD, infile.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(err);
  err = MatLoad(L, viewer); CHKERRQ(err);
  err = MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
  err = MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY); CHKERRQ(err);

  PetscInt m;
  PetscInt n;
  err = MatGetSize(L, &m, &n); CHKERRQ(err);
  assert(m == n);
  PetscPrintf(PETSC_COMM_WORLD, "Laplacian is size %d.\n", m);
  
  EPS eps;
  err = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(err);
  err = EPSSetOperators(eps, L, NULL); CHKERRQ(err);
  err = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(err);
  err = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE); CHKERRQ(err);
  err = EPSSetFromOptions(eps); CHKERRQ(err);

  PetscPrintf(PETSC_COMM_WORLD, "Solving...\n");
  err = EPSSolve(eps); CHKERRQ(err);

  PetscInt num_its;
  err = EPSGetIterationNumber(eps, &num_its); CHKERRQ(err);
  PetscPrintf(PETSC_COMM_WORLD, "Took %d iterations.\n", num_its);
  
  
  PetscInt num_ev;
  err = EPSGetConverged(eps, &num_ev); CHKERRQ(err);
  PetscPrintf(PETSC_COMM_WORLD, "Have %d eigenpairs.\n", num_ev);

  Vec vec;
  err = MatCreateVecs(L, &vec, NULL); CHKERRQ(err);
  PetscReal val;
  PetscReal cutoff = 1e-6;
  PetscInt estimated_betti = 0;
  PetscInt vec_size;
  err = VecGetSize(vec, &vec_size); CHKERRQ(err);
  
  Vec vec_r0;
  if (mpi_rank == 0)
  {
    err = VecCreateSeq(PETSC_COMM_SELF, vec_size, &vec_r0); CHKERRQ(err);
  }

  VecScatter scatter;
  err = VecScatterCreateToZero(vec, &scatter, &vec_r0); CHKERRQ(err);

  std::ofstream ofs;
  if (mpi_rank == 0)
  {
    ofs.open(outfile.c_str(), std::ios::out);
  }
    
  for (PetscInt i = 0; i < num_ev; i++)
  {
    err = EPSGetEigenpair(eps, i, &val, NULL, vec, NULL); CHKERRQ(err);
    PetscPrintf(PETSC_COMM_WORLD, "%d: %.20f\n", i, val);

    if (std::abs(val) < cutoff)
    {
      estimated_betti++;

      err = VecScatterBegin(scatter, vec, vec_r0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(err);
      err = VecScatterEnd(scatter, vec, vec_r0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(err);

      if (mpi_rank == 0)
      {
        PetscReal * tmp;
        err = VecGetArray(vec_r0, &tmp); CHKERRQ(err);
        for (PetscInt j = 0; j < vec_size; j++)
        {
          ofs << tmp[j] << " ";
        }
        ofs << std::endl;
        err = VecRestoreArray(vec_r0, &tmp); CHKERRQ(err); 
      }
    }
  }
 

  PetscPrintf(PETSC_COMM_WORLD, "Estimated Betti number: %d.\n", estimated_betti);

  if (mpi_rank == 0)
  {
    ofs.close();
  }

  err = VecScatterDestroy(&scatter); CHKERRQ(err);
  err = VecDestroy(&vec); CHKERRQ(err);
  err = VecDestroy(&vec_r0); CHKERRQ(err);
  
  err = EPSDestroy(&eps); CHKERRQ(err);
  err = MatDestroy(&L); CHKERRQ(err);
  SlepcFinalize();
  return 0;
}
