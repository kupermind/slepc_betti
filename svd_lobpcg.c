
static char help[] = "Nullspace problem with the SVD.\n\n"
  "Must run indicating an estimate of the nullspace dimension:\n"
  "  $ ./svd -file l4_lvl3.petsc -svd_nsv 639\n\n";

#include <slepcsvd.h>
#include <slepc/private/epsimpl.h>  /* private EPS header */

PetscErrorCode MyStoppingTest(EPS eps,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nsv,EPSConvergedReason *reason,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* check if usual termination conditions are met */
  ierr = EPSStoppingBasic(eps,its,max_it,nconv,nsv,reason,NULL);CHKERRQ(ierr);
  if (*reason==EPS_CONVERGED_ITERATING) {
    /* this is a horrible hack, that requires including the private EPS header;
       also, the check errest[.]>0.0 is required when using LOBPCG */
    if (nconv>1 && eps->errest[nconv]>0.0 && eps->errest[nconv]<1e-5 && PetscAbsScalar(eps->eigr[nconv])>0.1) *reason = EPS_CONVERGED_USER;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  SVD            svd;             /* singular value problem solver context */
  SVDType        type;
  EPS            eps;
  ST             st;
  KSP            ksp;
  PC             pc;
  PetscReal      tol,nrm;
  Vec            v,w;
  PetscInt       i,nsv,nconv,maxit,its,m,n;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the operator matrix that defines the singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nNullspace problem via the SVD.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-file",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file option");

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatGetSize(A,&n,&m);
  ierr = PetscPrintf(PETSC_COMM_WORLD," The matrix %ld x %ld is loaded\n", n, m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);
  ierr = SVDSetWhichSingularTriplets(svd,SVD_SMALLEST);CHKERRQ(ierr);
  ierr = SVDSetTolerances(svd,1e-15,50000);CHKERRQ(ierr);
  ierr = SVDSetConvergenceTest(svd,SVD_CONV_ABS);CHKERRQ(ierr);
  ierr = SVDSetType(svd,SVDCROSS);CHKERRQ(ierr);
  ierr = SVDCrossSetExplicitMatrix(svd,PETSC_TRUE);CHKERRQ(ierr);
  ierr = SVDCrossGetEPS(svd,&eps);CHKERRQ(ierr);
  ierr = EPSSetType(eps,EPSLOBPCG);CHKERRQ(ierr);
  ierr = EPSLOBPCGSetBlockSize(eps,1);CHKERRQ(ierr);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCSOR);CHKERRQ(ierr);
  ierr = EPSSetStoppingTestFunction(eps,MyStoppingTest,NULL,NULL);CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDGetIterationNumber(svd,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
  ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
  ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check computed vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateVecs(A,&w,&v);CHKERRQ(ierr);
  ierr = SVDGetConverged(svd,&nconv);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"The computed nullity is %d\n",nconv);CHKERRQ(ierr);
/*
  for (i=0;i<nconv;i++) {
    ierr = SVDGetSingularTriplet(svd,i,NULL,v,NULL);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,v,w);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"norm of A'*x_%d = %g\n",i,nrm);CHKERRQ(ierr);
  }
*/
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);

  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
