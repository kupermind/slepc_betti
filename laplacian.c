
static char help[] = "Nullspace problem with the EPS.\n\n"
  "Must run indicating an estimate of the nullspace dimension:\n"
  "  $ ./laplacian -file l4_lvl3_laplacian.petsc -eps_nev 639\n\n";

#include <slepceps.h>
#include <slepc/private/epsimpl.h>  /* private EPS header */

PetscErrorCode MyStoppingTest(EPS eps,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,EPSConvergedReason *reason,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* check if usual termination conditions are met */
  ierr = EPSStoppingBasic(eps,its,max_it,nconv,nev,reason,NULL);CHKERRQ(ierr);
  if (*reason==EPS_CONVERGED_ITERATING) {
    /* this is a horrible hack, that requires including the private EPS header;
       also, the check errest[.]>0.0 is required when using LOBPCG */
    if (nconv>1 && eps->errest[nconv]>0.0 && eps->errest[nconv]<1e-5 && PetscAbsScalar(eps->eigr[nconv])>0.1) *reason = EPS_CONVERGED_USER;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A;               /* Laplacian matrix */
  EPS            eps;             /* eigenvalue problem solver context */
  EPSType        type;
  ST             st;
  KSP            ksp;
  PC             pc;
  PetscReal      tol,nrm;
  Vec            v,w;
  PetscInt       i,nev,nconv,maxit,its,m,n;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the operator matrix that defines the eigenvalue problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nNullspace problem via eigendecomposition of the Laplacian.\n\n");CHKERRQ(ierr);
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

  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps,1e-15,50000);CHKERRQ(ierr);
  ierr = EPSSetConvergenceTest(eps,EPS_CONV_ABS);CHKERRQ(ierr);
  ierr = EPSSetType(eps,EPSLOBPCG);CHKERRQ(ierr);
  ierr = EPSLOBPCGSetBlockSize(eps,1);CHKERRQ(ierr);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCSOR);CHKERRQ(ierr);
  ierr = EPSSetStoppingTestFunction(eps,MyStoppingTest,NULL,NULL);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %D\n",nev);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check computed vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateVecs(A,&w,&v);CHKERRQ(ierr);
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"The Betti number is %d\n",nconv);CHKERRQ(ierr);
/*
  for (i=0;i<nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,NULL,NULL,v,NULL);CHKERRQ(ierr);
    ierr = MatMult(A,v,w);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"norm of A*x_%d = %g\n",i,nrm);CHKERRQ(ierr);
  }
*/
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);

  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
