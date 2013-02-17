   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.6 (r4159) - 21 Sep 2011 10:11
   !
   !  Differentiation of derivativerotmatrixrigid in forward (tangent) mode:
   !   variations   of useful results: rotationmatrix
   !   with respect to varying inputs: timeref
   !   Plus diff mem management of: coscoeffourzrot:in sincoeffourxrot:in
   !                sincoeffouryrot:in sincoeffourzrot:in coefpolxrot:in
   !                coefpolyrot:in coefpolzrot:in coscoeffourxrot:in
   !                coscoeffouryrot:in
   !
   !      ******************************************************************
   !      *                                                                *
   !      * File:          derivativeRotMatrixRigid.f90                    *
   !      * Author:        Edwin van der Weide                             *
   !      * Starting date: 06-01-2004                                      *
   !      * Last modified: 06-12-2005                                      *
   !      *                                                                *
   !      ******************************************************************
   !
   SUBROUTINE DERIVATIVEROTMATRIXRIGID_D(rotationmatrix, rotationmatrixd, &
   &  rotationpoint, t)
   USE FLOWVARREFSTATE
   USE MONITOR
   USE INPUTMOTION
   IMPLICIT NONE
   !
   !      ******************************************************************
   !      *                                                                *
   !      * derivativeRotMatrixRigid determines the derivative of the      *
   !      * rotation matrix at the given time for the rigid body rotation, *
   !      * such that the grid velocities can be determined analytically.  *
   !      * Also the rotation point of the current time level is           *
   !      * determined. This value can change due to translation of the    *
   !      * entire grid.                                                   *
   !      *                                                                *
   !      ******************************************************************
   !
   !
   !      Subroutine arguments.
   !
   REAL(kind=realtype), INTENT(IN) :: t
   REAL(kind=realtype), DIMENSION(3), INTENT(OUT) :: rotationpoint
   REAL(kind=realtype), DIMENSION(3, 3), INTENT(OUT) :: rotationmatrix
   REAL(kind=realtype), DIMENSION(3, 3), INTENT(OUT) :: rotationmatrixd
   !
   !      Local variables.
   !
   INTEGER(kind=inttype) :: i, j
   REAL(kind=realtype) :: phi, dphix, dphiy, dphiz
   REAL(kind=realtype) :: dphixd, dphiyd, dphizd
   REAL(kind=realtype) :: cosx, cosy, cosz, sinx, siny, sinz
   REAL(kind=realtype), DIMENSION(3, 3) :: dm, m
   REAL(kind=realtype), DIMENSION(3, 3) :: dmd, md
   !
   !      Function definitions.
   !
   REAL(kind=realtype) :: RIGIDROTANGLE
   REAL(kind=realtype) :: DERIVATIVERIGIDROTANGLE
   REAL(kind=realtype) :: DERIVATIVERIGIDROTANGLE_D
   INTRINSIC COS
   INTRINSIC SIN
   !
   !      ******************************************************************
   !      *                                                                *
   !      * Begin execution                                                *
   !      *                                                                *
   !      ******************************************************************
   !
   ! Determine the rotation angle around the x-axis for the new
   ! time level and the corresponding values of the sine and cosine.
   phi = RIGIDROTANGLE(degreepolxrot, coefpolxrot, degreefourxrot, &
   &    omegafourxrot, coscoeffourxrot, sincoeffourxrot, t)
   sinx = SIN(phi)
   cosx = COS(phi)
   !print *,'phix',phi
   ! Idem for the y-axis.
   phi = RIGIDROTANGLE(degreepolyrot, coefpolyrot, degreefouryrot, &
   &    omegafouryrot, coscoeffouryrot, sincoeffouryrot, t)
   siny = SIN(phi)
   cosy = COS(phi)
   !print *,'phiY',phi
   ! Idem for the z-axis.
   phi = RIGIDROTANGLE(degreepolzrot, coefpolzrot, degreefourzrot, &
   &    omegafourzrot, coscoeffourzrot, sincoeffourzrot, t)
   sinz = SIN(phi)
   cosz = COS(phi)
   !print *,'phiz',phi
   ! Compute the time derivative of the rotation angles around the
   ! x-axis, y-axis and z-axis.
   dphixd = DERIVATIVERIGIDROTANGLE_D(degreepolxrot, coefpolxrot, &
   &    degreefourxrot, omegafourxrot, coscoeffourxrot, sincoeffourxrot, t, &
   &    dphix)
   !print *,'dphix',dphix
   dphiyd = DERIVATIVERIGIDROTANGLE_D(degreepolyrot, coefpolyrot, &
   &    degreefouryrot, omegafouryrot, coscoeffouryrot, sincoeffouryrot, t, &
   &    dphiy)
   !print *,'dphiy',dphiy
   dphizd = DERIVATIVERIGIDROTANGLE_D(degreepolzrot, coefpolzrot, &
   &    degreefourzrot, omegafourzrot, coscoeffourzrot, sincoeffourzrot, t, &
   &    dphiz)
   dmd = 0.0_8
   !print *,'dphiz',dphiz
   ! Compute the time derivative of the rotation matrix applied to
   ! the coordinates at t == 0.
   ! Part 1. Derivative of the z-rotation matrix multiplied by the
   ! x and y rotation matrix, i.e. dmz * my * mx
   dmd(1, 1) = -(cosy*sinz*dphizd)
   dm(1, 1) = -(cosy*sinz*dphiz)
   dmd(1, 2) = (-(cosx*cosz)-sinx*siny*sinz)*dphizd
   dm(1, 2) = (-(cosx*cosz)-sinx*siny*sinz)*dphiz
   dmd(1, 3) = (sinx*cosz-cosx*siny*sinz)*dphizd
   dm(1, 3) = (sinx*cosz-cosx*siny*sinz)*dphiz
   dmd(2, 1) = cosy*cosz*dphizd
   dm(2, 1) = cosy*cosz*dphiz
   dmd(2, 2) = (sinx*siny*cosz-cosx*sinz)*dphizd
   dm(2, 2) = (sinx*siny*cosz-cosx*sinz)*dphiz
   dmd(2, 3) = (cosx*siny*cosz+sinx*sinz)*dphizd
   dm(2, 3) = (cosx*siny*cosz+sinx*sinz)*dphiz
   dmd(3, 1) = 0.0_8
   dm(3, 1) = zero
   dmd(3, 2) = 0.0_8
   dm(3, 2) = zero
   dmd(3, 3) = 0.0_8
   dm(3, 3) = zero
   ! Part 2: mz * dmy * mx.
   dmd(1, 1) = dmd(1, 1) - siny*cosz*dphiyd
   dm(1, 1) = dm(1, 1) - siny*cosz*dphiy
   dmd(1, 2) = dmd(1, 2) + sinx*cosy*cosz*dphiyd
   dm(1, 2) = dm(1, 2) + sinx*cosy*cosz*dphiy
   dmd(1, 3) = dmd(1, 3) + cosx*cosy*cosz*dphiyd
   dm(1, 3) = dm(1, 3) + cosx*cosy*cosz*dphiy
   dmd(2, 1) = dmd(2, 1) - siny*sinz*dphiyd
   dm(2, 1) = dm(2, 1) - siny*sinz*dphiy
   dmd(2, 2) = dmd(2, 2) + sinx*cosy*sinz*dphiyd
   dm(2, 2) = dm(2, 2) + sinx*cosy*sinz*dphiy
   dmd(2, 3) = dmd(2, 3) + cosx*cosy*sinz*dphiyd
   dm(2, 3) = dm(2, 3) + cosx*cosy*sinz*dphiy
   dmd(3, 1) = dmd(3, 1) - cosy*dphiyd
   dm(3, 1) = dm(3, 1) - cosy*dphiy
   dmd(3, 2) = dmd(3, 2) - sinx*siny*dphiyd
   dm(3, 2) = dm(3, 2) - sinx*siny*dphiy
   dmd(3, 3) = dmd(3, 3) - cosx*siny*dphiyd
   dm(3, 3) = dm(3, 3) - cosx*siny*dphiy
   ! Part 3: mz * my * dmx
   dmd(1, 2) = dmd(1, 2) + (sinx*sinz+cosx*siny*cosz)*dphixd
   dm(1, 2) = dm(1, 2) + (sinx*sinz+cosx*siny*cosz)*dphix
   dmd(1, 3) = dmd(1, 3) + (cosx*sinz-sinx*siny*cosz)*dphixd
   dm(1, 3) = dm(1, 3) + (cosx*sinz-sinx*siny*cosz)*dphix
   dmd(2, 2) = dmd(2, 2) + (cosx*siny*sinz-sinx*cosz)*dphixd
   dm(2, 2) = dm(2, 2) + (cosx*siny*sinz-sinx*cosz)*dphix
   dmd(2, 3) = dmd(2, 3) - (sinx*siny*sinz+cosx*cosz)*dphixd
   dm(2, 3) = dm(2, 3) - (sinx*siny*sinz+cosx*cosz)*dphix
   dmd(3, 2) = dmd(3, 2) + cosx*cosy*dphixd
   dm(3, 2) = dm(3, 2) + cosx*cosy*dphix
   dmd(3, 3) = dmd(3, 3) - sinx*cosy*dphixd
   dm(3, 3) = dm(3, 3) - sinx*cosy*dphix
   ! Determine the rotation matrix at t == t.
   md(1, 1) = 0.0_8
   m(1, 1) = cosy*cosz
   md(2, 1) = 0.0_8
   m(2, 1) = cosy*sinz
   md(3, 1) = 0.0_8
   m(3, 1) = -siny
   md(1, 2) = 0.0_8
   m(1, 2) = sinx*siny*cosz - cosx*sinz
   md(2, 2) = 0.0_8
   m(2, 2) = sinx*siny*sinz + cosx*cosz
   md(3, 2) = 0.0_8
   m(3, 2) = sinx*cosy
   md(1, 3) = 0.0_8
   m(1, 3) = cosx*siny*cosz + sinx*sinz
   md(2, 3) = 0.0_8
   m(2, 3) = cosx*siny*sinz - sinx*cosz
   md(3, 3) = 0.0_8
   m(3, 3) = cosx*cosy
   rotationmatrixd = 0.0_8
   ! Determine the matrix product dm * inverse(m), which will give
   ! the derivative of the rotation matrix when applied to the
   ! current coordinates. Note that inverse(m) == transpose(m).
   DO j=1,3
   DO i=1,3
   rotationmatrixd(i, j) = m(j, 1)*dmd(i, 1) + m(j, 2)*dmd(i, 2) + m(&
   &        j, 3)*dmd(i, 3)
   rotationmatrix(i, j) = dm(i, 1)*m(j, 1) + dm(i, 2)*m(j, 2) + dm(i&
   &        , 3)*m(j, 3)
   END DO
   END DO
   ! Determine the rotation point at the new time level; it is
   ! possible that this value changes due to translation of the grid.
   !  aInf = sqrt(gammaInf*pInf/rhoInf)
   !  RotationPoint(1) = LRef*rotPoint(1) &
   !                    + MachGrid(1)*aInf*t/timeRef
   !  rotationPoint(2) = LRef*rotPoint(2) &
   !                    + MachGrid(2)*aInf*t/timeRef
   !  rotationPoint(3) = LRef*rotPoint(3) &
   !                    + MachGrid(3)*aInf*t/timeRef
   rotationpoint(1) = lref*rotpoint(1)
   rotationpoint(2) = lref*rotpoint(2)
   rotationpoint(3) = lref*rotpoint(3)
   END SUBROUTINE DERIVATIVEROTMATRIXRIGID_D
