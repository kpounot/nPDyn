/* absco.c:
 *   Absorption corrections for X-ray and neutron diffraction.
 *
 * Copyright:
 *   (C) 2011-12 Joachim Wuttke
 *
 * Licence:
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published
 *   by the Free Software Foundation; either version 3 of the License, or
 *   (at your option) any later version. Alternative licenses can be
 *   obtained through written agreement from the author.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but without any warranty; without even the implied warranty of
 *   merchantability or fitness for a particular purpose.
 *   See the GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author:
 *   Joachim Wuttke
 *   Forschungszentrum Jülich, Germany
 *   j.wuttke@fz-juelich.de
 *
 * API documentation:
 *   man 7 absco
 *
 * Website:
 *   http://apps.jcns.fz-juelich.de/absco
 *
 * Reference:
 *   Wuttke, libabsco white paper (see website for link)
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <errno.h>

#include <gsl/gsl_integration.h>

#include "absco.h"

#define SQR(x) ((x)*(x))

#define EPS_ANGLE 0.001
static double epsilon = 1e-14;

/*****************************************************************************/
/*  generic checks, for all geometries                                       */
/*****************************************************************************/

void absco_checks(
    const char* method,
    double *const A_S_SC,
    double *const A_C_SC,
    double *const A_C_C,
    double scattering_angle,
    double mu_i_S,
    double mu_f_S,
    double mu_i_C,
    double mu_f_C )
{
    if ( scattering_angle<0 || scattering_angle>180 ) {
        fprintf( stderr, "%s: scattering angle not between 0 and 180 deg\n",
                 method );
        exit( EINVAL );
    }
    if ( mu_i_S<0 ) {
        fprintf( stderr, "%s: mu_i_S < 0\n", method );
        exit( EINVAL );
    }
    if ( mu_f_S<0 ) {
        fprintf( stderr, "%s: mu_f_S < 0\n", method );
        exit( EINVAL );
    }
    if ( !mu_i_S && !mu_f_S ) {
        fprintf( stderr, "%s: mu_S = 0\n", method );
        exit( EINVAL );
    }
    if ( mu_i_C<0 ) {
        fprintf( stderr, "%s: mu_i_C < 0\n", method );
        exit( EINVAL );
    }
    if ( mu_f_C<0 ) {
        fprintf( stderr, "%s: mu_f_C < 0\n", method );
        exit( EINVAL );
    }
    if ( !A_S_SC ) {
        fprintf( stderr, "%s: A_S_SC is a null pointer\n", method );
        exit( EINVAL );
    }
    if ( !A_C_SC || !A_C_C ) {
        if ( A_C_SC ) {
            fprintf( stderr, "%s: A_C_C is a null pointer"
                     " while A_C_SC is not\n", method );
            exit( EINVAL );
        }
        if ( A_C_C ) {
            fprintf( stderr, "%s: A_C_SC is a null pointer"
                     " while A_C_C is not\n", method );
            exit( EINVAL );
        }
        if ( mu_i_C || mu_f_C ) {
            fprintf( stderr, "%s: mu_C is not null though A_C_[S]C"
                     " are null pointers\n", method );
            exit( EINVAL );
        }
    } else {
        if ( !mu_i_C && !mu_f_C ) {
            fprintf( stderr, "%s: mu_C = 0 though A_C are requested \n",
                     method );
            exit( EINVAL );
        }
    }
}

void transmission_checks(
    const char* method,
    double mu_S,
    double mu_C)
{
    if ( mu_S<0 ) {
        fprintf( stderr, "%s: mu_S < 0\n", method );
        exit( EINVAL );
    }
    if ( mu_C<0 ) {
        fprintf( stderr, "%s: mu_C < 0\n", method );
        exit( EINVAL );
    }
}


/*****************************************************************************/
/*  absco_slab (auxiliary)                                                   */
/*****************************************************************************/

// Absorption factor for one layer, transmissing geometry

double absco_slab_Atrans ( const double ki, const double kf )
{
    // Arguments are effective thicknesses k=mu*thickness/sin(angle)
    if ( ki<0 || kf<0 ) {
        fprintf( stderr, "BUG: absco_slab_Atrans called with k<0\n" );
        exit( EINVAL );
    }
    return fabs(ki-kf)<sqrt(epsilon) ?
        exp(-ki) * (1 - (kf-ki)/2 + SQR(kf-ki)/12) :
        ( exp(-kf) - exp(-ki) ) / ( ki - kf );
}

// Absorption factor for one layer, reflection geometry

double absco_slab_Arefl ( const double ki, const double kf )
{
    if ( ki<0 || kf<0 ) {
        fprintf( stderr, "BUG: absco_slab_Arefl called with k<0\n" );
        exit( EINVAL );
    }
    return fabs(ki+kf)<epsilon ?
        1 - SQR(ki+kf)/2 :
        ( 1 - exp(-ki-kf) ) / ( ki + kf );
}

void slab_checks(
    const char*  method,
    const bool   withC,
    const double slab_angle,
    const double thickness_S,
    const double thickness_C_front,
    const double thickness_C_rear)
{
    if ( slab_angle<=0 || slab_angle>=180 ) {
        fprintf( stderr, "%s: slab angle not between 0 and 180 deg\n",
                 method );
        exit( EINVAL );
    }
    if ( thickness_S<=0 ) {
        fprintf( stderr, "%s: thickness_S <= 0\n", method );
        exit( EINVAL );
    }
    if ( thickness_C_front<0 ) {
        fprintf( stderr, "%s: thickness_C_front < 0\n", method );
        exit( EINVAL );
    }
    if ( thickness_C_rear<0 ) {
        fprintf( stderr, "%s: thickness_C_rear < 0\n", method );
        exit( EINVAL );
    }
    if ( withC && thickness_C_front+thickness_C_rear<=0 ) {
        fprintf( stderr, "%s: thickness_C <= 0\n", method );
        exit( EINVAL );
    }
}

/*****************************************************************************/
/*  The slab API                                                             */
/*****************************************************************************/

void absco_slab (
    double *const A_S_SC,
    double *const A_C_SC,
    double *const A_C_C,
    const double scattering_angle,
    const double mu_i_S,
    const double mu_f_S,
    const double mu_i_C,
    const double mu_f_C,
    const double slab_angle,
    const double thickness_S,
    const double thickness_C_front,
    const double thickness_C_rear)
{
    // Check input variables.
    absco_checks( "absco_slab", A_S_SC, A_C_SC, A_C_C, scattering_angle,
                  mu_i_S, mu_f_S, mu_i_C, mu_f_C );
    slab_checks( "absco_slab", A_C_SC!=NULL, slab_angle,
                 thickness_S, thickness_C_front, thickness_C_rear );

    // Conversion to internal variables.
    const double Theta = scattering_angle*M_PI/180;
    const double alpha = slab_angle      *M_PI/180;
    const double sin_i = fabs( sin( alpha ) );
    const double sin_f = fabs( sin( Theta-alpha) );
    const double eff_S_i  = mu_i_S * thickness_S;
    const double eff_S_f  = mu_f_S * thickness_S;
    const double eff_C1_i = mu_i_C * thickness_C_front;
    const double eff_C1_f = mu_f_C * thickness_C_front;
    const double eff_C2_i = mu_i_C * thickness_C_rear;
    const double eff_C2_f = mu_f_C * thickness_C_rear;
    const double eff_i    = eff_S_i + eff_C1_i + eff_C2_i;
    const double eff_f    = eff_S_f + eff_C1_f + eff_C2_f;
    const double thickness_C = thickness_C_front + thickness_C_rear;

    // Absorption factors: scattering in slab direction.
    if ( sin_f< epsilon ) {
        if( A_S_SC )
            *A_S_SC = 0;
        if( A_C_SC )
            *A_C_SC = 0;
        if( A_C_C )
            *A_C_C = 0;
        return;
    }

    // Further reduced variables.
    const double k_S_i  = eff_S_i  / sin_i;
    const double k_S_f  = eff_S_f  / sin_f;
    const double k_C1_i = eff_C1_i / sin_i;
    const double k_C1_f = eff_C1_f / sin_f;
    const double k_C2_i = eff_C2_i / sin_i;
    const double k_C2_f = eff_C2_f / sin_f;

    // Absorption factors.
    if( Theta<alpha ) {
        // Transmission geometry.
        if( A_S_SC )
            *A_S_SC = exp( -eff_C1_i/sin_i-eff_C2_f/sin_f ) *
                absco_slab_Atrans( k_S_i, k_S_f );
        if( A_C_C )
            *A_C_C =
                ( thickness_C_front * exp(-eff_C2_f/sin_f)
                  * absco_slab_Atrans( k_C1_i, k_C1_f ) +
                  thickness_C_rear * exp(-eff_C1_i/sin_i)
                  * absco_slab_Atrans( k_C2_i, k_C2_f ) ) /
                thickness_C;
        if( A_C_SC )
            *A_C_SC =
                ( thickness_C_front * exp(-(eff_C2_f+eff_S_f)/sin_f)
                  * absco_slab_Atrans( k_C1_i, k_C1_f ) +
                  thickness_C_rear * exp(-(eff_C1_i+eff_S_i)/sin_i)
                  * absco_slab_Atrans( k_C2_i, k_C2_f )  ) /
                thickness_C;
    } else {
        // Reflection geometry.
        if( A_S_SC )
            *A_S_SC = exp( -eff_C1_i/sin_i-eff_C1_f/sin_f ) *
                absco_slab_Arefl( k_S_i, k_S_f );
        if( A_C_C )
            *A_C_C =
                ( thickness_C_front
                  * absco_slab_Arefl( k_C1_i, k_C1_f ) +
                  thickness_C_rear
                  * exp(-eff_C1_i/sin_i-eff_C1_f/sin_f)
                  * absco_slab_Arefl( k_C2_i, k_C2_f ) ) /
                thickness_C;
        if( A_C_SC )
            *A_C_SC =
                ( thickness_C_front
                  * absco_slab_Arefl( k_C1_i, k_C1_f ) +
                  thickness_C_rear
                  * exp(-(eff_C1_i+eff_S_i)/sin_i-(eff_C1_f+eff_S_f)/sin_f)
                  * absco_slab_Arefl( k_C2_i, k_C2_f ) ) /
                thickness_C;
    }
}

double transmission_slab (
    const double outgoing_angle,
    const double mu_S,
    const double mu_C,
    const double slab_angle,
    const double thickness_S,
    const double thickness_C_front,
    const double thickness_C_rear)
{
    transmission_checks( "transmission_slab", mu_S, mu_C );
    slab_checks( "transmission_slab", false, slab_angle,
                 thickness_S, thickness_C_front, thickness_C_rear );

    const double sin_f = sin( fabs(outgoing_angle-slab_angle)*M_PI/180 );
    const double eff   = mu_S * thickness_S +
        mu_C * ( thickness_C_front + thickness_C_rear );

    if( sin_f<epsilon )
        return 0;
    return exp( -eff/sin_f );
}

/*****************************************************************************/
/*  absco_tube (auxiliary)                                                   */
/*****************************************************************************/

void tube_checks(
    const char*  method,
    const bool   withC,
    const double radius,
    const double thickness_S,
    const double thickness_C_inner,
    const double thickness_C_outer)
{
    if ( thickness_S<=0 ) {
        fprintf( stderr, "%s: thickness_S <= 0\n", method );
        exit( EINVAL );
    }
    if ( thickness_C_inner<0 ) {
        fprintf( stderr, "%s: thickness_C_inner < 0\n", method );
        exit( EINVAL );
    }
    if ( thickness_C_outer<0 ) {
        fprintf( stderr, "%s: thickness_C_outer < 0\n", method );
        exit( EINVAL );
    }
    if ( radius-thickness_S-thickness_C_inner<-epsilon ) {
        fprintf( stderr, "%s: inner radius < 0\n", method );
        exit( EINVAL );
    }
    if ( withC && thickness_C_inner+thickness_C_outer<=0 ) {
        fprintf( stderr, "%s: thickness_C <= 0\n", method );
        exit( EINVAL );
    }
}

//! Compute path length across a tube segment.

double tubePath(
    const double r,      // radius where ray
    const double angle,  // angle where ray starts (rad)
                         //   angle=0 is the outgoing ray direction
    const double Ro,     // outer radius of tube
    const double d )     // thickness of tube
{
    const double Ri = Ro-d;

    const double radic_o = SQR(Ro) - SQR( r*sin(angle) );
    const double radic_i = SQR(Ri) - SQR( r*sin(angle) );
    // Intersections are at -r*cos(angle)+-sqrt(radic) if the sqrt exists.

    if ( r<Ri ) {
        return sqrt( radic_o ) - sqrt( radic_i );
    }

    double path;
    if ( r>Ro ) {
        if ( radic_o <= 0 ) {
            return 0;
        } else if ( cos(angle) < 0 ) {
            path = 2*sqrt( radic_o );
        } else {
            return 0;
        }
    } else {
        path = -r*cos(angle) + sqrt( radic_o );
    }

    if ( radic_i <= 0 ) {
        return path;
    } else if ( cos(angle) < 0 ) {
        return path - 2*sqrt( radic_i );
    } else {
        return path;
    }
}

typedef struct {
    size_t nws;
    gsl_integration_workspace *ws;
    gsl_function F;
    double epsabs;
    double epsrel;
} quadAuxTyp;

typedef struct {
    double phi;
    double Theta;
    double mu_i_S;
    double mu_f_S;
    double mu_i_C;
    double mu_f_C;
    double mu_i[3];
    double mu_f[3];
    double R[4];
    bool   withS;
    double Ri;
    double Ro;
    quadAuxTyp *K;
} tubeParTyp;

//! Inner integrand.

double abscoTubeKernel( double r, void *data )
{
    tubeParTyp *P = (tubeParTyp*) data;

    double arg = 0;
    for ( int j=0; j<3; ++j ) {
        if ( !P->mu_i[j] && !P->mu_f[j] )
            continue;
        arg +=
            P->mu_i[j]
            * tubePath( r, P->phi-M_PI,     P->R[j+1], P->R[j+1]-P->R[j] ) +
            P->mu_f[j]
            * tubePath( r, P->phi-P->Theta, P->R[j+1], P->R[j+1]-P->R[j] );
    }
    return r*exp( -arg );
}

//! Outer integrand, performing the inner integration.

double abscoTubeIntegrand( double phi, void *data )
{
    tubeParTyp *P = (tubeParTyp*) data;
    quadAuxTyp *K = (quadAuxTyp*) P->K;

    double result, abserr;
    int ret;
    P->phi = phi;
    ret = gsl_integration_qags (
        &(K->F), P->Ri, P->Ro,
        K->epsabs, K->epsrel, K->nws, K->ws,
        &result, &abserr);
    return result;
}

//! Integrate absorption factor over interaction points within segment j.

double tubeQuad( quadAuxTyp *Q, int j, bool withS )
{
    tubeParTyp *P = (tubeParTyp *) Q->F.params;
    double result, abserr;
    int ret;
    if ( withS ) {
        P->mu_i[1] = P->mu_i_S;
        P->mu_f[1] = P->mu_f_S;
    } else {
        P->mu_i[1] = 0;
        P->mu_f[1] = 0;
    }
    P->withS = withS;
    P->Ri = P->R[j];
    P->Ro = P->R[j+1];
    ret = gsl_integration_qags (
        &(Q->F), 0, 2*M_PI,
        Q->epsabs, Q->epsrel, Q->nws, Q->ws,
        &result, &abserr );
    return result;
}

//! Integrand for transmission.

double transmissionTubeKernel( double y, void*data )
{
    tubeParTyp *P = (tubeParTyp*) data;
    double R = P->R[3];
    if ( fabs(y)>R ) {
        fprintf( stderr, "Unexpected |y|>R in transmissionTubeKernel\n" );
        exit( EINVAL );
    }
    double arg = 0;
    double path, ro, ri;
    for ( int j=0; j<3; ++j ) {
        if ( !P->mu_f[j] )
            continue;
        ro = SQR(P->R[j+1]) - SQR(y);
        if( ro<= 0 )
            continue;
        path = sqrt( ro );
        ri = SQR(P->R[j]) - SQR(y);
        if( ri> 0 )
            path -= sqrt(ri);
        arg += P->mu_f[j] * 2 * path;
    }
    double r = sqrt( SQR(R) - SQR(y) );
    return exp(-arg);
}

/*****************************************************************************/
/*  The tube API                                                             */
/*****************************************************************************/

void absco_tube (
    double *const A_S_SC,
    double *const A_C_SC,
    double *const A_C_C,
    const double scattering_angle,
    const double mu_i_S,
    const double mu_f_S,
    const double mu_i_C,
    const double mu_f_C,
    const double radius,
    const double thickness_S,
    const double thickness_C_inner,
    const double thickness_C_outer)
{
    // Check input.
    absco_checks( "absco_tube", A_S_SC, A_C_SC, A_C_C, scattering_angle,
                  mu_i_S, mu_f_S, mu_i_C, mu_f_C );
    tube_checks( "absco_tube", A_C_SC!=NULL,
                 radius, thickness_S, thickness_C_inner, thickness_C_outer );

    // Conversion to internal variables.
    const double Theta = scattering_angle*M_PI/180;

    tubeParTyp P;
    quadAuxTyp Q, K;

    // for the outer integration
    Q.nws = 10000;
    Q.ws = gsl_integration_workspace_alloc( Q.nws );
    Q.F.function = abscoTubeIntegrand;
    Q.F.params = (void*) &P;
    Q.epsabs = 1.e-4;
    Q.epsrel = 1.e-4;

    // for the inner integration
    K.nws = 10000;
    K.ws = gsl_integration_workspace_alloc( K.nws );
    K.F.function = abscoTubeKernel;
    K.F.params = (void*) &P;
    K.epsabs = 3.e-5;
    K.epsrel = 3.e-5;

    // for both integrands
    P.K = &K;
    P.Theta  = Theta;
    P.mu_i_S = mu_i_S;
    P.mu_f_S = mu_f_S;
    P.mu_i_C = mu_i_C;
    P.mu_f_C = mu_f_C;
    P.R[0] = radius-thickness_S-thickness_C_inner;
    P.R[1] = radius-thickness_S;
    P.R[2] = radius;
    P.R[3] = radius+thickness_C_outer;
    P.mu_i[0] = mu_i_C;
    P.mu_f[0] = mu_f_C;
    P.mu_i[2] = mu_i_C;
    P.mu_f[2] = mu_f_C;

    // Areas of the layers, and another round of checks.
    double A[3];
    for ( int j=0; j<3; ++j )
        A[j] = M_PI * ( SQR(P.R[j+1]) - SQR(P.R[j]) );
    if ( A_S_SC && A[1]<=0 ) {
        fprintf( stderr, "absco_tube: finite sample layer required for"
                 " computation of A_S_SC\n" );
        exit( EINVAL );
    }
    if ( ( A_C_SC || A_C_C ) && ( A[0]+A[2] <= 0 ) ) {
        fprintf( stderr, "absco_tube: finite container layer required for"
                 " computation of A_C_SC or A_C_C\n" );
        exit( EINVAL );
    }

    // Compute the absorption coefficients.
    if ( A_S_SC )
        *A_S_SC = tubeQuad( &Q, 1, 1 ) / A[1];
    if ( A_C_SC )
        *A_C_SC =(tubeQuad( &Q, 0, 1 ) + tubeQuad( &Q, 2, 1 )) / (A[0] + A[2]);
    if ( A_C_C )
        *A_C_C  =(tubeQuad( &Q, 0, 0 ) + tubeQuad( &Q, 2, 0 )) / (A[0] + A[2]);

    // Cleanup.
    gsl_integration_workspace_free( Q.ws );
    gsl_integration_workspace_free( K.ws );
}

double transmission_tube (
    const double mu_S,
    const double mu_C,
    const double radius,
    const double thickness_S,
    const double thickness_C_inner,
    const double thickness_C_outer)
{
    // Check input.
    transmission_checks( "transmission_tube", mu_S, mu_C );
    tube_checks( "absco_tube", false,
                 radius, thickness_S, thickness_C_inner, thickness_C_outer );

    // Prepare integration.
    tubeParTyp P;
    quadAuxTyp Q;

    Q.nws = 10000;
    Q.ws = gsl_integration_workspace_alloc( Q.nws );
    Q.F.function = transmissionTubeKernel;
    Q.F.params = (void*) &P;
    Q.epsabs = 1.e-4;
    Q.epsrel = 1.e-4;

    P.K = &Q;
    P.R[0] = radius-thickness_S-thickness_C_inner;
    P.R[1] = radius-thickness_S;
    P.R[2] = radius;
    P.R[3] = radius+thickness_C_outer;
    P.mu_f[0] = mu_C;
    P.mu_f[1] = mu_S;
    P.mu_f[2] = mu_C;

    // Integrate.
    double result, abserr;
    int ret;
    ret = gsl_integration_qags ( &(Q.F), 0, P.R[3],
                                 Q.epsabs, Q.epsrel, Q.nws, Q.ws,
                                 &result, &abserr );
    result /= P.R[3];

    // Cleanup.
    gsl_integration_workspace_free( Q.ws );

    return result;
}
