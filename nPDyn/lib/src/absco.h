/* absco.h:
 *   Absorption corrections for X-ray and neutron diffraction.
 *
 * Author:
 *   Joachim Wuttke.
 *   http://apps.jcns.fz-juelich.de/absco
 *   More info in absco.cpp
 *
 * Usage:
 *   See man 7 absco
 */

#ifndef __ABSCO_H__
#define __ABSCO_H__

#ifdef __cplusplus
extern "C" {
#endif

void absco_slab (
    double *A_S_SC,
    double *A_C_SC,
    double *A_C_C,
    const double scattering_angle,
    const double mu_i_S,
    const double mu_f_S,
    const double mu_i_C,
    const double mu_f_C,
    const double slab_angle,
    const double thickness_S,
    const double thickness_C_front,
    const double thickness_C_rear);

double transmission_slab (
    const double outgoing_angle,
    const double mu_S,
    const double mu_C,
    const double slab_angle,
    const double thickness_S,
    const double thickness_C_front,
    const double thickness_C_rear);

void absco_tube (
    double *A_S_SC,
    double *A_C_SC,
    double *A_C_C,
    const double scattering_angle,
    const double mu_i_S,
    const double mu_f_S,
    const double mu_i_C,
    const double mu_f_C,
    const double radius,
    const double thickness_S,
    const double thickness_C_inner,
    const double thickness_C_outer);

double transmission_tube (
    const double mu_S,
    const double mu_C,
    const double radius,
    const double thickness_S,
    const double thickness_C_inner,
    const double thickness_C_outer);

#ifdef __cplusplus
}
#endif

#endif /* __ABSCO_H__ */
