''' This module is used to compute the paalman-pings coefficients for corrections of
    the neutron scattering data. It's adapted for nPDyn package from the libabsco code (Wuttke J., 2012). 

    References:
    Wuttke J. (2012) Absorption-Correction Factors for Scattering
    from Flat or Tubular Samples: Open-Source Implementation libabsco, and Why it Should 
    be Used with Caution. '''

import sys
import numpy as np

def pp_coeff_flat(angle, mu_i_S=0.8, mu_i_C=0.104, mu_f_S=0.8, mu_f_C=0.104, 
             thickness_C_rear=0.5, thickness_C_front=0.5, thickness_S=0.03, slab_angle=45, 
             angle_unit='q', neutron_wavelength=6.27):

    ''' Function to compute the As,sc, Ac,sc and Ac,c paalman-pings coefficients
        The angle_unit keyword argument can be 'q' for momentum transfer (kf - ki) 
        or 'degree' for the real angle between ki and kf ( arccos[<ki, kf> / (||ki||*||kf||)] )
        
        Returns --> As,sc, Ac,sc, Ac,c '''


    #_Convert momentum transfer values to the angle between ki and kf
    if angle_unit == 'q':
        angle = np.arcsin(neutron_wavelength * angle / (4 * np.pi))

    #_Define the internal variables
    thickness_total = thickness_S + thickness_C_rear + thickness_C_front
    thickness_C = thickness_C_rear + thickness_C_front
    alpha = slab_angle * np.pi / 180
    theta = angle * np.pi / 180
    sin_i = np.abs(np.sin(alpha))
    sin_f = np.abs(np.sin(theta-alpha))
    ki_S = mu_i_S * thickness_S / sin_i
    kf_S = mu_f_S * thickness_S / sin_f
    ki_C_front = mu_i_C * thickness_C_front / sin_i
    kf_C_front = mu_f_C * thickness_C_front / sin_f
    ki_C_rear = mu_i_C * thickness_C_rear / sin_i
    kf_C_rear = mu_f_C * thickness_C_rear / sin_f

    #_Analytic results for the integral (9) in Wuttke (2012)
    trans_func = lambda ki, kf: (np.exp(-kf) - np.exp(-ki)) / (ki - kf)
    refl_func = lambda ki, kf: (1 - np.exp(-(ki + kf))) / (ki + kf)

    #_Calculate the As,sc, Ac,sc and Ac,c parameters
    if theta < alpha:       #_Transmission
        A_s_sc = np.exp(-ki_C_front - kf_C_rear) * trans_func(ki_S, kf_S)

        A_c_sc = ((thickness_C_front * np.exp(-(kf_C_rear + kf_S)) 
                 * trans_func(ki_C_front, kf_C_front)
                 + thickness_C_rear * np.exp(-(ki_C_front + ki_S))
                 * trans_func(ki_C_rear, kf_C_rear)) 
                 / thickness_C)

        A_c_c = ((thickness_C_front * np.exp(-kf_C_rear) * trans_func(ki_C_front, kf_C_front)
                + thickness_C_rear * np.exp(-ki_C_front) * trans_func(ki_C_rear, kf_C_rear))
                / thickness_C)


    elif theta > alpha:     #_Reflection
        A_s_sc = np.exp(-ki_C_front - kf_C_front) * refl_func(ki_S, kf_S)

        A_c_sc = ((thickness_C_front * refl_func(ki_C_front, kf_C_front)
                 + thickness_C_rear * np.exp(-(ki_C_front + ki_S) - (kf_C_front + kf_S))
                 * refl_func(ki_C_rear, kf_C_rear))
                 / thickness_C)

        A_c_c = ((thickness_C_front * refl_func(ki_C_front, kf_C_front)
                + thickness_C_rear * np.exp(-(ki_C_front + kf_C_front)) 
                * refl_func(ki_C_rear, kf_C_rear))
                / thickness_C)



    else:   #_theta=alpha --> full absorption
        A_s_sc = 0
        A_c_sc = 0
        A_c_c = 0    

    return A_s_sc, A_c_sc, A_c_c







def pp_coeff_tubular(angle, mu_i_S=0.8, mu_i_C=0.104, mu_f_S=0.8, mu_f_C=0.104, 
             radius=30, thickness_S=0.15, thickness_C_inner=0.2, thickness_C_outer=0.2, 
             angle_unit='q', neutron_wavelength=6.27):

    ''' Function to compute the As,sc, Ac,sc and Ac,c paalman-pings coefficients
        The angle_unit keyword argument can be 'q' for momentum transfer (kf - ki) 
        or 'degree' for the real angle between ki and kf ( arccos[<ki, kf> / (||ki||*||kf||)] )
        
        Returns --> As,sc, Ac,sc, Ac,c '''


    #_Convert momentum transfer values to the angle between ki and kf
    if angle_unit == 'q':
        angle = np.arcsin(neutron_wavelength * angle / (4 * np.pi))

    #_Define the internal variables
    theta = angle * np.pi / 180
    sin_i = np.abs(np.sin(alpha))
    sin_f = np.abs(np.sin(theta-alpha))
    ki_S = mu_i_S * thickness_S / sin_i
    kf_S = mu_f_S * thickness_S / sin_f
    ki_C_front = mu_i_C * thickness_C_front / sin_i
    kf_C_front = mu_f_C * thickness_C_front / sin_f
    ki_C_rear = mu_i_C * thickness_C_rear / sin_i
    kf_C_rear = mu_f_C * thickness_C_rear / sin_f

    #_Analytic results for the integral (9) in Wuttke (2012)
    trans_func = lambda ki, kf: (np.exp(-kf) - np.exp(-ki)) / (ki - kf)
    refl_func = lambda ki, kf: (1 - np.exp(-(ki + kf))) / (ki + kf)

    #_Calculate the As,sc, Ac,sc and Ac,c parameters
    if theta < alpha:       #_Transmission
        A_s_sc = np.exp(-ki_C_front - kf_C_rear) * trans_func(ki_S, kf_S)

        A_c_sc = ((thickness_C_front * np.exp(-(kf_C_rear + kf_S)) 
                 * trans_func(ki_C_front, kf_C_front)
                 + thickness_C_rear * np.exp(-(ki_C_front + ki_S))
                 * trans_func(ki_C_rear, kf_C_rear)) 
                 / thickness_C)

        A_c_c = ((thickness_C_front * np.exp(-kf_C_rear) * trans_func(ki_C_front, kf_C_front)
                + thickness_C_rear * np.exp(-ki_C_front) * trans_func(ki_C_rear, kf_C_rear))
                / thickness_C)


    elif theta > alpha:     #_Reflection
        A_s_sc = np.exp(-ki_C_front - kf_C_front) * refl_func(ki_S, kf_S)

        A_c_sc = ((thickness_C_front * refl_func(ki_C_front, kf_C_front)
                 + thickness_C_rear * np.exp(-(ki_C_front + ki_S) - (kf_C_front + kf_S))
                 * refl_func(ki_C_rear, kf_C_rear))
                 / thickness_C)

        A_c_c = ((thickness_C_front * refl_func(ki_C_front, kf_C_front)
                + thickness_C_rear * np.exp(-(ki_C_front + kf_C_front)) 
                * refl_func(ki_C_rear, kf_C_rear))
                / thickness_C)



    else:   #_theta=alpha --> full absorption
        A_s_sc = 0
        A_c_sc = 0
        A_c_c = 0    

    return A_s_sc, A_c_sc, A_c_c


