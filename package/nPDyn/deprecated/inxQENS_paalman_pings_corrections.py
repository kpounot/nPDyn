''' This module is used to compute the paalman-pings coefficients for corrections of
    the neutron scattering data. It's currently limited to flat cans and is adapted
    for nPDyn package from the libabsco code (Wuttke J., 2012). 

    References:
    Wuttke J. (2012) Absorption-Correction Factors for Scattering
    from Flat or Tubular Samples: Open-Source Implementation libabsco, and Why it Should 
    be Used with Caution. '''

import sys
import numpy as np
import argParser
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QWidget, QDialog 
import inxBinQENS

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


def plot_coeff(kwargs):

    ''' Plot the As,sc, Ac,sc and Ac,c coefficient for each angle between 0 and 180° '''

    #_Compute the coefficients for each angle and store them in an array
    coeff_list = [pp_coeff(angle, **kwargs) 
                        for angle in np.arange(1, 180, 1)] 
    coeff_list = np.vstack([[val[0], val[1], val[2]] for val in coeff_list])     
    name_list = [r'$A_{s, sc}$', r'$A_{c, sc}$', r'$A_{c, c}$']

    #_Plot the three coefficients
    for i in range(3):
        plt.plot(np.arange(1, 180, 1), coeff_list[:,i], label=name_list[i])
    
    plt.xticks(np.arange(5, 180, 10), ('%d° (%.1f)' % 
               (val, 4*np.pi*np.sin(np.pi*val/180) / kwargs['neutron_wavelength']) 
               for val in np.arange(5, 180, 10)),
               rotation=45, ha='right')
    plt.xlabel(r'$Angle \ \Theta \ (q)$', fontsize=18)
    plt.ylabel(r'$Absorption \ coefficient$', fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right', fontsize=18)
    plt.show() 

def transform_data(dataFile, geometry='flat'):
    
    ''' Use the pp_coeff function to calculate the new corrected intensities. 
  
        Returns --> dataList as a named tuple as used within the nPDyn package '''


    if geometry == 'flat':
        kwargs = {'mu_i_C': 0.107, 'mu_f_C': 0.107, 'mu_i_S': 0.8, 'mu_f_S': 0.8, 
                  'slab_angle': 45, 'thickness_C_front': 0.4, 'thickness_C_rear': 0.4,
                  'thickness_S': 0.03, 'angle_unit': 'degree', 'neutron_wavelength': 6.27}
       
        #_Replace de default values by the ones given by the user in the system shell 
        for key in kwargs.keys():
            if key in karg:
                kwargs[key] = float(karg[key]) if key != 'angle_unit' else karg[key]


    #_Extract the data from the file
    dataList = inxBinQENS.inxBin(dataFile, karg['binS'])

    #_Get the empty cell data
    message = QMessageBox.information(QWidget(), 'File selection',
                                      'Please select the empty cell data file.')
    fileToOpen = QFileDialog().getOpenFileName()[0]
    EC_data = inxBinQENS.inxBin(fileToOpen, karg['binS'])

    #_For each q-value, compute the corresponding coefficients and correct the data
    kwargs['angle_unit'] = 'q'
    for i, val in enumerate(dataList):
        A_s_sc, A_c_sc, A_c_c = pp_coeff(val.qVal, **kwargs)
        dataList[i] = val._replace(intensities=(
                        1/A_s_sc * (EC_data[i].intensities + val.intensities)
                        - A_c_sc/(A_s_sc*A_c_c) * EC_data[i].intensities))

    return dataList

if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    #plot_coeff(kwargs)

    arg, karg

    dataList = transform_data(sys.argv[1])
    r = ''                                          
    for i, val in enumerate(dataList):
        r += '%d    1    2    0   0   0   0   %d\n' % (len(val.energies)+3, len(val.energies))
        r += '    title...\n'
        r += '     %s    %s    0    0    0    0\n' % (val.qVal, dataList[-1].qVal)
        r += '    0    0    0\n'
        for j, values in enumerate(val.energies):
            r += '    %.5f    %.5f    %.5f\n' % (values, val.intensities[j], val.errors[j])
    r = r[:-2]
    print(r, flush=True)

    sys.exit(app.exec_())
