cimport cython


cdef extern from "absco.h":
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




def py_absco_slab(double scattering_angle, double mu_i_S=0.45,
                  double mu_f_S=0.45, double mu_i_C=0.05,
                  double mu_f_C=0.05, double slab_angle=45,
                  double thickness_S=0.03, double thickness_C_front=0.5,
                  double thickness_C_rear=0.5 ):

    cdef double A_S_SC = 0
    cdef double A_C_SC = 0
    cdef double A_C_C  = 0

    cdef double *assc_ptr = &A_S_SC
    cdef double *acsc_ptr = &A_C_SC
    cdef double *acc_ptr  = &A_C_C

    absco_slab(assc_ptr, acsc_ptr, acc_ptr, scattering_angle, mu_i_S,
               mu_f_S, mu_i_C, mu_f_C, slab_angle,
               thickness_S, thickness_C_front, thickness_C_rear)


    return A_S_SC, A_C_SC, A_C_C






def py_absco_tube( double scattering_angle, double mu_i_S=0.20, double mu_f_S=0.20, double mu_i_C=0.05,
        double mu_f_C=0.05, double radius=20, double thickness_S=0.05, double thickness_C_inner=0.5,
        double thickness_C_outer=0.5 ):

    cdef double A_S_SC = 0
    cdef double A_C_SC = 0
    cdef double A_C_C  = 0

    cdef double *assc_ptr = &A_S_SC
    cdef double *acsc_ptr = &A_C_SC
    cdef double *acc_ptr  = &A_C_C

    absco_tube(assc_ptr, acsc_ptr, acc_ptr, scattering_angle, mu_i_S, mu_f_S, mu_i_C, mu_f_C, radius,
                thickness_S, thickness_C_inner, thickness_C_outer)


    return A_S_SC, A_C_SC, A_C_C
