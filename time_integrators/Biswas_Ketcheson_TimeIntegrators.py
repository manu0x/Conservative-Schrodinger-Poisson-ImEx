#!/usr/bin/env python
# coding: utf-8
##
#       This file is picked is originally from https://github.com/abhibsws/Multiple_Relaxation_NLS/blob/main/NLS_Multi_Soliton_SP_One_Inv/TimeIntegrators.py
###
import numpy as np

# Imex scheme
def ImEx_schemes(s,p,emp,sch_no):
    #3rd order ImEx with b and 2nd order ImEx with bhat. This method is taken from Implicit-explicit 
    # Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.
    if s == 4 and p == 3 and emp == 2 and sch_no == 2:
        rkim = np.array([ [ 0,              0,             0,             0],
                          [ 0,   0.4358665215,             0,             0],
                          [ 0,   0.2820667392,  0.4358665215,             0],
                          [ 0,   1.2084966490, -0.6443631710,  0.4358665215] ])
        rkex = np.array([ [ 0,                       0,             0,             0],
                          [ 0.4358665215,            0,             0,             0],
                          [ 0.3212788860, 0.3966543747,             0,             0],
                          [-0.1058582960, 0.5529291479,  0.5529291479,             0] ])
        c = sum(rkex.T)
        b = np.array([0,   1.2084966490, -0.6443631710,  0.4358665215])
        bhat = np.array([0,0.886315063820486,0 ,0.113684936179514])
    #3rd order ImEx with b and 2nd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
    #for convection–diffusion–reaction equations by Kennedy and Carpenter.
    ############################################
    #Following method is ARK3(2)4L[2]SA
    if s == 4 and p == 3 and emp == 2 and sch_no == 3:
        rkim = np.array([ [ 0,              0,             0,             0],
                  [1767732205903/4055673282236, 1767732205903/4055673282236, 0, 0],
                  [2746238789719/10658868560708, -640167445237/6845629431997, 1767732205903/4055673282236, 0],              
                  [1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821,                 1767732205903/4055673282236] ])

        rkex = np.array([ [0,                            0,         0,             0],
                          [1767732205903/2027836641118,  0,         0,             0],
                          [5535828885825/10492691773637, 788022342437/10882634858940, 0, 0],
                          [6485989280629/16251701735622, -4246266847089/9704473918619, 10755448449292/10357097424841, 0] ])

        c = np.array([0, 1767732205903/2027836641118, 3/5, 1])
        b = np.array([1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821, 1767732205903/4055673282236])
        bhat = np.array([2756255671327/12835298489170, -10771552573575/22201958757719, 9247589265047/10645013368117, 2193209047091/5459859503100])

    #4th order ImEx with b and 3rd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
    #for convection–diffusion–reaction equations by Kennedy and Carpenter.
    ##########Following method is  ARK4(3)6L[2]SA   ##############
    elif s == 6 and p == 4 and emp == 3 and sch_no == 4:
        rkex = np.array([ [0, 0, 0, 0, 0, 0],
                  [1/2, 0, 0, 0, 0, 0],
                  [13861/62500, 6889/62500, 0, 0, 0, 0], 
                  [-116923316275/2393684061468, -2731218467317/15368042101831, 9408046702089/11113171139209, 0, 0, 0], 
                  [-451086348788/2902428689909, -2682348792572/7519795681897, 12662868775082/11960479115383, 3355817975965/11060851509271, 0, 0], 
                  [647845179188/3216320057751, 73281519250/8382639484533, 552539513391/3454668386233, 3354512671639/8306763924573, 4040/17871, 0] 
                ])

        rkim = np.array([ [0, 0, 0, 0, 0, 0],
                          [1/4, 1/4, 0, 0, 0, 0],
                          [8611/62500, -1743/31250, 1/4, 0, 0, 0],
                          [5012029/34652500, -654441/2922500, 174375/388108, 1/4, 0, 0],
                          [15267082809/155376265600, -71443401/120774400, 730878875/902184768, 2285395/8070912, 1/4, 0],
                          [82889/524892, 0, 15625/83664, 69875/102672, -2260/8211, 1/4]
                        ])

        c = np.array([0, 1/2, 83/250, 31/50, 17/20, 1])
        b = np.array([82889/524892, 0, 15625/83664, 69875/102672, -2260/8211, 1/4])
        bhat = np.array([4586570599/29645900160, 0, 178811875/945068544, 814220225/1159782912, -3700637/11593932, 61727/225920])

    
        
    return rkim, rkex, c, b, bhat 

# Operator splitting methods
def Op_Sp_Coeff(s,p,sch_no):
    # A 1-stage 1st order operator splitting method: Lie-Trotter
    if s == 1 and p == 1 and sch_no == 0:      
        a = np.array([1.])
        b = np.array([1.])
    # A 2-stage 2nd order operator splitting method: Strang splitting
    elif s == 2 and p == 2 and sch_no == 1:  
        a = np.array([1/2,1/2])
        b = np.array([1,0.])
    # A 5-stage 4th order operator splitting method
    elif s == 5 and p == 4 and sch_no == 2:
        a = np.array([0.267171359000977615,-0.0338279096695056672,
                      0.5333131013370561044,-0.0338279096695056672
                      ,0.267171359000977615])
        b = np.array([-0.361837907604416033,0.861837907604416033,
                      0.861837907604416033,-0.361837907604416033,0.])
        
    return a, b


def load_imex_scheme(scheme_name):

    if scheme_name=="SSP2-ImEx(3,3,2)":
    
        # SSP2-IMEX(3,3,2): 2nd order L-stable type I ImEX-RK, explicit part - NOT FSAL, implicit part - SA; NOT GSA.


        l = 1.0
        A_im = np.array([[l/4.0, 0.0, 0.0],
                   [0, l/4, 0],
                  [l/3.0, l/3.0, l/3.0]])
        b_im = np.array([l/3.0, l/3.0, l/3.0])
        c_im = np.array([l/4.0, l/4.0, l])

        A_ex = np.array([[0.0, 0.0, 0.0],
                        [l/2.0, 0.0, 0.0],
                        [l/2.0, l/2.0, 0.0]])
        b_ex = np.array([l/3.0, l/3.0, l/3.0])
        c_ex = np.array([0, l/2.0, l])
        return A_im, A_ex, b_im,b_ex,c_im,c_ex,3
    
    elif scheme_name=="SSP3-ImEx(3,4,3)":
        # SSP3-IMEX(3,4,3): 3rd order L-stable type I ImEX-RK, explicit part - NOT FSAL, implicit part - NOT SA; NOT GSA.
        l = 1.0
        a= 0.24169426078821
        b = 0.06042356519705
        eta = 0.12915286960590
        A_im = np.array([[a, 0.0, 0.0, 0.0],
               [-a, a, 0.0, 0.0],
               [0.0, l-a, a, 0.0],
               [b, eta, l/2-b-eta-a, a]])
        
        b_im = np.array([0.0, l/6.0, l/6.0, 2.0*l/3.0])
        c_im = np.array([a, 0.0,l,l/2.0])
        
        A_ex = np.array([[0.0, 0.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0, 0.0],
                  [0.0, l, 0.0, 0.0],
                  [0.0, l/4.0, l/4.0, 0.0]])
        b_ex = np.array([0.0, l/6.0, l/6.0, 2.0*l/3.0])
        c_ex = np.array([0.0, 0.0, l, l/2.0])

        return A_im, A_ex, b_im,b_ex,c_im,c_ex,4
    
    elif scheme_name=="AGSA(3,4,2)":
        # AGSA(3,4,2): 2nd order type I ImEX-RK, explicit part - FSAL, implicit part - SA; GSA.

        l = 1.0
        A_im = np.array([ [168999711.0*l/74248304.0, 0.0, 0.0, 0.0],
                [44004295.0*l/24775207.0, 202439144.0*l/118586105.0, 0.0, 0.0],
                [-6418119.0*l/169001713.0, -748951821.0*l/1043823139.0, 12015439.0*l/183058594.0, 0.0],
               [-370145222.0*l/355758315.0, l/3.0, 0.0, 202439144.0*l/118586105.0]])
        
        b_im = np.array([-370145222.0*l/355758315.0, l/3.0, 0.0, 202439144.0*l/118586105.0])
        c_im = np.array([168999711.0*l/74248304.0, 10233769644823783.0*l/2937995298698735.0, -22277245178359531777915943.0*l/32292981880895017395247558.0 , 1.0])

        A_ex = np.array([ [0.0, 0.0, 0.0, 0.0],
                  [-139833537*l/38613965, 0.0, 0.0, 0.0],
                   [85870407*l/49798258, -121251843*l/1756367063, 0, 0],
                   [l/6, l/6, 2*l/3, 0]])
        b_ex = np.array([l/6, l/6, 2*l/3, 0])
        c_ex = np.array([0, -139833537*l/38613965, 144781823980515147*l/87464020145976254, 1 ])

        return A_im, A_ex, b_im,b_ex,c_im,c_ex,4
    
    elif scheme_name=="ARS(4,4,3)":
        #Here is a type II method that is GSA:

        l = 1.0
        A_im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, l/2.0, 0.0, 0.0, 0.0],
                    [0.0, l/6, l/2, 0.0, 0.0],
                    [0.0, -l/2, l/2, l/2, 0.0],
                    [0, 3*l/2, -3*l/2, l/2, l/2]])
        b_im = np.array([0, 3*l/2, -3*l/2, l/2, l/2])
        c_im = np.array([0, l/2, 2*l/3, l/2, l])

        A_ex = np.array([[0, 0, 0, 0, 0],
                        [l/2, 0, 0, 0, 0],
                        [11*l/18, l/18,0, 0, 0],
                        [5*l/6, -5*l/6, l/2, 0, 0],
                        [l/4, 7*l/4, 3*l/4, -7*l/4, 0]])
        b_ex = np.array([l/4, 7*l/4, 3*l/4, -7*l/4, 0])
        c_ex = np.array([0, l/2, 2*l/3, l/2, l])

        return A_im, A_ex, b_im,b_ex,c_im,c_ex,5

        
        


