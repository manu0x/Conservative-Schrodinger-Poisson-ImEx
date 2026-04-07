import functools

import numpy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np

import math

from jax import grad
import pickle
import os
#import optimistix as optx

from jaxopt import Broyden
import jaxopt
import time




import sys

cwd = "/".join(os.getcwd().split("/")[:-1])
print("CWD:",cwd)
sys.path.append(cwd)
sys.path.append(cwd+"/time_integrators")
from GPE import ImEx



fft = np.fft.fftn
ifft = np.fft.ifftn


##############################  CHoosing ImEX Scheme  

#Choose ImEx scheme

def choose_imex(imex_scheme="default"):
    from Biswas_Ketcheson_TimeIntegrators import ImEx_schemes
    from Biswas_Ketcheson_TimeIntegrators import load_imex_scheme

    if imex_scheme=="default":
        #print("Choosin default ImEx")
        A_ex    = np.array([[0,0,0],[5/6.,0,0],[11/24,11/24,0]])
        A_im = np.array([[2./11,0,0],[205/462.,2./11,0],[2033/4620,21/110,2/11]])
        b_ex = np.array([24/55.,1./5,4./11])
        b_im = b_ex     
        C=None
        imex_stages=3
    if imex_scheme=="a" or imex_scheme=="ImEx3" :
        #3rd order ImEx with b This method is taken from Implicit-explicit 
        # Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.
        #print("Using 3rd order ImEx with b  This method is taken from Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.")

        A_im,A_ex,C,b_im,b_hat = ImEx_schemes(4,3,2,2)
        b_ex = b_im
        imex_stages=4

    if imex_scheme=="b" or imex_scheme=="ARK3(2)4L[2]SA":
        #3rd order ImEx with b. This method is taken from Additive Runge–Kutta schemes 
        #for convection–diffusion–reaction equations by Kennedy and Carpenter.
        #print("Using 3rd order ImEx with b . This method is taken from Additive Runge–Kutta schemes for convection–diffusion–reaction equations by Kennedy and Carpenter.")
        A_im,A_ex,C,b_im,b_hat = ImEx_schemes(4,3,2,3)
        b_ex = b_im
        imex_stages=4
    if imex_scheme=="c" or imex_scheme=="ImEx4" or imex_scheme=="ARK4(3)6L[2]SA" :
        #4th order ImEx with b and 3rd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
        #for convection–diffusion–reaction equations by Kennedy and Carpenter.
        #print("4th order ImEx with b. This method is taken from Additive Runge–Kutta scheme for convection–diffusion–reaction equations by Kennedy and Carpenter.")
        A_im,A_ex,C,b_im,b_hat = ImEx_schemes(6,4,3,4)
        b_ex = b_im
        imex_stages=6

    if imex_scheme=="SSP2-ImEx(3,3,2)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    if imex_scheme=="SSP3-ImEx(3,4,3)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    if imex_scheme=="AGSA(3,4,2)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    if imex_scheme=="ARS(4,4,3)":
        #print("Using ImEx ",imex_scheme)
        A_im, A_ex, b_im,b_ex,c_im,c_ex,imex_stages = load_imex_scheme(imex_scheme)
        C=c_im

    return A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages





################################################################
















######################   Define Norms  ###################

def Lp_norm(f,p):
    abs_f = np.abs(f)
  
    if p<1:
        print("L^p Error, p should be >=1")
        norm = None

    elif p==1:

        norm = np.mean(abs_f)
    elif p==np.inf:
        norm = np.max(abs_f)
    else:
        p = float(p)
        norm = np.mean(np.power(abs_f,p))
        norm = np.power(norm,1.0/p)
    return  norm




## Conserved Quantities   ##############







        



def kin_energy_1(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        u_ft = fft(u,u.shape)
        
        ##################################################### EP 1 #################################
        u_x = ifft(1j*xix*u_ft,u.shape)
        u_y = ifft(1j*xiy*u_ft,u.shape)
        E_k = np.mean(np.abs(np.conj(u_x)*u_x + np.conj(u_y)*u_y)) ## Kinetic energy





        return 4.0*lap_fac*lap_fac*E_k   ## If EP1 is used


def pot_energy_1(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        xi2 = xix*xix + xiy*xiy
        V_ft = -fft(np.square(np.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft = V_ft.at[0,0].set(0.0)

        ################################### EP 1#################################
        V_x = ifft(1j*xix*V_ft,u.shape)
        V_y = ifft(1j*xiy*V_ft,u.shape)
        
        E_p = np.mean(np.abs(np.conj(V_x)*V_x + np.conj(V_y)*V_y)) ## Potential energy

        ############################################# EP 2#################################

       
        
        return 2.0*kppa*lap_fac*E_p    #if E_p1 is used         
        
        




   
def kin_energy_2(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        u_ft = fft(u,u.shape)
    
        xi2 = xix*xix + xiy*xiy
        D2_u = ifft(-u_ft*(xi2),u.shape)
        E_k = -np.mean(np.conj(u)*D2_u).real  ## Kinetic energy




        return 4.0*lap_fac*lap_fac*E_k  ## If EP2 is used

def pot_energy_2(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        xi2 = xix*xix + xiy*xiy
        V_ft = -fft(np.square(np.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft = V_ft.at[0,0].set(0.0)


        V = ifft(V_ft,u.shape).real
        E_p = -np.mean(V*(np.square(np.abs(u))-mu))
              
        return 2.0*kppa*lap_fac*E_p   #if E_p2 is used
        






def Q_dummy(u,xi2,kppa,lap_fac,mu,Q):
     
     return Q


############################################################hjjdhkckdjhcdjhdjkcd
############### SP-2d euqtaion   ################################################

def run_example(dt,X,Xi,kppa,t_ini,T,L,imx,u_ini,energy_type=None,exact_soln_np=None,log_errs=False,lap_fac=1.0,num_plots=100,
                    data_dir=None):
    

    tmax = T
    t=t_ini

    m = X.shape[-1] 
    xi2 = np.zeros_like(Xi[-1])
    for i in range(len(Xi)):
        xi2 = xi2+np.square(Xi[i])
    lmbda = lap_fac*xi2

    nplt = np.floor((tmax/num_plots)/dt)
    nmax = int(round(tmax/dt))
    n=0
    print_cntr=0

    
   
   
    psi = 1.0*u_ini
    np.savez(data_dir+"/frame_initial",frame=psi)
    mass_ini = np.mean(np.abs(u_ini)**2)
    print("Initial mass is",mass_ini,L)
    mu = mass_ini#*L*L

    f = np.zeros_like(psi)
    f_t = np.zeros_like(psi)

    im_K = np.zeros([imx.s,m,m],dtype=np.complex128)
    ex_K = np.zeros([imx.s,m,m],dtype=np.complex128)



    


    dt_orig = 1.0*dt
    opt_succ=True
    frames = [u_ini,]
    mass_err_list = [0.0,]
    energy_err_list = [0.0,]
    energy_err_loc_list = [0.0,]
    t_list = [t_ini,]
    
    


    
    
    if energy_type=="E1":
        kin_energy = kin_energy_1
        pot_energy = pot_energy_1
    elif energy_type=="E2":
        kin_energy = kin_energy_2
        pot_energy = pot_energy_2

    
  

    
    


    

    def rhs_linear(uft,u,t,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu):
    
      
        xi2 = np.sum(np.square(Xi),axis=0)
    #rho = u[:m]; q = u[m:];
       
        v = np.zeros_like(u)
        q0hat = uft
     
        q0_x = ifft(-q0hat*(xi2),u.shape)
  
        rhs_q0 = lap_fac*1j*q0_x
    
        v = rhs_q0/np.power(t,1.5)
   
        return v

    def rhs_nonlinear(u,uft,t,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu):
    #Evaluate the nonlinear term
        xi2 = np.sum(np.square(Xi),axis=0)
        
        V_ft = -fft(np.square(np.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft = V_ft.at[0,0].set(0.0)
        #V_ft[0,0] = 0.0  ## Set mean value of potential to zero
        V = (ifft(V_ft,V_ft.shape)).real
        q_rhs = -1j*kppa*V*u/np.sqrt(t)
       
      
        return q_rhs


            

    
    
  
    
    #print("Ini invariant", inv_ini)
    print_list = [0.023,0.033,0.088,0.00001,0.00002]
    compare_print_cntr=0

    


    def do_fft(dt,f,s_cntr,lmda,im_A):
        '''This function does the fft on summed up contributions of all previous stages and then multiplies the FT vector f_t
            by needed factors to do implicit calculations and then does inverse fft. Arguments: s_cntr-> no of stage working on,lmda->consists of necessary xi factors
            from the format (1+i*dt*im_A[s][s]*lmda)*f_t = ft(rhs)'''
        f_t = fft(f,f.shape)

        #(1+i*dt*a[s][s]*lmda)*f_t = ft(rhs)
        f_t = f_t/(1.0+1j*dt*lmda*im_A[s_cntr][s_cntr])
        f = ifft(f_t,f_t.shape)
        #print("f shape",self.f_t.shape,self.f.shape)
        return f


    def update_stage_sum(dt,psi,im_A,ex_A,im_K,ex_K,s_cntr):
        '''This function sums up contriutions from all the previous substages to calc. rhs, on which we do fft in do_fft() function, for implicit calc.'''
        
        #print("Shapes",ex_A[s_cntr,:].shape,ex_K.shape)
        #f = psi + dt*(np.einsum("i,ij->j", ex_A[s_cntr,:], ex_K) + np.einsum("i,ij->j", im_A[s_cntr,:], im_K))
        f = psi + dt*(np.einsum("i,ijk->jk", ex_A[s_cntr,:], ex_K) + np.einsum("i,ijk->jk", im_A[s_cntr,:], im_K))
        return f




           
    def update_K(dt,t,f,f_t,im_C,ex_C,im_K,ex_K,s_cntr):
        '''This function stores the contribution from particular stage into K vectors'''
        #print("sncntr ",s_cntr)
       # print("psi shape",self.psi.shape,"f shape",self.f.shape)
        
        ex_t = t+ex_C[s_cntr]*dt
        im_t = t+im_C[s_cntr]*dt
        c_ex_K = rhs_nonlinear(f,f_t,ex_t)
        c_im_K = rhs_linear(f_t,f,im_t)
        #print("shape check ",im_K[s_cntr].shape,ex_K[s_cntr].shape,c_im_K.shape,c_ex_K.shape)

        return  im_K.at[s_cntr].set(c_im_K),ex_K.at[s_cntr].set(c_ex_K)
            


            
            
        
    def sum_contributions(dt,t,psi,s,im_B,ex_B,emb_B,im_K,ex_K):
        '''This function sums up the final contributions from all the stages weighted by respective coefficients(B(or b) from Butcher Tableau)'''
        term = np.zeros_like(psi)
        term_emb = np.zeros_like(psi)

        for i in range(s):
            term+=(dt*ex_B[i]*ex_K[i]+ dt*im_B[i]*im_K[i])
            #term_emb+=(dt*emb_B[i]*ex_K[i]+ dt*emb_B[i]*im_K[i])

        

        return term


    @jax.jit
    def Q_integrand(t,psi,Xi=Xi,kppa= kppa,lap_fac=lap_fac,mu=mu):
        Q_val = -pot_energy(psi,Xi,kppa,lap_fac,mu)
        return Q_val
    
    @jax.jit
    def time_stepper(dt,t,n,psi,im_K,ex_K,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu,imx=imx,lmbda=lmbda):
        Q_int = 0.0
        for k in range(imx.s):
            f = update_stage_sum(dt,psi,imx.im_A,imx.ex_A,im_K,ex_K,k)
            im_t = t+imx.im_C[k]*dt
            lap_t = np.power(im_t,-1.5)
            f = do_fft(dt,f,k,lmbda*lap_t,imx.im_A)
            f_t = fft(f,f.shape)
            im_K,ex_K = update_K(dt,t,f,f_t,imx.im_C,imx.ex_C,im_K,ex_K,k)
            Q_int +=  imx.ex_B[k]*Q_integrand(im_t,f,Xi,kppa,lap_fac,mu)*dt
        term = sum_contributions(dt,t,psi,imx.s,imx.im_B,imx.ex_B,imx.emb_B,im_K,ex_K)

       

        return term, Q_int
    
    @jax.jit
    def mass(psi):
        return np.mean(np.abs(psi)**2)
    

    @jax.jit
    def calc_energy(t,psi,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu):

        Energy = kin_energy(psi,Xi,kppa,lap_fac,mu)-pot_energy(psi,Xi,kppa,lap_fac,mu)*t
        #Energy = pot_energy(psi,Xi,kppa,lap_fac,mu)
        return Energy
    
    #dEnergy_dt = grad(calc_energy,0)
   #jac_psi_dt = jax.jit(jax.jacfwd(time_stepper,0))
    #jac_E_dt = jax.jit(jax.jacrev(calc_energy,0))


   
    
    
    
    
    mass_ini = mass(psi)
    energy_ini = calc_energy(t,psi)
   
    Q_int_l = 0.0
    while (t<tmax):

        
   
        

        term,Q_int = time_stepper(dt,t,n,psi,im_K,ex_K)
        Q_int_l += Q_int

        

        
        energy_now = calc_energy(t,psi)
        mass_now = mass(psi)

        psi= psi + term
        t = t+dt

        energy_new = calc_energy(t,psi)
        mass_new = mass(psi)
        #print("En change",energy_new,energy_now,Q_int)
        
        if (t>=(tmax-dt)) and (t<tmax)  :
                dt = (tmax-t)

        
        if n%int(nplt)==0:
            print("t=",t,np.mean(np.abs(psi)))
            mass_relative_err = (mass_new - mass_ini)/mass_ini
            energy_err = (energy_new - energy_ini -Q_int_l)
            energy_err_loc = (energy_new - energy_now - Q_int)
            mass_err_list.append(mass_relative_err)
            energy_err_list.append(energy_err)
            energy_err_loc_list.append(energy_err_loc)
            print(t,dt,"Mass new:",mass_new,"Relative Mass diff:",mass_relative_err,"Energy diff:",energy_err)
            if data_dir is not None:
                np.savez(data_dir+"/frame_"+str(t)[:6],frame=psi)
            else:
                frames.append(psi)
            t_list.append(t)

            
      

             
        n=n+1

    print("t=",t,np.mean(np.abs(psi)))
    mass_relative_err = (mass_new - mass_ini)/mass_ini
    energy_err = (energy_new - energy_ini -Q_int_l)
    energy_err_loc = (energy_new - energy_now - Q_int)
    mass_err_list.append(mass_relative_err)
    energy_err_list.append(energy_err)
    energy_err_loc_list.append(energy_err_loc)
    print(t,dt,"Mass new:",mass_new,"Relative Mass diff:",mass_relative_err,"Energy diff:",energy_err)
    print("Total no of steps taken:",n)
    if data_dir is not None:
                np.savez(data_dir+"/frame_"+str(t)[:6],frame=psi)
    else:
                frames.append(psi)
    t_list.append(t)





    
    
    print("Final mass:",mass(psi),"at time t=",t)  
    return psi,mass_err_list,energy_err_list,energy_err_loc_list,t_list,frames







if __name__=="__main__":
    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

   
    imex_sch = str(sys.argv[1])
    energy_type= sys.argv[2]  ## 
    m = int(sys.argv[3])  ## Number of grid points
    dt = float(sys.argv[4])     ## Choose dt

    print("Command line args:")
    print("ImEx scheme:",imex_sch)
    print("Energy type:",energy_type)
    print("Number of grid points:",m)
    print("Time step dt:",dt)
    A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages = choose_imex(imex_sch)  
    

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex,emb_B=b_hat,im_C=C,ex_C=C)

    


    A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages = choose_imex(imex_sch)  
 

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex,emb_B=b_hat,im_C=C,ex_C=C)


    # List of tau values for which u wanna run simulation
    
    
    hbar_mass = 6e-5
    beta = 1.5
    epsilon = hbar_mass
    kppa = beta/(epsilon)  ## Coefficient in front of non-linear term
    lap_fac = epsilon/(2.0)  ## Coefficient in front of laplacian term
    
    #dt=1e-6     ## Choose dt
    t_ini = 0.01
    T = 0.088
    

   #m = 1024  ## Number of grid points
    xL = -0.5; xR = 0.5; L = xR-xL
    Lf = 1.0
    x_1d = np.arange(-m/2,m/2)*(L/m)
    x,y = np.meshgrid(x_1d,x_1d)

    xi = np.fft.fftfreq(m)*m*2*np.pi/L
    xix,xiy = np.meshgrid(xi,xi)

    xi2 = xix*xix + xiy*xiy

    X = np.stack([x,y],axis=0)
    Xi = np.stack([xix,xiy],axis=0)



    
    case="imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    if energy_type=="E1":
        save_dir = "./_data_sine_wave_E1/"
    elif energy_type=="E2":
        save_dir = "./_data_sine_wave_E2/"
    else:
        print("Energy type not recognized")
        exit()   
    #save_dir = "_test/"

    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    save_dir_case = save_dir+"/"+case
    if not(os.path.exists(save_dir_case)):
            os.makedirs(save_dir_case)
    #save_dir_case = "test_metal"
    
    ######### Initial Condition: Sine Wave Collapse
    if m==1024:
        data = numpy.loadtxt("./x_y_n_phi_1024.dat")
        data = numpy.reshape(data,(1024,1024,2))
        n_d = data[::,::,0]
        phi_d = data[::,::,1]
        print("n_d from file",n_d.max(),n_d.min(),phi_d.max(),phi_d.min())
        ###########################################################################
        #N = len(x)
        #### Argument to exponential 

        u_exp_arg = 1j*phi_d/hbar_mass
        u_ini = np.sqrt(n_d)*np.exp(u_exp_arg)
    elif m==512:
        data = numpy.loadtxt("./x_y_n_phi_512.dat")
        data = numpy.reshape(data,(512,512,2))
        n_d = data[::,::,0]
        phi_d = data[::,::,1]
        print("n_d from file",n_d.max(),n_d.min(),phi_d.max(),phi_d.min())
        ###########################################################################
        #N = len(x)
        #### Argument to exponential 

        u_exp_arg = 1j*phi_d/hbar_mass
        u_ini = np.sqrt(n_d)*np.exp(u_exp_arg)
    elif m==2048 or m==4096:
        data = numpy.loadtxt("./x_y_n_phi_4096.dat")
        data = numpy.reshape(data,(4096,4096,2))
        if m==2048:
            data = data[::2,::2,:]
        n_d = data[::,::,0]
        phi_d = data[::,::,1]
        print("n_d from file",n_d.max(),n_d.min(),phi_d.max(),phi_d.min())
        ###########################################################################
        #N = len(x)
        #### Argument to exponential 

        u_exp_arg = 1j*phi_d/hbar_mass
        u_ini = np.sqrt(n_d)*np.exp(u_exp_arg)
  
    

  

    print("Running solitons with scheme ",imex_sch)
   
    t0 = time.time()
    psi,mass_err_list,energy_err_list,energy_err_loc_list,t_list,frames = run_example(dt,X,Xi,kppa,t_ini,T,L,imx,u_ini,energy_type=energy_type,exact_soln_np=None,\
                                                            log_errs=True,lap_fac=lap_fac,num_plots=100,data_dir=save_dir_case)
    psi.block_until_ready()
    t1 = time.time()
    print("Time taken for simulation is ",t1-t0," seconds")
    
    case_dict={"scheme":imex_sch,"frame_list":frames,"t_list":t_list,\
                     "kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_list,"energy_err_l":energy_err_list,"energy_err_loc_l":energy_err_loc_list}
  
    with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
            pickle.dump(case_dict,f)

    mass_err = np.array(mass_err_list)
    energy_err = np.array(energy_err_list)
    energy_err_loc = np.array(energy_err_loc_list)

    np.savez(save_dir+"/mass_err",array=mass_err)
    np.savez(save_dir+"/energy_err"+case,array=energy_err)
    np.savez(save_dir+"/energy_err_loc"+case,array=energy_err_loc)
    print("Saved data to ",save_dir_case)
    print("#######################")
    print("                       ")
    


    
    











