import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import math

from jax import grad
import pickle
import os
import optimistix as optx

from jaxopt import Broyden
import jaxopt
import numpy as numpy
import MAS_library as MASL
import h5py
import time





import sys

cwd = "/".join(os.getcwd().split("/")[:-1])
print("CWD:",cwd)
sys.path.append(cwd)
sys.path.append(cwd+"/time_integrators")
from GPE import ImEx



fft = jnp.fft.fftn
ifft = jnp.fft.ifftn


##############################  CHoosing ImEX Scheme  

#Choose ImEx scheme

def choose_imex(imex_scheme="default"):
    from Biswas_Ketcheson_TimeIntegrators import ImEx_schemes
    from Biswas_Ketcheson_TimeIntegrators import load_imex_scheme

    if imex_scheme=="default":
        #print("Choosin default ImEx")
        A_ex    = jnp.array([[0,0,0],[5/6.,0,0],[11/24,11/24,0]])
        A_im = jnp.array([[2./11,0,0],[205/462.,2./11,0],[2033/4620,21/110,2/11]])
        b_ex = jnp.array([24/55.,1./5,4./11])
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
    abs_f = jnp.abs(f)
  
    if p<1:
        print("L^p Error, p should be >=1")
        norm = None

    elif p==1:

        norm = jnp.mean(abs_f)
    elif p==jnp.inf:
        norm = jnp.max(abs_f)
    else:
        p = float(p)
        norm = jnp.mean(jnp.power(abs_f,p))
        norm = jnp.power(norm,1.0/p)
    return  norm




## Conserved Quantities   ##############



def kin_energy_1(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        xiz = Xi[2]
        u_ft = fft(u,u.shape)
        
        ##################################################### EP 1 #################################
        u_x = ifft(1j*xix*u_ft,u.shape)
        u_y = ifft(1j*xiy*u_ft,u.shape)
        u_z = ifft(1j*xiz*u_ft,u.shape)
        E_k = jnp.mean(jnp.abs(jnp.conj(u_x)*u_x + jnp.conj(u_y)*u_y + jnp.conj(u_z)*u_z)) ## Kinetic energy





        return 4.0*lap_fac*lap_fac*E_k   ## If EP1 is used


def pot_energy_1(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        xiz = Xi[2]
        xi2 = xix*xix + xiy*xiy + xiz*xiz
        V_ft = -fft(jnp.square(jnp.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft = V_ft.at[0,0,0].set(0.0)

        ################################### EP 1#################################
        V_x = ifft(1j*xix*V_ft,u.shape)
        V_y = ifft(1j*xiy*V_ft,u.shape)
        V_z = ifft(1j*xiz*V_ft,u.shape)
        
        E_p = jnp.mean(jnp.abs(jnp.conj(V_x)*V_x + jnp.conj(V_y)*V_y + jnp.conj(V_z)*V_z)) ## Potential energy

        ############################################# EP 2#################################

       
        
        return 2.0*kppa*lap_fac*E_p    #if E_p1 is used         
        
        




   
def kin_energy_2(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        xiz = Xi[2]
        u_ft = fft(u,u.shape)
    
        xi2 = xix*xix + xiy*xiy + xiz*xiz
        D2_u = ifft(-u_ft*(xi2),u.shape)
        E_k = -jnp.mean(jnp.conj(u)*D2_u).real  ## Kinetic energy




        return 4.0*lap_fac*lap_fac*E_k  ## If EP2 is used

def pot_energy_2(u,Xi,kppa,lap_fac,mu,Q=None):
        xix = Xi[0]
        xiy = Xi[1]
        xiz = Xi[2]
        xi2 = xix*xix + xiy*xiy + xiz*xiz
        ############################### EP 2 #################################
        V_ft = -fft(jnp.square(jnp.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft = V_ft.at[0,0,0].set(0.0)


        V = ifft(V_ft,u.shape).real
        E_p = -jnp.mean(V*(jnp.square(jnp.abs(u))-mu))
              
        return 2.0*kppa*lap_fac*E_p   #if E_p2 is used
        






def Q_dummy(u,xi2,kppa,lap_fac,mu,Q):
     
     return Q

############################################################hjjdhkckdjhcdjhdjkcd
############### SP-2d euqtaion   ################################################
                

def run_example(dt,X,Xi,kppa,lap_fac,omega_m,t_ini,T,L,imx,u_ini,energy_type=None,exact_soln_np=None,log_errs=False,num_plots=100,p=3.0,
                    data_dir=None,opt_algo="optx_Newton",opt_options={"rtol":1e-12,"atol":1e-12,"throw":True,"cauchy_termination":False}):
    
    tmax = T
    t=t_ini

    m = X.shape[-1] 
    xi2 = jnp.zeros_like(Xi[-1])
    for i in range(len(Xi)):
        xi2 = xi2+jnp.square(Xi[i])
    lmbda = lap_fac*xi2

    nplt = jnp.floor((tmax/num_plots)/dt)
    nmax = int(round(tmax/dt))
    n=0
    print_cntr=0

    
   
   
    psi = 1.0*u_ini
    jnp.savez(data_dir+"/frame_initial",frame=psi)
    mass_ini = jnp.mean(jnp.abs(u_ini)**2)
    mu = mass_ini#*L*L

    f = jnp.zeros_like(psi)
    f_t = jnp.zeros_like(psi)

    im_K = jnp.zeros([imx.s,m,m,m],dtype=jnp.complex128)
    ex_K = jnp.zeros([imx.s,m,m,m],dtype=jnp.complex128)



    
    print("Initial mass is",mass_ini,L,"m",m,"dt",dt)


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

    
  

    

    def HbyH0(a,omega_m=omega_m):
        return jnp.sqrt(omega_m/(a**3) + (1.0-omega_m) )
    

    def rhs_linear(uft,u,t,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu):
    
      
        xi2 = jnp.sum(jnp.square(Xi),axis=0)
    #rho = u[:m]; q = u[m:];
       
        v = jnp.zeros_like(u)
        q0hat = uft
     
        q0_x = ifft(-q0hat*(xi2),u.shape)
  
        rhs_q0 = lap_fac*1j*q0_x
    
        pfac = 1.0/(HbyH0(t)*t*t*t)
        v = pfac*rhs_q0
   
        return v

    def rhs_nonlinear(u,uft,t,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu):
    #Evaluate the nonlinear term
        xi2 = jnp.sum(jnp.square(Xi),axis=0)
        
        V_ft = -fft(jnp.square(jnp.abs(u))-mu,u.shape)/(xi2+1e-14) ## Poisson eqn. in FT space
        V_ft = V_ft.at[0,0,0].set(0.0)
       
        V = (ifft(V_ft,V_ft.shape)).real
        qfac = 1.0/(HbyH0(t)*t*t)
        q_rhs = -1j*kppa*V*u*qfac
       
      
        return q_rhs


            

    
    
    
  
   


    def do_fft(dt,f,s_cntr,lmda,im_A):
        '''This function does the fft on summed up contributions of all previous stages and then multiplies the FT vector f_t
            by needed factors to do implicit calculations and then does inverse fft. Arguments: s_cntr-> no of stage working on,lmda->consists of necessary Xi factors
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
        #f = psi + dt*(jnp.einsum("i,ij->j", ex_A[s_cntr,:], ex_K) + jnp.einsum("i,ij->j", im_A[s_cntr,:], im_K))
        f = psi + dt*(jnp.einsum("i,ijkl->jkl", ex_A[s_cntr,:], ex_K) + jnp.einsum("i,ijkl->jkl", im_A[s_cntr,:], im_K))
        return f




           
    def update_K(dt,t,f,f_t,im_C,ex_C,im_K,ex_K,s_cntr):
        '''This function stores the contribution from particular stage into K vectors'''
        #print("sncntr ",s_cntr)
       # print("psi shape",self.psi.shape,"f shape",self.f.shape)
        
        ex_t = t+ex_C[s_cntr]*dt
        im_t = t+im_C[s_cntr]*dt
        c_ex_K = rhs_nonlinear(f,f_t,ex_t)
        c_im_K = rhs_linear(f_t,f,im_t)

        return  im_K.at[s_cntr].set(c_im_K),ex_K.at[s_cntr].set(c_ex_K)
            


            
            
        
    def sum_contributions(dt,t,psi,s,im_B,ex_B,im_K,ex_K):
        '''This function sums up the final contributions from all the stages weighted by respective coefficients(B(or b) from Butcher Tableau)'''
        term = jnp.zeros_like(psi)
        term_emb = jnp.zeros_like(psi)

        for i in range(s):
            term+=(dt*ex_B[i]*ex_K[i]+ dt*im_B[i]*im_K[i])
            #term_emb+=(dt*emb_B[i]*ex_K[i]+ dt*emb_B[i]*im_K[i])

        

        return term#jnp.stack([term,term_emb],axis=0)


    @jax.jit
    def Q_integrand(t,psi,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu):
        Q_val = -pot_energy(psi,Xi,kppa,lap_fac,mu)
        return Q_val

    @jax.jit
    def time_stepper(dt,t,n,psi,im_K,ex_K,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu,imx=imx,lmbda=lmbda,omega_m=omega_m):
        Q_int = 0.0
        for k in range(imx.s):
            f = update_stage_sum(dt,psi,imx.im_A,imx.ex_A,im_K,ex_K,k)
            im_t = t+imx.im_C[k]*dt
            lap_t = 1.0/(HbyH0(im_t)*im_t*im_t*im_t)
            f = do_fft(dt,f,k,lmbda*lap_t,imx.im_A)
            f_t = fft(f,f.shape)
            im_K,ex_K = update_K(dt,t,f,f_t,imx.im_C,imx.ex_C,im_K,ex_K,k)
            Q_int +=  imx.im_B[k]*Q_integrand(im_t,f,Xi,kppa,lap_fac,mu)*dt
        term = sum_contributions(dt,t,psi,imx.s,imx.im_B,imx.ex_B,im_K,ex_K)

       

        return term,Q_int
    

    @jax.jit
    def calc_energy(t,psi,Xi=Xi,kppa=kppa,lap_fac=lap_fac,mu=mu):

        Energy = kin_energy(psi,Xi,kppa,lap_fac,mu)-pot_energy(psi,Xi,kppa,lap_fac,mu)*t
        return Energy
    
    #dEnergy_dt = grad(calc_energy,0)
   #jac_psi_dt = jax.jit(jax.jacfwd(time_stepper,0))
    #jac_E_dt = jax.jit(jax.jacrev(calc_energy,0))


   
    
    @jax.jit
    def mass(psi):
        return jnp.mean(jnp.abs(psi)**2)
    
    


    
    def tobe_minimized(gamma,args):
        
        term_proj = args[0]

        psi = args[1]
        mass_old = args[2]
        energy_old = args[3]
        Q_int = args[4]
        t = args[5]
        dt = args[6]
        tn = t + dt*gamma
 

        psi_new = psi + term_proj*gamma
        mass_new = mass(psi_new)
        psi_new_proj = jnp.sqrt(mass_old/mass_new)*psi_new
        energy_new = calc_energy(tn,psi_new_proj)
        



        return jnp.array([energy_new-energy_old-Q_int*gamma]) #+ eqn
    def tobe_minimized_srscalar(gamma,args):
        
        term_proj = args[0]

        psi = args[1]
        mass_old = args[2]
        energy_old = args[3]
        Q_int = args[4]
        t = args[5]
        dt = args[6]
        tn = t + dt*gamma
 

        psi_new = psi + term_proj*gamma
        mass_new = mass(psi_new)
        psi_new_proj = jnp.sqrt(mass_old/mass_new)*psi_new
        energy_new = calc_energy(tn,psi_new_proj)
        



        return energy_new-energy_old-Q_int*gamma #+ eqn

    if opt_algo=="optx_Newton":
        rtol = opt_options["rtol"]
        atol = opt_options["atol"]
        throw = opt_options["throw"]
        ct = opt_options["cauchy_termination"]
        solver = optx.Newton(rtol=rtol, atol=atol,cauchy_termination=ct)
    elif opt_algo=="optx_IndirectLevenbergMarquardt":
        rtol = opt_options["rtol"]
        atol = opt_options["atol"]
        lambda_0 = opt_options["lambda_0"]
        throw = opt_options["throw"]
        solver = optx.IndirectLevenbergMarquardt(rtol=rtol, atol=atol,lambda_0 = lambda_0)#,root_finder=optx.Newton(rtol=1e-5, atol=1e-5))
    elif opt_algo=="jaxopt_scipy" or opt_algo=="jaxopt_broyden":
        tol = opt_options["tol"]
        solver = jaxopt.ScipyRootFinding(optimality_fun = tobe_minimized, method="hybr", tol=tol)
    elif opt_algo=="scipy_root_scalar":
        print("Using scipy root_scalar with secant method")
        from scipy.optimize import root_scalar
    elif opt_algo=="scipy_fsolve":
        print("Using scipy fsolve")
        from scipy.optimize import fsolve
        
    # solver = optx.BFGS(rtol=1e-8, atol=1e-8)
    #solver = optx.Chord(rtol=1e-14, atol=1e-14)
    #best_so_far = solver#optx.BestSoFarRootFinder(solver)
    mass_ini = mass(psi)
    energy_ini = calc_energy(t,psi)

    Q_int_l = 0.0
    fail_cntr = 0
    fail_cntr_step=0
    while (t<tmax):

        
   
        
        
        #print(n,t)
        if n==0:
            gamma = 1.0#jnp.array([0.0,0.0],dtype=jnp.float64)
            gamma_0 = 1.0
        # term,dterm_dt = gv(dt,t,n,psi,im_K,ex_K)

        # dterm_dt1 = dterm_dt[:m]
        # dterm_dt2 = dterm_dt[m:]
        energy_now = calc_energy(t,psi)
        mass_now = mass(psi)

        term,Q_int = time_stepper(dt,t,n,psi,im_K,ex_K)
        
        
        psi_new = psi + term
        mass_new = mass(psi_new)
        pri_proj = jnp.sqrt(mass_now/mass_new)*psi_new
        term_proj = pri_proj - psi

        
        
        
        #sol,info = brd.run(gamma, args=(term1,term2,psi,mass_ini,energy_ini))
        if opt_algo=="optx_Newton" or opt_algo=="optx_IndirectLevenbergMarquardt":
            sol = optx.root_find(tobe_minimized, solver, gamma, args=(term_proj,psi,mass_now,energy_now,Q_int,t,dt), max_steps=10,throw=throw)
            gamma = sol.value
        elif opt_algo=="jaxopt_scipy":
            
            sol = solver.run(gamma, args=(term_proj,psi,mass_now,energy_now,Q_int,t,dt))
            gamma = sol.params
        elif opt_algo=="jaxopt_broyden":
            brd = Broyden(fun =tobe_minimized,tol=1e-12,stop_if_linesearch_fails = True)
            sol = brd.run(gamma, args=(term_proj,psi,mass_now,energy_now,Q_int,t,dt))
            gamma = sol.params
        elif opt_algo=="scipy_root_scalar":
            
            #sol = root(tobe_minimized,gamma_0,args=(term1,term2,psi,mass_now,energy_now,t), method='hybr',tol=1e-14)
            sol = root_scalar(tobe_minimized_srscalar,args=((term_proj,psi,mass_now,energy_now,Q_int,t,dt),),x0 = gamma_0, method='secant',xtol=1e-14)
            
            gamma = sol.root
        elif opt_algo =="scipy_fsolve":
            
            sol = fsolve(tobe_minimized_srscalar,gamma_0,args=((term_proj,psi,mass_now,energy_now,Q_int,t,dt),),xtol=1e-14)
            gamma = sol[0]
             

             
             
        else:
            print("Unknown opt_algo ",opt_algo," Exiting")
            sys. exit()
        
 
        fval = tobe_minimized(gamma,(term_proj,psi,mass_now,energy_now,Q_int,t,dt))
        #print("gama=",gamma," func val:",fval)
        #print("sol",sol.value)#,info)
        if (jnp.sum(jnp.abs(fval))>1e-14 or gamma<0.0 or jnp.isnan(gamma) or gamma==0.0 or gamma<1e-8) and (T-t)>1e-10:
            print("Warning: Root finding in relaxation step did not converge, func value:",fval," gamma=",gamma)
            fail_cntr = fail_cntr+1
            if jnp.isclose(dt,dt_orig):
                 fail_cntr_step = fail_cntr_step+1
            gamma = 1.0#np.array([0.0,0.0],dtype = jnp.float64)
            opt_succ=False
            dt = dt*0.5
            print("Reducing dt to ",dt)
            if dt<1e-12:
                print("dt too small, exiting")
                fail_data = jnp.array([fail_cntr,fail_cntr_step,n])
                if data_dir is not None:
                    jnp.savez(data_dir+"/fail_data",data=fail_data)
                sys.exit()
        ##########################************************************
        # print("Solved status ",optx.RESULTS[sol.result])
        # print( " gamma=",sol.value,1.0+jnp.sum(sol.value))
        #print("func value:",tobe_minimized(sol.value,(term1,term2,psi,mass_ini,calc_energy(psi))))
        else:
            if dt<=1e-10:
                psi= psi + term_proj
                mass_new = mass(psi)
                psi = jnp.sqrt(mass_now/mass_new)*psi
                t = t+dt
            else:
                psi= psi + term_proj*gamma
                mass_new = mass(psi)
                psi = jnp.sqrt(mass_now/mass_new)*psi
                t = t+gamma*dt
                Q_int_l += Q_int*gamma
                dt = dt_orig
            if (t>=(tmax-dt)) and (t<tmax)  :
                dt = (tmax-t)
                print("Adjusting final dt to ",dt)

        mass_new = mass(psi)
        energy_new = calc_energy(t,psi)
        if n%int(nplt)==0:
            print("t=",t,"gamma=",gamma)
            mass_relative_err = (mass_new - mass_ini)/mass_ini
            energy_err = (energy_new - energy_ini-Q_int_l)#/energy_ini
            energy_err_loc = (energy_new - energy_now-gamma*Q_int)#/energy_ini
            mass_err_list.append(mass_relative_err)
            energy_err_list.append(energy_err)
            energy_err_loc_list.append(energy_err_loc)
            t_list.append(t)
            print(t,dt,"Mass new:",mass_new,"Relative Mass diff:",mass_relative_err,"Energy diff:",energy_err)
            if data_dir is not None:
                jnp.savez(data_dir+"/frame_"+str(t)[:6],frame=psi)
            else:
                frames.append(psi)

            if log_errs:
                if exact_soln_np is not None:
                    u_exact = exact_soln_np(t,x,kppa)
                    err_u = psi - u_exact
                    err_now = jnp.array([[Lp_norm(err_u,jnp.inf),Lp_norm(err_u,1.0),Lp_norm(err_u,2.0)]])
                    err_array = jnp.concatenate((err_array, err_now),axis=0)
      

             
        n=n+1
    print("t=",t,"gamma=",gamma)
    mass_relative_err = (mass_new - mass_ini)/mass_ini
    energy_err = (energy_new - energy_ini-Q_int_l)#/energy_ini
    energy_err_loc = (energy_new - energy_now-gamma*Q_int)#/energy_ini
    mass_err_list.append(mass_relative_err)
    energy_err_list.append(energy_err)
    energy_err_loc_list.append(energy_err_loc)
    t_list.append(t)
    print(t,dt,"Mass new:",mass_new,"Relative Mass diff:",mass_relative_err,"Energy diff:",energy_err)
    fail_data = jnp.array([fail_cntr,fail_cntr_step,n])
    print("Fail data:",fail_data)
    if data_dir is not None:
                jnp.savez(data_dir+"/frame_"+str(t)[:6],frame=psi)
                jnp.savez(data_dir+"/fail_data",data=fail_data)
    else:
                frames.append(psi)

    
    
    print("Final mass:",mass(psi),"at time t=",t)  
    return psi,mass_err_list,energy_err_list,energy_err_loc_list,t_list,frames









if __name__=="__main__":

    #print("Yes",cwd)


    #########       Setup and Run Numerical Experiment

   
    imex_sch = str(sys.argv[1])
    energy_type= sys.argv[2]  ## 
    m = int(sys.argv[3])  ## Number of grid points
    dt = float(sys.argv[4])     ## Choose dt
    vapower = float(sys.argv[5])  ## v =   a^{vapower}*v_snap where v is the velocity used to calculate initial wavefunction from snapshot data
    rand_ini = sys.argv[6]

    print("Command line args:")
    print("ImEx scheme:",imex_sch)
    print("Energy type:",energy_type)
    print("Number of grid points:",m)
    print("Time step dt:",dt)
    print("Velocity a power:",vapower)

    if len(sys.argv)>7:
        opt_algo = sys.argv[7]  ## Optimization algorithm
        opt_options = dict(eval(sys.argv[8]))  ## Optimization options
        print("Using optimization algo:",opt_algo)
        print("With options:",opt_options)
    else:
        opt_algo = "optx_Newton"
        opt_options = {"rtol":1e-4,"atol":1e-14,"throw":False,"cauchy_termination":True}
    print("Command line args:",imex_sch,energy_type,opt_algo,opt_options)
    


    A_im,A_ex,C,b_im,b_ex,b_hat,imex_stages = choose_imex(imex_sch)  
 

    # Initialize imex table for simulation
    imx = ImEx(imex_stages,A_im,A_ex,b_im,b_ex,emb_B=b_hat,im_C=C,ex_C=C)
    ##Cosmo parameters
    omega_m = 0.3  ## Matter density parameter


    # List of tau values for which u wanna run simulation
    hbox = 6.582
    cbox = 3.0
    pcbox  = 3.0857
    h0 = 0.7
    mbox = 0.175
    beta = 1.5*omega_m
    
    hbym = h0*((hbox*cbox**2)/(mbox*pcbox))*1e-3
    hbymH0 = (hbym)*0.01
    epsilon = hbymH0

    kppa = beta/(epsilon)  ## Coefficient in front of non-linear term
    lap_fac = epsilon/(2.0)  ## Coefficient in front of laplacian term
    print("##########################")
    print("Simulation parameters:")
    print("kappa is ",kppa)
    print("Laplacian factor is ",lap_fac)   
    print("hbym is ",hbym)
    print("hbymH0 is ",hbymH0)
    print("m_box is ",mbox)
    print("##########################")
    
    #dt=1e-6     ## Choose dt
    t_ini = 0.0078125
    T = 1.0
    

   #m = 1024  ## Number of grid points
    xL = 0.0; xR = 1.0; L = xR-xL
    Lf = 1.0
    x_1d = jnp.arange(0.0,m)*(L/m)
    x,y,z = jnp.meshgrid(x_1d,x_1d,x_1d,indexing='ij')

    xi = jnp.fft.fftfreq(m)*m*2*jnp.pi/L
    xix,xiy,xiz = jnp.meshgrid(xi,xi,xi,indexing='ij')

    xi2 = xix*xix + xiy*xiy + xiz*xiz

    X = jnp.stack([x,y,z],axis=0)
    Xi = jnp.stack([xix,xiy,xiz],axis=0)



    
    case="imex_"+imex_sch+"_"+str(m)+"_"+str(dt)
    if energy_type=="E1":
        save_dir = "./_dataij_hunit_TSC_cosmo_I_projRlx_E1_v"
    elif energy_type=="E2":
        save_dir = "./_dataij_hunit_TSC_cosmo_I_projRlx_E2_v"
    else:
        print("Energy type not recognized")
        exit()   
    save_dir = save_dir+str(vapower)+"_"+rand_ini.split("_")[1]+rand_ini.split("_")[2]+"/"
    #save_dir = "_test/"

    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    save_dir_case = save_dir+"/"+case
    if not(os.path.exists(save_dir_case)):
            os.makedirs(save_dir_case)
    #save_dir_case = "test_metal"
    
    ######### Initial Condition: Sine Wave Collapse

    initial_data_file = rand_ini
    with h5py.File(initial_data_file, 'r') as f:
        config = f['Config']
        header = f['Header']
        parameters = f['Parameters']
        partType1 = f['PartType1']
        part_coords = numpy.array(partType1['Coordinates'])
        part_vels = numpy.array(partType1['Velocities'])
    delta = numpy.zeros((m,m,m), dtype=numpy.float32)
    v_x = numpy.zeros((m,m,m), dtype=numpy.float32)
    v_y = numpy.zeros((m,m,m), dtype=numpy.float32)
    v_z = numpy.zeros((m,m,m), dtype=numpy.float32)
    pos = part_coords#

    MAS     = 'TSC' # mass assignment scheme
    # construct 3D density field
    MASL.MA(pos, delta, L, MAS)
    MASL.MA(pos, v_x, L, MAS,W = part_vels[:,0])
    MASL.MA(pos, v_y, L, MAS,W=part_vels[:,1])
    MASL.MA(pos, v_z, L, MAS,W=part_vels[:,2])

    delta /= jnp.mean(delta, dtype=jnp.float64);  delta -= 1.0
    delta = jnp.array(delta, dtype=jnp.float64)

    v_x = jnp.power(t_ini,vapower)*jnp.array(v_x, dtype=jnp.float64)
    v_y = jnp.power(t_ini,vapower)*jnp.array(v_y, dtype=jnp.float64)
    v_z = jnp.power(t_ini,vapower)*jnp.array(v_z, dtype=jnp.float64)

    n_d = delta+1.0



    dx_vx = (1.0/hbym)*ifft(1j*xix*fft(v_x,v_x.shape),v_x.shape).real
    dy_vy = (1.0/hbym)*ifft(1j*xiy*fft(v_y,v_y.shape),v_y.shape).real
    dz_vz = (1.0/hbym)*ifft(1j*xiz*fft(v_z,v_z.shape),v_z.shape).real

    div_v = dx_vx + dy_vy + dz_vz
    alpha = -ifft(fft(div_v,div_v.shape)/(xi2+1e-14),div_v.shape).real
    alpha = alpha.at[0,0,0].set(0.0)
    u_exp_arg= alpha*1j
    u_ini = jnp.sqrt(n_d)*jnp.exp(u_exp_arg)
    
  
    

  

    print("Running with scheme ",imex_sch)
   
    t0 = time.time()
    psi,mass_err_list,energy_err_list,energy_err_loc_list,t_list,frames = run_example(dt,X,Xi,kppa,lap_fac,omega_m,t_ini,T,L,imx,u_ini,energy_type=energy_type,exact_soln_np=None,\
                                                            log_errs=True,num_plots=10,\
                                                            data_dir=save_dir_case,opt_algo=opt_algo,opt_options=opt_options)
    psi.block_until_ready()
    t1 = time.time()
    print("Time taken for simulation is ",t1-t0," seconds")
    
    case_dict={"scheme":imex_sch,"frame_list":frames,"t_list":t_list,\
                     "kappa":kppa,"dt":dt,"m":m,"mass_err_l":mass_err_list,"energy_err_l":energy_err_list,"energy_err_loc_l":energy_err_loc_list,\
                        "mass_fdm":mbox,"hbar_by_m":hbym,"omega_m":omega_m,"energy_type":energy_type,"vapower":vapower}
  
    with open(save_dir_case+"/case_dict.pkl", 'wb') as f:
            pickle.dump(case_dict,f)

    print("#######################")
    print("                       ")

    
    











