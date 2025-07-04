### Calculation of finite w,q conductivity with generic small-angle scattering
### Jonathan Curtis 
### 06/12/25

import numpy as np

from scipy import integrate as intg

import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr

rng = np.random.default_rng()


############################
### Scattering potential ###
############################

### This gives the impurity scattering form factor u(2kF |sintheta|) with form
### u propto exp(-beta^2 sin^2 theta)
def u_exp(theta,beta):
	f = lambda theta : np.exp(beta * np.cos(theta) )
	norm = intg.quad(f,0.,2.*np.pi)[0]/(2.*np.pi)
	return f(theta)/norm

### This computse the transport time correction given a scattering potential 
def calc_tr(u):
	f = lambda x : u(x)*np.cos(x)
	return 1. - intg.quad(f,0.,2.*np.pi)[0]/(2.*np.pi)

#######################
### Bethe Saltpeter ###
#######################

### This function solves the BS equation for a particular frequency and momentum 
### w = Frequency is in units of scattering rate omega/gamma 
### q = Momentum is in units of inverse scattering length q vF/gamma. We take this along the x axis 
### Scattering rate is single-particle scattering rate not transport time 
### u = scattering potential as a function of angles 
### thetas is an array of angles we compute for 
def solve_BSE(w,q,u,nthetas):
	thetas = np.linspace(0.,2.*np.pi,nthetas,endpoint=False) ### Grid of angles we compute the vertex function on 

	### We want vector functions which means we have 2 x nthetas size arrays to find 
	### Will be RHS of BSE 
	kF = np.concatenate((np.cos(thetas),np.sin(thetas)),dtype=complex) 

	### Meshgrid
	thetas_grid_1, thetas_grid_2 = np.meshgrid(thetas,thetas,indexing='ij')
	u_matrix = u(thetas_grid_2-thetas_grid_1) ### nthetas x nthetas 

	### This is an array of size nthetas 
	#diffuson = 1./(1. - 1.j*(w +  q[0]*np.cos(thetas) + q[1]*np.sin(thetas) ) ) ### This is a function of angles only 
	diffuson = 1./(1. - 1.j*(w +  q*np.cos(thetas) ) ) ### This is a function of cosine(angle) only if we can take q || x

	### Now we contruct the integral equation as a matrix for a single component
	matrix = np.eye(nthetas,dtype=complex) - 1./float(nthetas)*u_matrix*diffuson

	matrix_doubled = np.kron(np.eye(2,dtype=complex),matrix) 

	vertex = np.linalg.solve(matrix_doubled,kF)

	return thetas,vertex

### This should now compute the conductivity given the vertex correction 
def conductivity(w,q,u,nthetas):
	### First we compute the vertex correction 
	thetas,vertex = solve_BSE(w,q,u,nthetas)

	### Next we compute the trasnport time correction 
	gamma_tr = calc_tr(u)

	### Now we construct the diffuson again 
	#diffuson = 1./(1. - 1.j*(w +  q[0]*np.cos(thetas) + q[1]*np.sin(thetas) ) )
	diffuson = 1./(1. - 1.j*(w +  q*np.cos(thetas) ) )

	### Now we compute the four components 
	### First we unpack the vertex into the components 
	vertex_x = vertex[:nthetas]
	vertex_y = vertex[nthetas:]

	sigma = np.zeros((2,2),dtype=complex)

	sigma[0,0] = np.mean(vertex_x*diffuson*np.cos(thetas))
	sigma[1,0] = np.mean(vertex_y*diffuson*np.cos(thetas))
	sigma[0,1] = np.mean(vertex_x*diffuson*np.sin(thetas))
	sigma[1,1] = np.mean(vertex_y*diffuson*np.sin(thetas))

	return 2.*gamma_tr*sigma ### This is the normalized vertex corrected conductivity 

# ### This method computes the conductivity for an array of frequencies and momenta 
# def sigma_perp_tensor(wvs,qvs,u,nthetas):
# 	tensor_shape = wvs.shape 
# 	sigmavs = np.zeros_like(wvs)

# 	def sigma_perp_vec(w,q):
# 		sigma = conductivity(w,q,u,nthetas)
# 		return np.real(sigma[1,1])

# 	sigmavs = sigma_perp_vec(w,q)

# 	return sigmavs 

### This method computes the frequency and distance dependence of the flux noise for a given temperature and scattering potential 
def flux_noise(ws,zs,T,u,nthetas=100,nqs=1000,qmax=10.):
	### First we generate a momentum meshgrid in order to tensorize the conductivity integral 
	qs = np.linspace(0.,qmax,nqs) ### This is the array of points for our integral

	z_grid , w_grid , q_grid = np.meshgrid(zs,ws,qs,indexing='ij')
	sigma_grid = np.zeros_like(z_grid)

	for i in range(len(ws)):
		for j in range(len(qs)):
			sigma = conductivity(ws[i],qs[j],u,nthetas)

			sigma_grid[:,i,j] = np.ones_like(zs)*np.real(sigma[1,1])

	dq = qs[1]-qs[0]

	filterfunction = dq*np.exp(-2.*z_grid*q_grid)

	sigma_integral = np.sum(filterfunction *sigma_grid,axis=(-1)) ### Should have shape of z x w grid 

	alpha = calc_tr(u) ### transport time correction factor 

	### Now we multiply by FDT factors 
	return sigma_integral *alpha**2 * w_grid[:,:,0]/np.tanh(0.5*w_grid[:,:,0]/T)


















if __name__ == "__main__":
	main()



