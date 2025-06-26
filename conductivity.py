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
### q = Momentum is in units of inverse scattering length q vF/gamma
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
	diffuson = 1./(1. - 1.j*(w +  q[0]*np.cos(thetas) + q[1]*np.sin(thetas) ) ) ### This is a function of angles only 

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
	diffuson = 1./(1. - 1.j*(w +  q[0]*np.cos(thetas) + q[1]*np.sin(thetas) ) )

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














if __name__ == "__main__":
	main()



