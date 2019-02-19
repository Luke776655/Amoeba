#!/usr/bin/python3

'''
Python implementation of the Nelder-Mead optimization algorithm (amoeba).
In this implementation algorithm is looking for the energetic minimum of the arrows system.
When all of the arrows are parallel, the system is in the global minimum.
GNU GPL license v3

Core algorithm code in line with
Numerical Receipes
The Art of Scientific Computing
Third Edition
W.H.Press, S.A.Teukolsky,
W.T Vetterling, B.P.Flannery

Łukasz Radziński
lukasz.radzinski _at_ gmail _dot_ com
'''

from scipy.optimize import rosen
import random as rd
import numpy as np
import sys

n = 10 #dimention of side of pseudo-square table
delta = 1 #displacement
ftol = 1e-4 #fractional convergence tolerance
nmax = 500000 #maximum allowed number of function evaluations
tiny = 1e-7 #tiny value preventing from dividing by 0

'''
def func(A):
	#simple multidimentional parabola function for testing optimisation
	#global minimum in 0
	s = 0.0
	for i in A:
		s += i*i
	return s
'''

'''
def func(A):
	#Rosen function for testing optimisation
	#global minimum in 0
	return rosen(A)	
'''

def func(A):
	'''
	potential energy function of the arrows system for testing optimisation
	when the neighbour arrow is parallel: E = 0
	when the neighbour arrow is orthogonal: E = 1
	when the neighbour arrow is antiparallel: E = 2
	global E is the sum of all arrows energy
	global energetic minimum is when all of the arrows are parallel, then global E = 0
	'''
	s = 0
	for i in range(n*n):
		if(i>=n): #top boundary
			s = s-np.cos(abs(A[i]-A[i-n]))+1
		if(i<n*n-n): #bottom boundary
			s = s-np.cos(abs(A[i]-A[i+n]))+1
		if(i%n!=0): #left boundary
			s = s-np.cos(abs(A[i]-A[i-1]))+1
		if(i%n!=(n-1)): #right boundary
			s = s-np.cos(abs(A[i]-A[i+1]))+1
	return s

def create_random_table(n):
	'''
	creating starting point of the system:
	n*n-dimentional table with random values
	values range: [0, 2*pi)
	'''
	tab = []
	for i in range(n*n):
		tab.append(rd.uniform(0, 2*np.pi))
	return tab

def create_table(n):
	'''
	creating starting point of the system:
	n*n-dimentional table with ordered values
	values range: [0, 2*pi)
	'''
	tab = []
	for i in range(n*n):
		tab.append(0 + i%(2*np.pi))
	return tab

def print_table(tab, n):
	'''
	printing values of the table
	'''
	for i in range(n*n):
		if(i%n==0):
			print("")
		if(tab[i]>=0):
			print(" ", end="")
		if(tab[i]<10 and tab[i]>-10):
			print(" ", end="")
		print(format(tab[i], ".2f"), end=" ")
	print("\n")

def print_arrow(s):
	'''
	printing value as an arrow with proper slope
	'''
	if(s<(np.pi/8+0*np.pi/4) or s>=(np.pi/8+7*np.pi/4)):
		print("↑", end=" ")
	elif(s>=(np.pi/8+0*np.pi/4) and s<(np.pi/8+1*np.pi/4)):
		print("↗", end=" ")
	elif(s>=(np.pi/8+1*np.pi/4) and s<(np.pi/8+2*np.pi/4)):
		print("→", end=" ")
	elif(s>=(np.pi/8+2*np.pi/4) and s<(np.pi/8+3*np.pi/4)):
		print("↘", end=" ")
	elif(s>=(np.pi/8+3*np.pi/4) and s<(np.pi/8+4*np.pi/4)):
		print("↓", end=" ")
	elif(s>=(np.pi/8+4*np.pi/4) and s<(np.pi/8+5*np.pi/4)):
		print("↙", end=" ")
	elif(s>=(np.pi/8+5*np.pi/4) and s<(np.pi/8+6*np.pi/4)):
		print("←", end=" ")
	elif(s>=(np.pi/8+6*np.pi/4) and s<(np.pi/8+7*np.pi/4)):
		print("↖", end=" ")
	else:
		print("x", end=" ")

def print_arrow_table(tab, n):
	'''
	printing values of the table as arrows with proper slope
	'''
	for i in range(n*n):
		if(i%n==0):
			print("")
		print_arrow(tab[i]%(2*np.pi))
	print("\n")

def print_result(y, p, ndim, ilo):
	'''
	printing result of optimisation
	'''
	z = 0
	k = y
	q = p

	z = k[0]
	k[0] = k[ilo]
	k[ilo] = z
	pmin = []
	for i in range(ndim):
		z = q[0][i]
		q[0][i] = q[ilo][i]
		q[ilo][i] = z
		pmin.append(q[0][i])
	fmin=k[0];
	print_table(pmin, n)
	print_arrow_table(pmin, n)
	print("Energy of the system", fmin)

def get_psum(p, ndim, mpts):
	'''
	counting partial sum
	'''
	psum = []
	for j in range(ndim):
		sum = 0.0
		for i in range(mpts):
			sum += p[i][j]
		psum.append(sum)
	return psum

def amotry(p, y, psum, ihi, fac):
	'''
	extrapolation by a factor fac through the face of the simplex across from the high point
	replacing the high point if the new point is better
	'''
	ptry = []
	fac1 = (1.0-fac)/ndim
	fac2 = fac1-fac
	for j in range(ndim):
		ptry.append(psum[j]*fac1-p[ihi][j]*fac2)
	ytry = func(ptry)
	if (ytry<y[ihi]):
		y[ihi]=ytry
		for j in range(ndim):
			psum[j] += ptry[j]-p[ihi][j]
			p[ihi][j]=ptry[j]
	return ytry

#creating point table as starting point in the system
point = create_random_table(n)
print_table(point, n)
print_arrow_table(point, n)
print(func(point))

#creating delta values table
delta_tab = []
for i in range(len(point)):
	delta_tab.append(delta)

#adding delta values to the point table with extended dimention as p simplex
p = []
ndim = len(point)
for i in range(ndim+1):
	k=[]
	for j in range(ndim):
		k.append(point[j])
	p.append(k)
	if(i!=0):
		p[i][i-1]+=delta_tab[i-1]

#getting y table of solutions
y = []
mpts = len(p) #number of rows
ndim = len(p[0]) #number of columns
for i in range(mpts):
	x = []
	for j in range(ndim):
		x.append(p[i][j])
	y.append(func(x))

#parameters for iterating
nfunc = 0
psum = get_psum(p, ndim, mpts)
it = 0

#algorithm working in the iterative mode
while(True):
	ilo = 0
	if(y[0]>y[1]):
		inhi = 1
		ihi = 0
	else:
		inhi = 0
		ihi = 1

	for i in range(mpts):
		if(y[i]<=y[ilo]):
			ilo = i
		if(y[i]>y[ihi]):
			inhi = ihi
			ihi = i
		elif(y[i] > y[inhi] and i != ihi):
			inhi = i
	
	rtol=2.0*abs(y[ihi]-y[ilo])/(abs(y[ihi])+abs(y[ilo])+tiny)

	if(rtol < ftol):
		print("\nIteration", it)
		print_result(y, p, ndim, ilo)
		print("Optimisation succeeded")
		break

	if (nfunc >= nmax):
		print("\nIteration", it)
		print_result(y, p, ndim, ilo)
		print("Maximal number of iterations exceeded")
		break

	nfunc += 2
	 
	ytry = amotry(p, y, psum, ihi, -1.0) #simplex reflection
	if(ytry<=y[ilo]):
		ytry = amotry(p, y, psum, ihi, 2.0) #simplex extrapolation
	elif(ytry >= y[inhi]):
		ysave = y[ihi]
		ytry = amotry(p, y, psum, ihi, 0.5) #simplex contraction
		if(ytry>=ysave):
			for i in range(mpts):
				if(i!=ilo):
					for j in range(ndim):
						p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j])
					y[i]=func(psum)
			nfunc += ndim;
			psum = get_psum(p, ndim, mpts)
	else:
		nfunc = nfunc-1

	if(it%100 == 0):
		print("\nIteration", it)
		print_result(y, p, ndim, ilo)

	it += 1



