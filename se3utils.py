"""
Utils for computations on the Lie Group SE(3)
Note: This is probably the most _cryptic_ file of the lot. Comments are sparse and the 
code is compact, mainly because I'm writing these functions for the Nth time.
"""

import numpy as np


epsil = 1e-6

# SO(3) hat operator
def SO3_hat(omega):
	
	omega_hat = np.zeros((3,3))
	omega_hat[0][1] = -omega[2]
	omega_hat[1][0] = omega[2]
	omega_hat[0][2] = omega[1]
	omega_hat[2][0] = -omega[1]
	omega_hat[1][2] = -omega[0]
	omega_hat[2][1] = omega[0]
	
	return omega_hat
	

# SE(3) exponential map
def SE3_Exp(xi):
	
	u = xi[:3]
	omega = xi[3:]
	omega_hat = SO3_hat(omega)
	
	theta = np.linalg.norm(omega)
	
	if np.linalg.norm(omega) < epsil:
		R = np.eye(3) + omega_hat
		V = np.eye(3) + omega_hat
	else:
		s = np.sin(theta)
		c = np.cos(theta)
		omega_hat_sq = np.dot(omega_hat, omega_hat)
		theta_sq = theta * theta
		A = s / theta
		B = (1 - c) / (theta_sq)
		C = (1 - A) / (theta_sq)
		R = np.eye(3) + A * omega_hat + B * omega_hat_sq
		V = np.eye(3) + B * omega_hat + C * omega_hat_sq
	t = np.dot(V, u.reshape(3,1))
	lastrow = np.zeros((1,4))
	lastrow[0][3] = 1.
	return np.concatenate((np.concatenate((R, t), axis = 1), lastrow), axis = 0)


# Functions to help compute the left jacobian for SE(3)
# See Tim Barfoot's book http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf

# SE(3) hat operator
def SE3_hat(xi):
	"""
	Takes in a 6 x 1 vector of SE(3) exponential coordinates and constructs a 4 x 4 tangent vector xi_hat
	in the Lie algebra se(3)
	"""
	
	v = xi[:3]
	omega = xi[3:]
	xi_hat = np.zeros((4,4)).astype(np.float32)
	xi_hat[0:3,0:3] = SO3_hat(omega)
	xi_hat[0:3,3] = v
	
	return xi_hat
