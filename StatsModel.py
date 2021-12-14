# MRep Manifold 
import manifolds
import numpy as np

import pylab 

from random import shuffle 

import itertools


# Stats Model
import statsmodels.api as sm

import matplotlib.pyplot as plt


#############################################################
##########  Generalized Gaussian Noise Generation  ##########
#############################################################

def GaussianNoisePerturbation( mu_0, sigma ):
	if mu_0.Type == "Sphere":
		return GaussianNoisePerturbation_Sphere( mu_0, sigma )
	# elif dataList[ 0 ].Type == "PositiveReal":
	# 	return GaussianNoisePerturbation_PosReal( mu_0, sigma )
	# elif dataList[ 0 ].Type == "Euclidean":
	# 	return GaussianNoisePerturbation_Euclidean( mu_0, sigma )
	# elif dataList[ 0 ].Type == "CMRep":
	# 	return GaussianNoisePerturbation_CMRep( mu_0, sigma )
	# elif dataList[ 0 ].Type == "CMRep_Abstract":
	# 	return GaussianNoisePerturbation_CMRep_Abstract( mu_0, sigma )

	# elif dataList[ 0 ].Type == "MRep":
	# 	return FrechetMean_MRep( dataList, maxIter, tol )
	else:
		print( "Manifold type is not known" )
		return -1

def GaussianNoisePerturbation_Sphere( mu_0, sigma ):
	nDimManifold = mu_0.nDim

	# Generate a random Gaussians with polar Box-Muller Method
	rand_pt = np.zeros( nDimManifold ).tolist()

	for i in range( nDimManifold ):
		r2 = 0
		x = 0
		y = 0

		while( r2 > 1.0 or r2 == 0 ):
			x = ( 2.0 * np.random.rand() - 1.0 )
			y = ( 2.0 * np.random.rand() - 1.0 )
			r2 = x * x + y * y 

		gen_rand_no = sigma * y * np.sqrt( -2.0 * np.log( r2 ) / r2 )
		rand_pt[ i ] = gen_rand_no
	# print( rand_pt )

	# Set Random Vector to Tangent Vector - ListToTangent
	rand_tVec = manifolds.sphere_tVec( nDimManifold )
	rand_tVec.SetTangentVector( rand_pt )


	# Projected Tangent to Mean Point
	rand_tVec_projected = mu_0.ProjectTangent( mu_0, rand_tVec )

	# Perturbed point at time_pt 
	pt_perturbed = mu_0.ExponentialMap( rand_tVec_projected )

	return pt_perturbed	

######################################
##########  Intrinsic Mean  ##########
######################################
def FrechetMean( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	if dataList[ 0 ].Type == "Sphere":
		return FrechetMean_Sphere( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "PositiveReal":
		return FrechetMean_PosReal( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "Euclidean":
		return FrechetMean_Euclidean( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "CMRep":
		return FrechetMean_CMRep( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "CMRep_BNDRNormals":
		return FrechetMean_CMRep_BNDRNormals( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "CMRep_Abstract":
		return FrechetMean_CMRep_Abstract( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "CMRep_Abstract_Normal":
		return FrechetMean_CMRep_Abstract_Normal( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "Kendall2D":
		return FrechetMean_Kendall2D( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "Scale_Kendall2D":
		return FrechetMean_Scale_Kendall2D( dataList, maxIter, tol, stepsize )

	# elif dataList[ 0 ].Type == "MRep":
	# 	return FrechetMean_MRep( dataList, maxIter, tol )
	else:
		print( "Manifold type is not known" )
		return -1

def FrechetMean_Sphere( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	mu = dataList[0]
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	for i in range( maxIter ):
		dMu = manifolds.sphere_tVec( nManDim )

		for j in range( nData ):
			Log_mu_to_y_j = mu.LogMap( dataList[ j ] )

			for d in range( nManDim ):
				dMu.tVector[ d ] += stepsize * ( ( 1.0 / nData ) * Log_mu_to_y_j.tVector[ d ] )
			
		Mu_i = mu.ExponentialMap( dMu )
		mu = Mu_i

	return mu	

def FrechetMean_Kendall2D( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	mu = dataList[0]
	nManDim = dataList[ 0 ].nPt
	nData = len( dataList )

	for i in range( maxIter ):
		dMu = manifolds.kendall2D_tVec( nManDim )

		for j in range( nData ):
			Log_mu_to_y_j = mu.LogMap( dataList[ j ] )

			for d in range( nManDim ):
				for k in range( 2 ):
					dMu.tVector[ k ][ d ] += stepsize * ( ( 1.0 / nData ) * Log_mu_to_y_j.tVector[ k ][ d ] )
			
		Mu_i = mu.ExponentialMap( dMu )
		mu = Mu_i

	return mu	

def FrechetMean_PosReal( dataList, maxIter = 500, tol = 0.001, stepsize=0.01):
	mu = dataList[0]
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	for i in range( maxIter ):
		dMu = manifolds.pos_real_tVec( nManDim )

		for j in range( nData ):
			Log_mu_to_y_j = mu.LogMap( dataList[ j ] )

			for d in range( nManDim ):
				dMu.tVector[ d ] += stepsize * ( ( 1.0 / nData ) * Log_mu_to_y_j.tVector[ d ] ) 			
			
		Mu_i = mu.ExponentialMap( dMu )
		mu = Mu_i

	return mu	

def FrechetMean_Euclidean( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	mu = dataList[0]
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	for i in range( maxIter ):
		dMu = manifolds.euclidean_tVec( nManDim )

		for j in range( nData ):
			Log_mu_to_y_j = mu.LogMap( dataList[ j ] )

			for d in range( nManDim ):
				dMu.tVector[ d ] += stepsize * ( ( 1.0 / nData ) * Log_mu_to_y_j.tVector[ d ] ) 			
			
		Mu_i = mu.ExponentialMap( dMu )
		mu = Mu_i

	return mu	


def FrechetMean_CMRep( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	mu = manifolds.cmrep( nManDim )
	nData = len( dataList )

	for i in range( nManDim ):
		data_list_pos_i = []
		data_list_rad_i = []

		for j in range( nData ):
			data_list_pos_i.append( dataList[ j ].pt[ i ][ 0 ] )
			data_list_rad_i.append( dataList[ j ].pt[ i ][ 1 ] )

		mu_pos_i = FrechetMean( data_list_pos_i, maxIter, tol )
		mu_rad_i = FrechetMean( data_list_rad_i, maxIter, tol )

		mu.SetPosition( i, mu_pos_i.pt )
		mu.SetRadius( i, mu_rad_i.pt )

	mu.UpdateMeanRadius()	

	return mu

def FrechetMean_CMRep_BNDRNormals( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	mu = manifolds.cmrep_bndr_normals( nManDim )
	nData = len( dataList )

	for i in range( nManDim ):
		data_list_pos_i = []
		data_list_rad_i = []
		data_list_spoke1_i = [] 
		data_list_spoke2_i = []

		for j in range( nData ):
			data_list_pos_i.append( dataList[ j ].pt[ i ][ 0 ] )
			data_list_rad_i.append( dataList[ j ].pt[ i ][ 1 ] )
			data_list_spoke1_i.append( dataList[ j ].pt[ i ][ 2 ] )
			data_list_spoke2_i.append( dataList[ j ].pt[ i ][ 3 ] )

		mu_pos_i = FrechetMean( data_list_pos_i, maxIter, tol )
		mu_rad_i = FrechetMean( data_list_rad_i, maxIter, tol )
		mu_spoke1_i = FrechetMean( data_list_spoke1_i, maxIter, tol ) 
		mu_spoke2_i = FrechetMean( data_list_spoke2_i, maxIter, tol ) 

		mu.SetPosition( i, mu_pos_i.pt )
		mu.SetRadius( i, mu_rad_i.pt )
		mu.SetSpoke1( i, mu_spoke1_i.pt )
		mu.SetSpoke2( i, mu_spoke2_i.pt )

	mu.UpdateMeanRadius()	

	return mu

def FrechetMean_CMRep_Abstract( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	mu = manifolds.cmrep_abstract( nManDim )
	nData = len( dataList )

	mu_pt_arr = []

	for i in range( 4 ):
		data_list_i = []

		for j in range( nData ):
			data_list_i.append( dataList[ j ].pt[ i ] )

		mu_i = FrechetMean( data_list_i, maxIter, tol, stepsize )

		mu_pt_arr.append( mu_i )

	mu.SetPoint( mu_pt_arr ) 

	return mu


def FrechetMean_CMRep_Abstract_Normal( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	mu = manifolds.cmrep_abstract_normal( nManDim )
	nData = len( dataList )

	mu_pt_arr = []

	for i in range( 4 ):
		data_list_i = []

		for j in range( nData ):
			data_list_i.append( dataList[ j ].pt[ i ] )

		mu_i = FrechetMean( data_list_i, maxIter, tol, stepsize )

		mu_pt_arr.append( mu_i )

	mu_bndr1 = [] 
	mu_bndr2 = []
	
	for i in range( nManDim ):
		bndr1_list_i = [] 
		bndr2_list_i = []

		for j in range( nData ):
			bndr1_list_i.append( dataList[ j ].pt[ 4 ][ i ] )
			bndr2_list_i.append( dataList[ j ].pt[ 5 ][ i ] )			

		mu_bndr1_i = FrechetMean( bndr1_list_i, maxIter, tol, stepsize )
		mu_bndr2_i = FrechetMean( bndr2_list_i, maxIter, tol, stepsize )

		mu_bndr1.append( mu_bndr1_i )
		mu_bndr2.append( mu_bndr2_i )	

	mu_pt_arr.append( mu_bndr1 )
	mu_pt_arr.append( mu_bndr2 )

	mu.SetPoint( mu_pt_arr ) 

	return mu

def FrechetMean_Scale_Kendall2D( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nPt
	mu = manifolds.scale_kendall2D( nManDim )
	nData = len( dataList )

	mu_pt_arr = []

	for i in range( 2 ):
		data_list_i = []

		for j in range( nData ):
			data_list_i.append( dataList[ j ].pt[ i ] )

		mu_i = FrechetMean( data_list_i, maxIter, tol, stepsize )

		mu_pt_arr.append( mu_i )

	mu.SetPoint( mu_pt_arr ) 

	return mu	

def WeightedFrechetMean( dataList, wList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	if dataList[ 0 ].Type == "Sphere":
		return WeightedFrechetMean_Sphere( dataList, wList, maxIter, tol, stepsize )
	# elif dataList[ 0 ].Type == "PositiveReal":
	# 	return FrechetMean_PosReal( dataList, maxIter, tol, stepsize )
	# elif dataList[ 0 ].Type == "Euclidean":
	# 	return FrechetMean_Euclidean( dataList, maxIter, tol, stepsize )
	# elif dataList[ 0 ].Type == "CMRep":
	# 	return FrechetMean_CMRep( dataList, maxIter, tol, stepsize )
	# elif dataList[ 0 ].Type == "CMRep_BNDRNormals":
	# 	return FrechetMean_CMRep_BNDRNormals( dataList, maxIter, tol, stepsize )
	# elif dataList[ 0 ].Type == "CMRep_Abstract":
	# 	return FrechetMean_CMRep_Abstract( dataList, maxIter, tol, stepsize )
	# elif dataList[ 0 ].Type == "Kendall2D":
	# 	return FrechetMean_Kendall2D( dataList, maxIter, tol, stepsize )
	# elif dataList[ 0 ].Type == "Scale_Kendall2D":
	# 	return FrechetMean_Scale_Kendall2D( dataList, maxIter, tol, stepsize )

	# elif dataList[ 0 ].Type == "MRep":
	# 	return FrechetMean_MRep( dataList, maxIter, tol )
	else:
		print( "Manifold type is not known" )
		return -1

def WeightedFrechetMean_Sphere( dataList, wList, maxIter = 500, tol = 0.001, stepsize=0.005 ):
	# Weight List should be sum-to-one normalized 
	mu = dataList[0]
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	for i in range( maxIter ):
		dMu = manifolds.sphere_tVec( nManDim )

		for j in range( nData ):
			if np.abs( wList[ j ] ) < 1e-12:
				continue 

			Log_mu_to_y_j = mu.LogMap( dataList[ j ] )

			for d in range( nManDim ):
				dMu.tVector[ d ] += stepsize * ( ( wList[ j ] ) * Log_mu_to_y_j.tVector[ d ] )
			
		Mu_i = mu.ExponentialMap( dMu )
		mu = Mu_i

	return mu	


######################################
##########   Tangent PGA    ##########
######################################
def TangentPGA( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	if dataList[ 0 ].Type == "Sphere":
		return TangentPGA_Sphere( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "PositiveReal":
		return TangentPGA_PosReal( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "Euclidean":
		return TangentPGA_Euclidean( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "CMRep":
		return TangentPGA_CMRep( dataList, maxIter, tol, stepsize )
	elif dataList[ 0 ].Type == "CMRep_Abstract":
		return TangentPGA_CMRep_Abstract( dataList, maxIter, tol, stepsize )

	# elif dataList[ 0 ].Type == "MRep":
	# 	return FrechetMean_MRep( dataList, maxIter, tol )
	else:
		print( "Manifold type is not known" )
		return -1

def TangentPGA_Sphere( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	# Intrinsic Mean
	mu = FrechetMean( dataList, maxIter, tol, stepsize )

	# Covariance matrix on a tangent vector space
	S = np.zeros( [ nManDim, nManDim ] )

	for i in range( nData ):
		tVec_i = mu.LogMap( dataList[ i ] )

		u_j_mat = np.asmatrix( tVec_i.tVector )
		u_j_mat = u_j_mat.flatten()

		u_j_u_j_t = np.dot( u_j_mat.T, u_j_mat )

		S = np.add( S, np.multiply( 1.0 / float( nData ), u_j_u_j_t ) )

	# w : Eigen values
	# v : Eigen vectors
	[ w, v ] = np.linalg.eig( S )

	w_sortIdx = np.abs( w ).argsort()
	w = w[ w_sortIdx[ ::-1 ] ] 
	v = v[ :, w_sortIdx[ ::-1 ] ]

	w = np.real( w )
	v = np.real( v )

	return w, v, mu

def TangentPGA_PosReal( dataList, maxIter = 500, tol = 0.001, stepsize=0.01):
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	# Intrinsic Mean
	mu = FrechetMean( dataList, maxIter, tol, stepSize )

	# Covariance matrix on a tangent vector space
	S = np.zeros( [ nManDim, nManDim ] )

	for i in range( nData ):
		tVec_i = mu.LogMap( dataList[ i ] )

		u_j_mat = np.asmatrix( tVec_i.tVector )
		u_j_mat = u_j_mat.flatten()

		u_j_u_j_t = np.dot( u_j_mat.T, u_j_mat )

		S = np.add( S, np.multiply( 1.0 / float( nData ), u_j_u_j_t ) )

	# w : Eigen values
	# v : Eigen vectors
	[ w, v ] = np.linalg.eig( S )

	w_sortIdx = np.abs( w ).argsort()
	w = w[ w_sortIdx[ ::-1 ] ] 
	v = v[ :, w_sortIdx[ ::-1 ] ]

	w = np.real( w )
	v = np.real( v )

	return w, v, mu

def TangentPGA_Euclidean( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	# Intrinsic Mean
	mu = FrechetMean( dataList, maxIter, tol, stepSize )

	# Covariance matrix on a tangent vector space
	S = np.zeros( [ nManDim, nManDim ] )

	for i in range( nData ):
		tVec_i = mu.LogMap( dataList[ i ] )

		u_j_mat = np.asmatrix( tVec_i.tVector )
		u_j_mat = u_j_mat.flatten()

		u_j_u_j_t = np.dot( u_j_mat.T, u_j_mat )

		S = np.add( S, np.multiply( 1.0 / float( nData ), u_j_u_j_t ) )

	# w : Eigen values
	# v : Eigen vectors
	[ w, v ] = np.linalg.eig( S )

	w_sortIdx = np.abs( w ).argsort()
	w = w[ w_sortIdx[ ::-1 ] ] 
	v = v[ :, w_sortIdx[ ::-1 ] ]

	w = np.real( w )
	v = np.real( v )

	return w, v, mu

# Deprecated for now
def TangentPGA_CMRep( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	mu = FrechetMean( dataList, maxIter, tol, stepsize )

	return mu

def TangentPGA_CMRep_Abstract( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	print( "# of Data" )
	print( nData )

	# Intrinsic Mean
	mu = FrechetMean( dataList, maxIter, tol, stepsize )

	# Covariance matrix on a tangent vector space 
	nCenterDim = dataList[ 0 ].pt[ 0 ].nDim
	nScaleDim = dataList[ 0 ].pt[ 1 ].nDim
	nPreShapeDim = dataList[ 0 ].pt[ 2 ].nDim 
	nRadiusDim = dataList[ 0 ].pt[ 3 ].nDim 
	
	# Total Dimension
	nManDim_Cov = nCenterDim + nScaleDim + nPreShapeDim + nRadiusDim

	S = np.zeros( [ nManDim_Cov, nManDim_Cov ] )

	for i in range( nData ):
		tVec_i = mu.LogMap( dataList[ i ] )

		u_j_mat = np.zeros( [ 1, nManDim_Cov ] )

		u_j_mat_center = np.asarray( tVec_i.tVector[ 0 ].tVector ).flatten()
		u_j_mat_scale = np.asarray( tVec_i.tVector[ 1 ].tVector ).flatten()
		u_j_mat_preshape = np.asarray( tVec_i.tVector[ 2 ].tVector ).flatten()
		u_j_mat_radius = np.asarray( tVec_i.tVector[ 3 ].tVector ).flatten()
		
		for d in range( nCenterDim ):
			u_j_mat[ 0, d ] = u_j_mat_center[ d ]

		for d in range( nScaleDim ):
			# u_j_mat[ 0, d + nCenterDim ] = dataList[ i ].meanRadius * u_j_mat_scale[ d ]
			u_j_mat[ 0, d + nCenterDim ] = u_j_mat_scale[ d ]

		for d in range( nPreShapeDim ):
			# u_j_mat[ 0, d + nCenterDim + nScaleDim ] = dataList[ i ].meanRadius * u_j_mat_preshape[ d ]
			u_j_mat[ 0, d + nCenterDim + nScaleDim ] = u_j_mat_preshape[ d ]

		for d in range( nRadiusDim ):
			# u_j_mat[ 0, d + nCenterDim + nScaleDim + nPreShapeDim ] = dataList[ i ].meanRadius * u_j_mat_radius[ d ]
			u_j_mat[ 0, d + nCenterDim + nScaleDim + nPreShapeDim ] = u_j_mat_radius[ d ]

		u_j_u_j_t = np.dot( u_j_mat.T, u_j_mat )

		print( u_j_u_j_t.shape )

		S = np.add( S, np.multiply( 1.0 / float( nData ), u_j_u_j_t ) )

	# w : Eigen values
	# v : Eigen vectors
	[ w, v ] = np.linalg.eig( S )

	w_sortIdx = np.abs( w ).argsort()
	w = w[ w_sortIdx[ ::-1 ] ] 
	v = v[ :, w_sortIdx[ ::-1 ] ]

	w = np.real( w )
	v = np.real( v )

	return w, v, mu


def TangentPGA_CMRep_Abstract_Normal_Arr( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nObj = len( dataList )
	nData = len( dataList[ 0 ] )	
	nManDim = dataList[ 0 ][ 0 ].nDim

	print( "# of Data" )
	print( nData )

	mu_arr = []

	for i in range( nObj ):
		mu_i = FrechetMean_CMRep_Abstract_Normal( dataList[ i ], maxIter, tol, stepsize )
		mu_arr.append( mu_i )

	return mu_arr

	# # Covariance matrix on a tangent vector space 
	# nCenterDim = dataList[ 0 ][ 0 ].pt[ 0 ].nDim
	# nScaleDim = dataList[ 0 ][ 0 ].pt[ 1 ].nDim
	# nPreShapeDim = dataList[ 0 ][ 0 ].pt[ 2 ].nDim 
	# nRadiusDim = dataList[ 0 ][ 0 ].pt[ 3 ].nDim 
	# nNormal1Dim = len( dataList[ 0 ][ 0 ].pt[ 4 ] )
	# nNormal2Dim = len( dataList[ 0 ][ 0 ].pt[ 5 ] ) 

	# # Total Dimension
	# nManDim_Cov = ( nCenterDim + nScaleDim + nPreShapeDim + nRadiusDim + ( nNormal1Dim * 3 ) + ( nNormal1Dim * 3 ) ) * nObj

	# nManDim_a = nCenterDim + nScaleDim + nPreShapeDim + nRadiusDim + ( nNormal1Dim * 3 ) + ( nNormal1Dim * 3 )

	# S = np.zeros( [ nManDim_Cov, nManDim_Cov ] )

	# for i in range( nData ):
	# 	u_mat_i = []

	# 	for a in range( nObj ):
	# 		tVec_a_i = mu_arr[ a ].LogMap( dataList[ a ][ i ] )
	# 		u_mat_a_i = tVec_a_i.GetTangentVectorArray()
	# 		u_mat_i.extend( u_mat_a_i )

	# 	u_mat_i = np.asarray( u_mat_i )
	# 	u_i_u_i_t = np.dot( u_mat_i.T, u_mat_i )

	# 	print( u_i_u_i_t.shape )

	# 	S = np.add( S, np.multiply( 1.0 / float( nData ), u_i_u_i_t ) )

	# # w : Eigen values
	# # v : Eigen vectors
	# [ w, v ] = np.linalg.eig( S )

	# w_sortIdx = np.abs( w ).argsort()
	# w = w[ w_sortIdx[ ::-1 ] ] 
	# v = v[ :, w_sortIdx[ ::-1 ] ]

	# w = np.real( w )
	# v = np.real( v )

	# return w, v, mu_arr

def TangentPGA_CMRep_Abstract_Normal_Mu_Arr( dataList, mu_arr, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nObj = len( dataList )
	nData = len( dataList[ 0 ] )	
	nManDim = dataList[ 0 ][ 0 ].nDim

	print( "# of Data" )
	print( nData )

	# mu_arr = []

	# for i in range( nObj ):
	# 	mu_i = FrechetMean_CMRep_Abstract_Normal( dataList[ 0 ], maxIter, tol, stepsize )
	# 	mu_arr.append( mu_i )

	# Covariance matrix on a tangent vector space 
	nCenterDim = dataList[ 0 ][ 0 ].pt[ 0 ].nDim
	nScaleDim = dataList[ 0 ][ 0 ].pt[ 1 ].nDim
	nPreShapeDim = dataList[ 0 ][ 0 ].pt[ 2 ].nDim 
	nRadiusDim = dataList[ 0 ][ 0 ].pt[ 3 ].nDim 
	nNormal1Dim = len( dataList[ 0 ][ 0 ].pt[ 4 ] )
	nNormal2Dim = len( dataList[ 0 ][ 0 ].pt[ 5 ] ) 

	# Total Dimension
	nManDim_Cov = ( nCenterDim + nScaleDim + nPreShapeDim + nRadiusDim + ( nNormal1Dim * 3 ) + ( nNormal1Dim * 3 ) ) * nObj

	nManDim_a = nCenterDim + nScaleDim + nPreShapeDim + nRadiusDim + ( nNormal1Dim * 3 ) + ( nNormal1Dim * 3 )

	S = np.zeros( [ nManDim_Cov, nManDim_Cov ] )

	for i in range( nData ):
		u_mat_i = []

		for a in range( nObj ):
			tVec_a_i = mu_arr[ a ].LogMap( dataList[ a ][ i ] )
			u_mat_a_i = tVec_a_i.GetTangentVectorArray()
			u_mat_i.extend( u_mat_a_i )

		u_mat_i = np.asarray( u_mat_i )
		u_i_u_i_t = np.dot( u_mat_i.T, u_mat_i )

		print( u_i_u_i_t.shape )

		S = np.add( S, np.multiply( 1.0 / float( nData ), u_i_u_i_t ) )

	# w : Eigen values
	# v : Eigen vectors
	[ w, v ] = np.linalg.eig( S )

	w_sortIdx = np.abs( w ).argsort()
	w = w[ w_sortIdx[ ::-1 ] ] 
	v = v[ :, w_sortIdx[ ::-1 ] ]

	w = np.real( w )
	v = np.real( v )

	return w, v, mu_arr

#####################################################################
#######              Geodesic Regression Models               #######
#####################################################################
def GeodesicRegression( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol = 0.01, verbose=True ):
	if pt_list[ 0 ].Type == "Sphere":
		return GeodesicRegression_Sphere( t_list, pt_list, max_iter, stepSize, step_tol, verbose )
	elif pt_list[ 0 ].Type == "PositiveReal":
		return GeodesicRegression_PosReal( t_list, pt_list, max_iter, stepSize, step_tol, verbose )
	elif pt_list[ 0 ].Type == "Euclidean":
		return GeodesicRegression_Euclidean( t_list, pt_list, max_iter, stepSize, step_tol, verbose )
	elif pt_list[ 0 ].Type == "CMRep":
		return GeodesicRegression_CMRep( t_list, pt_list, max_iter, stepSize, step_tol, verbose )
	elif pt_list[ 0 ].Type == "CMRep_Abstract":
		return GeodesicRegression_CMRep_Abstract( t_list, pt_list, max_iter, stepSize, step_tol, verbose )
	elif pt_list[ 0 ].Type == "Kendall2D":
		return GeodesicRegression_Kendall2D( t_list, pt_list, max_iter, stepSize, step_tol, verbose )
	elif pt_list[ 0 ].Type == "Scale_Kendall2D":
		return GeodesicRegression_Scale_Kendall2D( t_list, pt_list, max_iter, stepSize, step_tol, verbose )
	else:
		print( "Manifold type is not known" )
		return -1

def GeodesicRegression_Sphere( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol = 1e-8, verbose=True ):
	nDimManifold = pt_list[ 0 ].nDim
	nData = len( pt_list )

	# Initial point on manifold and tangent vector
	t_min_idx = np.argmin( t_list )		
	p_anchor = pt_list[ t_min_idx ]
	t_max_idx = np.argmax( t_list )
	p_end = pt_list[ t_max_idx ]

	# Initial point on manifold and tangent vector
	init_Interp = manifolds.sphere( nDimManifold )
	init_Interp.SetPoint( p_anchor.pt )
	init_tVec = p_anchor.LogMap( p_end ) 

	base = init_Interp
	tangent = init_tVec
	
	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prevTangent = tangent

	for i in range( max_iter ):
		pt_grad = manifolds.sphere( nDimManifold )
		pt_grad.SetPoint( np.zeros( nDimManifold ).tolist() )
		tVec_grad = manifolds.sphere_tVec( nDimManifold )
		energy = 0.0

		for n in range( nData ):
			target = pt_list[ n ]
			time_pt = t_list[ n ]

			current_tangent = manifolds.sphere_tVec( nDimManifold ) 

			for d in range( nDimManifold ):
				current_tangent.tVector[ d ] = tangent.tVector[ d ] * time_pt

			estimate = base.ExponentialMap( current_tangent )

			# Tangent from base to estimate
			be = base.LogMap( estimate )
			# The tangential error on one data point
			et = estimate.LogMap( target )
			# Shooting in the opposite direction
			eb = estimate.LogMap( base )
		
			# Energy of the tangential error
			energy += et.normSquared()

			# Calculate adjoint gradient using Jacobi Field		
			jOutput, jOutputDash = estimate.AdjointGradientJacobi( eb, et, manifolds.sphere_tVec( nDimManifold ) )

			# Sum individual gradient from each data point to gradient
			for d in range( nDimManifold ):
				pt_grad.pt[ d ] = pt_grad.pt[ d ] + jOutput.tVector[ d ] 
				tVec_grad.tVector[ d ] = tVec_grad.tVector[ d ] + ( jOutputDash.tVector[ d ] * time_pt )
		
		# Gradient * stepSize
		pointGradient_Step = manifolds.sphere_tVec( nDimManifold )

		for d in range( nDimManifold ):
			pointGradient_Step.tVector[ d ] = pt_grad.pt[ d ] * stepSize

		# Update Base
		newBase = base.ExponentialMap( pointGradient_Step )

		# Update Tangent
		updatedTangent = manifolds.sphere_tVec( nDimManifold )

		for d in range( nDimManifold ):
			updatedTangent.tVector[ d ] = tangent.tVector[ d ] + tVec_grad.tVector[ d ] * stepSize

		# Parallel translate updated tangent from a previous base to the updated base
		newTangent = base.ParallelTranslateAtoB( base, newBase, updatedTangent )

		if energy > prevEnergy:
			stepSize = stepSize * 0.5
			base = prevBase
			tangent = prevTangent
			if verbose:
				print( "==================================" )
				print( "Warning: Energy Increased")			
				print( "Half the step size")			
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		else:
			prevBase = base
			prevTangent = tangent
			base = newBase
			tangent = newTangent
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )

		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent

def GeodesicRegression_Kendall2D( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol = 1e-8, verbose=True ):
	nDimManifold = pt_list[ 0 ].nPt
	nData = len( pt_list )

	t_min_idx = np.argmin( t_list )		
	p_anchor = pt_list[ t_min_idx ]
	t_max_idx = np.argmax( t_list )
	p_end = pt_list[ t_max_idx ]

	# Initial point on manifold and tangent vector
	init_Interp = manifolds.kendall2D( nDimManifold )
	init_Interp.SetPoint( p_anchor.pt )

	init_tVec = p_anchor.LogMap( p_end ) 

	base = init_Interp
	tangent = init_tVec

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prevTangent = tangent

	nUpdated = 0

	for i in range( max_iter ):
		pt_grad = manifolds.kendall2D( nDimManifold )
		tVec_grad = manifolds.kendall2D_tVec( nDimManifold )
		energy = 0.0

		for n in range( nData ):
			target = pt_list[ n ]
			time_pt = t_list[ n ]

			current_tangent = manifolds.kendall2D_tVec( nDimManifold ) 

			for d in range( nDimManifold ):
				for k in range( 2 ):
					current_tangent.tVector[ k, d ] = tangent.tVector[ k, d ] * time_pt

			estimate = base.ExponentialMap( current_tangent )

			# Tangent from base to estimate
			be = base.LogMap( estimate )
			# The tangential error on one data point
			et = estimate.LogMap( target )
			# Shooting in the opposite direction
			eb = estimate.LogMap( base )
		
			# Energy of the tangential error
			energy += et.normSquared()

			# Calculate adjoint gradient using Jacobi Field		
			jOutput, jOutputDash = estimate.AdjointGradientJacobi( eb, et, manifolds.kendall2D_tVec( nDimManifold ) )

			# Sum individual gradient from each data point to gradient
			for d in range( nDimManifold ):
				for k in range( 2 ):
					pt_grad.pt[ k, d ] = pt_grad.pt[ k, d ] + jOutput.tVector[ k, d ] 
					tVec_grad.tVector[ k, d ] = tVec_grad.tVector[ k, d ] + ( jOutputDash.tVector[ k, d ] * time_pt )
		
		# Gradient * stepSize
		pointGradient_Step = manifolds.kendall2D_tVec( nDimManifold )

		for d in range( nDimManifold ):
			for k in range( 2 ):
				pointGradient_Step.tVector[ k, d ] = pt_grad.pt[ k, d ] * stepSize

		# Update Base
		newBase = base.ExponentialMap( pointGradient_Step )

		# Update Tangent
		updatedTangent = manifolds.kendall2D_tVec( nDimManifold )

		for d in range( nDimManifold ):
			for k in range( 2 ):
				updatedTangent.tVector[ k, d ] = tangent.tVector[ k, d ] + tVec_grad.tVector[ k, d ] * stepSize

		# Parallel translate updated tangent from a previous base to the updated base
		newTangent = base.ParallelTranslateAtoB( base, newBase, updatedTangent )

		if energy >= prevEnergy:
			stepSize = stepSize * 0.5
			base = prevBase
			tangent = prevTangent

			if verbose:
				print( "==================================" )
				print( "Warning: Energy Increased")			
				print( "Half the step size")			
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		else:
			prevBase = base
			prevTangent = tangent
			base = newBase
			tangent = newTangent
			prevEnergy = energy
			nUpdated += 1 
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )

		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	print( "=============================" ) 
	print( " # of Actual Updates " )
	print( nUpdated )	
	print( "=============================" ) 

	return base, tangent

def GeodesicRegression_PosReal( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol =  1e-8, verbose=True ):
	nDimManifold = pt_list[ 0 ].nDim
	nData = len( pt_list )

	# Initial point on manifold and tangent vector
	init_Interp = manifolds.pos_real( nDimManifold )
	init_tVec = manifolds.pos_real_tVec( nDimManifold )

	init_Interp.SetPoint( pt_list[ 0 ].pt ) 

	base = init_Interp
	tangent = init_tVec


	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prevTangent = tangent

	for i in range( max_iter ):
		pt_grad = manifolds.pos_real( nDimManifold )
		pt_grad.SetPoint( np.ones( nDimManifold ).tolist() )
		tVec_grad = manifolds.pos_real_tVec( nDimManifold )
		energy = 0.0

		for n in range( nData ):
			target = pt_list[ n ]
			time_pt = t_list[ n ]

			current_tangent = manifolds.pos_real_tVec( nDimManifold ) 

			for d in range( nDimManifold ):
				current_tangent.tVector[ d ] = tangent.tVector[ d ] * time_pt

			estimate = base.ExponentialMap( current_tangent )

			# Tangent from base to estimate
			be = base.LogMap( estimate )
			# The tangential error on one data point
			et = estimate.LogMap( target )
			# Shooting in the opposite direction
			eb = estimate.LogMap( base )
		
			# Energy of the tangential error
			energy += et.normSquared()

			# Calculate adjoint gradient using Jacobi Field		
			jOutput, jOutputDash = estimate.AdjointGradientJacobi( eb, et, manifolds.pos_real_tVec( nDimManifold ) )

			# Sum individual gradient from each data point to gradient
			for d in range( nDimManifold ):
				pt_grad.pt[ d ] = pt_grad.pt[ d ] + jOutput.tVector[ d ] 
				tVec_grad.tVector[ d ] = tVec_grad.tVector[ d ] + ( jOutputDash.tVector[ d ] * time_pt )
		
		# Gradient * stepSize
		pointGradient_Step = manifolds.pos_real_tVec( nDimManifold )

		for d in range( nDimManifold ):
			pointGradient_Step.tVector[ d ] = pt_grad.pt[ d ] * stepSize

		# Update Base
		newBase = base.ExponentialMap( pointGradient_Step )

		# Update Tangent
		updatedTangent = manifolds.pos_real_tVec( nDimManifold )

		for d in range( nDimManifold ):
			updatedTangent.tVector[ d ] = tangent.tVector[ d ] + tVec_grad.tVector[ d ] * stepSize

		# Parallel translate updated tangent from a previous base to the updated base
		newTangent = base.ParallelTranslateAtoB( base, newBase, updatedTangent )

		if energy > prevEnergy:
			stepSize = stepSize * 0.5
			base = prevBase
			tangent = prevTangent
			if verbose:
				print( "==================================" )
				print( "Warning: Energy Increased")			
				print( "Half the step size")			
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		else:
			prevBase = base
			prevTangent = tangent
			base = newBase
			tangent = newTangent
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )

		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent

def GeodesicRegression_Euclidean( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol =  1e-8, verbose=True ):
	return LinearizedGeodesicRegression_Euclidean( t_list, pt_list, 100, stepSize, step_tol, False, verbose )

def GeodesicRegression_CMRep( t_list, pt_list, max_iter = 500, stepSize = 0.01, step_tol =  1e-8, verbose=True ):
	nManDim = pt_list[ 0 ].nDim
	base = manifolds.cmrep( nManDim )
	tangent = manifolds.cmrep_tVec( nManDim )

	nData = len( pt_list )

	for i in range( nManDim ):
		pt_list_pos_i = []
		pt_list_rad_i = []

		for j in range( nData ):
			pt_list_pos_i.append( pt_list[ j ].pt[ i ][ 0 ] )
			pt_list_rad_i.append( pt_list[ j ].pt[ i ][ 1 ] )

		t_list_pos_i = list( t_list )
		t_list_rad_i = list( t_list )

		print( t_list_pos_i )

		base_pos_i, tangent_pos_i = GeodesicRegression( t_list_pos_i, pt_list_pos_i, max_iter, stepSize, step_tol, False )
		base_rad_i, tangent_rad_i = GeodesicRegression( t_list_rad_i, pt_list_rad_i, max_iter, 1e-3, step_tol, True )

		base.SetPosition( i, base_pos_i.pt )
		base.SetRadius( i, base_rad_i.pt )

		tangent.SetPositionTangentVector( i, tangent_pos_i.tVector )
		tangent.SetRadiusTangentVector( i, tangent_rad_i.tVector )

	return base, tangent

def GeodesicRegression_CMRep_Abstract( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, verbose=True ):
	nManDim = pt_list[ 0 ].nDim
	base = manifolds.cmrep_abstract( nManDim )
	tangent = manifolds.cmrep_abstract_tVec( nManDim )
	nData = len( pt_list )

	base_pt_arr = []
	tangent_tVec_arr = []

	for i in range( 4 ):
		pt_list_i = []
		t_list_i = list( t_list )

		for j in range( nData ):
			pt_list_i.append( pt_list[ j ].pt[ i ] )
		
		base_i, tangent_i = GeodesicRegression( t_list_i, pt_list_i, max_iter, stepSize, step_tol, verbose )			

		base_pt_arr.append( base_i )
		tangent_tVec_arr.append( tangent_i )

	base.SetPoint( base_pt_arr )
	tangent.SetTangentVector( tangent_tVec_arr )

	base.UpdateMeanRadius()

	return base, tangent


def GeodesicRegression_Scale_Kendall2D( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, verbose=True ):
	nManDim = pt_list[ 0 ].nPt
	base = manifolds.scale_kendall2D( nManDim )
	tangent = manifolds.scale_kendall2D_tVec( nManDim )
	nData = len( pt_list )

	base_pt_arr = []
	tangent_tVec_arr = []

	for i in range( 2 ):
		pt_list_i = []
		t_list_i = list( t_list )

		for j in range( nData ):
			pt_list_i.append( pt_list[ j ].pt[ i ] )
		
		base_i, tangent_i = GeodesicRegression( t_list_i, pt_list_i, max_iter, stepSize, step_tol, verbose )			

		base_pt_arr.append( base_i )
		tangent_tVec_arr.append( tangent_i )

	base.SetPoint( base_pt_arr )
	tangent.SetTangentVector( tangent_tVec_arr )

	return base, tangent



#############################################################################
###               Anchor Point Linearized Geodesic Regression             ###
#############################################################################

def LinearizedGeodesicRegression( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol = 0.01, useFrechetMeanAnchor = False, verbose=False ):
	if pt_list[ 0 ].Type == "Sphere":
		return LinearizedGeodesicRegression_Sphere( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "Kendall2D":
		return LinearizedGeodesicRegression_Kendall2D( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "PositiveReal":
		return LinearizedGeodesicRegression_PosReal( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "Euclidean":
		return LinearizedGeodesicRegression_Euclidean( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "CMRep":
		return LinearizedGeodesicRegression_CMRep( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "CMRep_Abstract":
		return LinearizedGeodesicRegression_CMRep_Abstract( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "CMRep_Abstract_Normal":
		return LinearizedGeodesicRegression_CMRep_Abstract_Normal( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "Scale_Kendall2D":
		return LinearizedGeodesicRegression_Scale_Kendall2D( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	else:
		print( "Manifold type is not known" )
		return -1

def LinearizedGeodesicRegression_Sphere( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nData = len( pt_list )

	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 


	# Initialize an anchor point 	
	if useFrechetMeanAnchor:
		p_anchor = FrechetMean( pt_list ) 
	else:
		t_min_idx = np.argmin( t_list )		

		p_anchor = pt_list[ t_min_idx ]

	nManifoldDim = p_anchor.nDim 


	# Initial point on manifold and tangent vector
	init_Interp = manifolds.sphere( nManifoldDim )
	init_tVec = manifolds.sphere_tVec( nManifoldDim )

	base = init_Interp
	tangent = init_tVec

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prevTangent = tangent

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( pt_list[ j ] )

			for d in range( nManifoldDim ):
				w_list[d].append( tVec_j.tVector[d] )

		estModel_list = []

		for d in range( nManifoldDim ):
			t_list_sm = sm.add_constant( t_list )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, t_list_sm )
			est_d = LS_model_d.fit(method='qr')

			# est_d = LS_model_d.fit()
			estModel_list.append( est_d )

			# if verbose:
			# 	print( est_d.summary() )

		v_tangent_on_p_anchor = manifolds.sphere_tVec( nManifoldDim )
		v_to_base_on_p_anchor = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_tangent_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 1 ]
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		print( "Anchor point to base" )
		print( v_to_base_on_p_anchor.tVector )

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		newTangent = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor )	

		energy = 0

		for n in range( nData ):
			time_pt = t_list[ n ]
			target = pt_list[ n ] 

			current_tangent = manifolds.sphere_tVec( nManifoldDim ) 

			for d in range( nManifoldDim ):
				current_tangent.tVector[ d ] = newTangent.tVector[ d ] * time_pt

			estimate_n = newBase.ExponentialMap( current_tangent ) 
			et = estimate_n.LogMap( target ) 
	
			# Energy of the tangential error
			energy += et.normSquared()

		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )

			break;
		else:
			prevBase = newBase
			prevTangent = newTangent
			p_anchor = newBase
			base = newBase
			tangent = newTangent
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent

def LinearizedGeodesicRegression_Kendall2D( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nData = len( pt_list )

	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

	# Initialize an anchor point 	
	if useFrechetMeanAnchor:
		p_anchor = FrechetMean( pt_list ) 
	else:
		t_min_idx = np.argmin( t_list )		

		p_anchor = pt_list[ t_min_idx ]

	nManifoldDim = p_anchor.nPt

	# Initial point on manifold and tangent vector
	init_Interp = manifolds.kendall2D( nManifoldDim )
	init_tVec = manifolds.kendall2D_tVec( nManifoldDim )

	base = init_Interp
	tangent = init_tVec

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prevTangent = tangent

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			for k in range( 2 ):
				w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( pt_list[ j ] )

			for k in range( 2 ):
				for d in range( nManifoldDim ):		
					w_list[ k * nManifoldDim + d ].append( tVec_j.tVector[k, d] )

		estModel_list = []

		for k in range( 2 ):
			for d in range( nManifoldDim ):
				t_list_sm = sm.add_constant( t_list )
				w_d_np = np.asarray( w_list[ k * nManifoldDim + d ] )
				LS_model_d = sm.OLS( w_d_np, t_list_sm )
				est_d = LS_model_d.fit(method='qr')

				# est_d = LS_model_d.fit()
				estModel_list.append( est_d )

			# if verbose:
			# 	print( est_d.summary() )

		v_tangent_on_p_anchor = manifolds.kendall2D_tVec( nManifoldDim )
		v_to_base_on_p_anchor = manifolds.kendall2D_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_to_base_on_p_anchor.tVector[ 0, d ] = estModel_list[ d ].params[ 0 ] 
			v_to_base_on_p_anchor.tVector[ 1, d ] = estModel_list[ nManifoldDim + d ].params[ 0 ] 

			if len( estModel_list[ d ].params ) < 2:
				v_tangent_on_p_anchor.tVector[ 0, d ] = 0
			else:
				v_tangent_on_p_anchor.tVector[ 0, d ] = estModel_list[ d ].params[ 1 ]

			if len( estModel_list[ nManifoldDim + d ].params ) < 2:
				v_tangent_on_p_anchor.tVector[ 1, d ] = 0
			else:
				v_tangent_on_p_anchor.tVector[ 1, d ] = estModel_list[ nManifoldDim + d ].params[ 1 ]

		# print( "Anchor point to base" )
		# print( v_to_base_on_p_anchor.tVector )

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		newTangent = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor )	

		energy = 0

		for n in range( nData ):
			time_pt = t_list[ n ]
			target = pt_list[ n ] 

			current_tangent = manifolds.kendall2D_tVec( nManifoldDim ) 

			for k in range( 2 ):
				for d in range( nManifoldDim ):
					current_tangent.tVector[ k, d ] = newTangent.tVector[ k, d ] * time_pt

			estimate_n = newBase.ExponentialMap( current_tangent ) 
			et = estimate_n.LogMap( target ) 
	
			# Energy of the tangential error
			energy += et.normSquared()

		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )

			break;
		else:
			prevBase = newBase
			prevTangent = newTangent
			p_anchor = newBase
			base = newBase
			tangent = newTangent
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent

def LinearizedGeodesicRegression_PosReal( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nData = len( pt_list )

	if verbose:
		print( "======================================================" )
		print( "      Data  on Anchor Point Tangent Vector Space    " )
		print( "======================================================" ) 

	# Initialize an anchor point 	
	if useFrechetMeanAnchor:
		p_anchor = FrechetMean( pt_list ) 
	else:
		p_anchor = pt_list[ 0 ]
		
	nManifoldDim = p_anchor.nDim 

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( pt_list[ j ] )

			for d in range( nManifoldDim ):
				w_list[d].append( tVec_j.tVector[d] )

		if verbose:
			print( "=================================================================" )
			print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
			print( "=================================================================" ) 

		estModel_list = []

		for d in range( nManifoldDim ):
			t_list_sm = sm.add_constant( t_list )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, t_list_sm )
			est_d = LS_model_d.fit(method='qr')

			estModel_list.append( est_d )

			if verbose:
				print( est_d.summary() )

		v_tangent_on_p_anchor = manifolds.pos_real_tVec( nManifoldDim )
		v_to_base_on_p_anchor = manifolds.pos_real_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_tangent_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 1 ]
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		base = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		tangent = p_anchor.ParallelTranslateAtoB( p_anchor, base, v_tangent_on_p_anchor )	

		p_anchor = base

	return base, tangent

def LinearizedGeodesicRegression_Euclidean( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nData = len( pt_list )

	if verbose:
		print( "======================================================" )
		print( "      Data  on Anchor Point Tangent Vector Space    " )
		print( "======================================================" ) 

	# Initialize an anchor point 	
	if useFrechetMeanAnchor:
		p_anchor = FrechetMean( pt_list ) 
	else:
		t_min_idx = np.argmin( t_list )		

		p_anchor = pt_list[ t_min_idx ]
		
	nManifoldDim = p_anchor.nDim 

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( pt_list[ j ] )

			for d in range( nManifoldDim ):
				w_list[d].append( tVec_j.tVector[d] )

		if verbose:
			print( "=================================================================" )
			print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
			print( "=================================================================" ) 

		estModel_list = []

		for d in range( nManifoldDim ):
			t_list_sm = sm.add_constant( t_list )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, t_list_sm )
			est_d = LS_model_d.fit(method='qr')

			estModel_list.append( est_d )

			if verbose:
				print( est_d.summary() )

		v_tangent_on_p_anchor = manifolds.euclidean_tVec( nManifoldDim )
		v_to_base_on_p_anchor = manifolds.euclidean_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_tangent_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 1 ]
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		base = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		tangent = p_anchor.ParallelTranslateAtoB( p_anchor, base, v_tangent_on_p_anchor )	

		p_anchor = base

	return base, tangent


def LinearizedGeodesicRegression_CMRep( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nManDim = pt_list[ 0 ].nDim
	base = manifolds.cmrep( nManDim )
	tangent = manifolds.cmrep_tVec( nManDim )

	nData = len( pt_list )

	for i in range( nManDim ):
		pt_list_pos_i = []
		pt_list_rad_i = []

		for j in range( nData ):
			pt_list_pos_i.append( pt_list[ j ].pt[ i ][ 0 ] )
			pt_list_rad_i.append( pt_list[ j ].pt[ i ][ 1 ] )

		t_list_pos_i = list( t_list )
		t_list_rad_i = list( t_list )

		base_pos_i, tangent_pos_i = LinearizedGeodesicRegression( t_list_pos_i, pt_list_pos_i, max_iter, stepSize, step_tol, useFrechetMeanAnchor, False )
		base_rad_i, tangent_rad_i = LinearizedGeodesicRegression( t_list_rad_i, pt_list_rad_i, max_iter, stepSize, step_tol, useFrechetMeanAnchor, False )

		base.SetPosition( i, base_pos_i.pt )
		base.SetRadius( i, base_rad_i.pt )

		tangent.SetPositionTangentVector( i, tangent_pos_i.tVector )
		tangent.SetRadiusTangentVector( i, tangent_rad_i.tVector )

	return base, tangent 	

def LinearizedGeodesicRegression_CMRep_Abstract( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nManDim = pt_list[ 0 ].nDim
	base = manifolds.cmrep_abstract( nManDim )
	tangent = manifolds.cmrep_abstract_tVec( nManDim )
	nData = len( pt_list )

	base_pt_arr = []
	tangent_tVec_arr = []

	for i in range( 4 ):
		pt_list_i = []
		t_list_i = list( t_list )

		for j in range( nData ):
			pt_list_i.append( pt_list[ j ].pt[ i ] )
		
		base_i, tangent_i = LinearizedGeodesicRegression( t_list_i, pt_list_i, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )			

		base_pt_arr.append( base_i )
		tangent_tVec_arr.append( tangent_i )

	base.SetPoint( base_pt_arr )
	tangent.SetTangentVector( tangent_tVec_arr )

	base.UpdateMeanRadius()

	return base, tangent

def LinearizedGeodesicRegression_CMRep_Abstract_Normal( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nManDim = pt_list[ 0 ].nDim
	base = manifolds.cmrep_abstract_normal( nManDim )
	tangent = manifolds.cmrep_abstract_normal_tVec( nManDim )
	nData = len( pt_list )

	base_pt_arr = []
	tangent_tVec_arr = []

	for i in range( 4 ):
		pt_list_i = []
		t_list_i = list( t_list )

		for j in range( nData ):
			pt_list_i.append( pt_list[ j ].pt[ i ] )
		
		base_i, tangent_i = LinearizedGeodesicRegression( t_list_i, pt_list_i, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )			

		base_pt_arr.append( base_i )
		tangent_tVec_arr.append( tangent_i )

	base_normal1_arr = []
	base_normal2_arr = []

	tangent_normal1_arr = [] 
	tangent_normal2_arr = []

	for i in range( nManDim ):
		pt_normal1_list_i = []
		pt_normal2_list_i = []

		t_list1_i = list( t_list )
		t_list2_i = list( t_list )

		for j in range( nData ):
			pt_normal1_list_i.append( pt_list[ j ].pt[ 4 ][ i ] )
			pt_normal2_list_i.append( pt_list[ j ].pt[ 5 ][ i ] )

		base_normal1_i, tangent_normal1_i = LinearizedGeodesicRegression( t_list1_i, pt_normal1_list_i, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
		base_normal2_i, tangent_normal2_i = LinearizedGeodesicRegression( t_list2_i, pt_normal2_list_i, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )

		base_normal1_arr.append( base_normal1_i )
		base_normal2_arr.append( base_normal2_i )

		tangent_normal1_arr.append( tangent_normal1_i )
		tangent_normal2_arr.append( tangent_normal2_i )

	base_pt_arr.append( base_normal1_arr )
	base_pt_arr.append( base_normal2_arr )

	tangent_tVec_arr.append( tangent_normal1_arr )
	tangent_tVec_arr.append( tangent_normal2_arr )

	base.SetPoint( base_pt_arr )
	tangent.SetTangentVector( tangent_tVec_arr )

	base.UpdateMeanRadius()

	return base, tangent	

def LinearizedGeodesicRegression_Scale_Kendall2D( t_list, pt_list, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	nManDim = pt_list[ 0 ].nPt
	base = manifolds.scale_kendall2D( nManDim )
	tangent = manifolds.scale_kendall2D_tVec( nManDim )
	nData = len( pt_list )

	base_pt_arr = []
	tangent_tVec_arr = []

	for i in range( 2 ):
		pt_list_i = []
		t_list_i = list( t_list )

		for j in range( nData ):
			pt_list_i.append( pt_list[ j ].pt[ i ] )
		
		base_i, tangent_i = LinearizedGeodesicRegression( t_list_i, pt_list_i, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )			

		base_pt_arr.append( base_i )
		tangent_tVec_arr.append( tangent_i )

	base.SetPoint( base_pt_arr )
	tangent.SetTangentVector( tangent_tVec_arr )
	return base, tangent


####################################################################
###                 Statistical Validations                      ###
####################################################################

# R2 Statistics 
def R2Statistics( t_list, pt_list, base, tangent ):
	if base.Type == "Sphere":
		return R2Statistics_Sphere( t_list, pt_list, base, tangent )
	elif base.Type == "PositiveReal":
		return R2Statistics_PosReal( t_list, pt_list, base, tangent )
	elif base.Type == "Euclidean":
		return R2Statistics_Euclidean( t_list, pt_list, base, tangent )
	elif base.Type == "CMRep":
		return R2Statistics_CMRep( t_list, pt_list, base, tangent )
	elif base.Type == "CMRep_Abstract":
		return R2Statistics_CMRep_Abstract( t_list, pt_list, base, tangent )
	else:
		print( "Manifold Type Unknown" )
		return -1 


def R2Statistics_Mu( t_list, pt_list, base, tangent, mu ):
	if base.Type == "Sphere":
		return R2Statistics_Mu_Sphere( t_list, pt_list, base, tangent, mu )
	elif base.Type == "PositiveReal":
		return R2Statistics_Mu_PosReal( t_list, pt_list, base, tangent, mu )
	elif base.Type == "Euclidean":
		return R2Statistics_Mu_Euclidean( t_list, pt_list, base, tangent, mu )
	# elif base.Type == "CMRep":
	# 	return R2Statistics_CMRep( t_list, pt_list, base, tangent, mu )
	# elif base.Type == "CMRep_Abstract":
	# 	return R2Statistics_CMRep_Abstract( t_list, pt_list, base, tangent, mu )
	else:
		print( "Manifold Type Unknown" )
		return -1 



def R2Statistics_Sphere( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	mu = FrechetMean( pt_list )
	var_mu = 0	

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

	# Explained Variance w.r.t esitmated geodesic
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	return R2


def R2Statistics_Mu_Sphere( t_list, pt_list, base, tangent, mu ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	# mu = FrechetMean( pt_list )
	var_mu = 0	

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

	# Explained Variance w.r.t esitmated geodesic
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	return R2


def R2Statistics_PosReal( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	mu = FrechetMean( pt_list )
	var_mu = 0	

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

	# Explained Variance w.r.t esitmated geodesic
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.pos_real_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	return R2

def R2Statistics_Mu_PosReal( t_list, pt_list, base, tangent, mu ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	# mu = FrechetMean( pt_list )
	var_mu = 0	

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

	# Explained Variance w.r.t esitmated geodesic
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.pos_real_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	return R2


def R2Statistics_Euclidean( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	mu = FrechetMean( pt_list )
	var_mu = 0	

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

	# Explained Variance w.r.t esitmated geodesic
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.euclidean_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	return R2

def R2Statistics_Mu_Euclidean( t_list, pt_list, base, tangent, mu ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	# mu = FrechetMean( pt_list )
	var_mu = 0	

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

	# Explained Variance w.r.t esitmated geodesic
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.euclidean_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	return R2


def R2Statistics_CMRep( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	mu = FrechetMean( pt_list )
	mu.UpdateMeanRadius()

	var_mu = 0	

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		tVec_mu_to_y_i.SetMeanRadius( mu.meanRadius )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

	# Explained Variance w.r.t esitmated geodesic
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.cmrep_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ][ 0 ].tVector[ 0 ] = ( tangent.tVector[ d ][ 0 ].tVector[ 0 ] * t_i )
			tVec_at_t_i.tVector[ d ][ 0 ].tVector[ 1 ] = ( tangent.tVector[ d ][ 0 ].tVector[ 1 ] * t_i )
			tVec_at_t_i.tVector[ d ][ 0 ].tVector[ 2 ] = ( tangent.tVector[ d ][ 0 ].tVector[ 2 ] * t_i )

			tVec_at_t_i.tVector[ d ][ 1 ].tVector[ 0 ] = ( tangent.tVector[ d ][ 1 ].tVector[ 0 ] * t_i )


		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )
		est_pt_at_t_i.UpdateMeanRadius()

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )
		tVec_est_to_y_i.SetMeanRadius( est_pt_at_t_i.meanRadius )

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	return R2


def R2Statistics_CMRep_Abstract( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# Calculate intrinsic mean 
	print( "Calculating Frechet Mean... " )
	mu = FrechetMean( pt_list )
	mu.UpdateMeanRadius()

	var_mu = 0	

	print( "Calculating Variance..." )

	mean_area_s = 0
	mean_radius = 0

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		tVec_mu_to_y_i.SetMeanRadius( mu.meanRadius )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

		mean_area_s += ( pt_list[ i ].pt[ 1 ].pt[ 0 ] / float( nData ) )
		pt_list[ i ].UpdateMeanRadius()
		mean_radius += pt_list[ i ].meanRadius


	print( "Data Variance w.r.t Frechet Mean" ) 
	print( var_mu )

	# Explained Variance w.r.t esitmated geodesic
	print( "Calculating Variance w.r.t Estimated....")
	var_est = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = tangent.ScalarMultiply( t_i ) 

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )
		est_pt_at_t_i.UpdateMeanRadius()
		est_pt_at_t_i.SetMeanScale(  np.sqrt( mean_area_s ) * (1.0 / 3.0 ) )		

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )
		tVec_est_to_y_i.SetMeanRadius( mean_radius )
		tVec_est_to_y_i.SetMeanScale(  np.sqrt( mean_area_s ) * (1.0 / 3.0 ) ) 		

		var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )

	print( "Data Variance w.r.t Estimated Trend" ) 
	print( var_est ) 	

	return R2


def R2Statistics_CMRep_Abstract_Array( t_list, pt_list, base, tangent ): 
	nObject = len( pt_list )
	nData = len( pt_list[0] )
	nManifoldDim = pt_list[0][ 0 ].nDim

	var_mu = 0	
	var_est = 0

	for n in range( nObject ):
		mean_area_s = 0
		mean_radius = 0

		for i in range( nData ):
			mean_area_s += ( pt_list[ n ][ i ].pt[ 1 ].pt[ 0 ] / float( nData ) )
			pt_list[ n ][ i ].UpdateMeanRadius()

			mean_radius += ( pt_list[n][ i ].meanRadius / float( nData ) )

		print( "Mean Area" )
		print( mean_area_s )
		print( "Mean Radius" )
		print( mean_radius )

		# Calculate intrinsic mean 
		print( "Calculating Frechet Mean... " )

		mu = FrechetMean( pt_list[ n ] )
		mu.UpdateMeanRadius()

		print( "Calculating Variance..." )

		# Variance w.r.t the mean 
		for i in range( nData ):
			tVec_mu_to_y_i = mu.LogMap( pt_list[ n ][ i ] )
			tVec_mu_to_y_i.SetMeanRadius( mean_radius )
			tVec_mu_to_y_i.SetMeanScale( np.sqrt( mean_area_s ) * (1.0 / 3.0 ) )

			var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

		# Explained Variance w.r.t esitmated geodesic
		print( "Calculating Variance w.r.t Estimated....")

		for i in range( nData ):
			t_i = t_list[ i ]

			# Tangent Vector * time
			tVec_at_t_i = tangent[ n ].ScalarMultiply( t_i ) 

			est_pt_at_t_i = base[ n ].ExponentialMap( tVec_at_t_i )
			est_pt_at_t_i.UpdateMeanRadius()

			tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ n ][ i ] )
			tVec_est_to_y_i.SetMeanRadius( mean_radius )
			tVec_est_to_y_i.SetMeanScale( np.sqrt( mean_area_s ) * (1.0 / 3.0 ) )

			var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	print( "Data Variance w.r.t Frechet Mean" ) 
	print( var_mu )

	print( "Data Variance w.r.t Estimated Trend" ) 
	print( var_est ) 	

	return R2

def R2Statistics_CMRep_Abstract_Normal_Array( t_list, pt_list, meanArray, base, tangent ): 
	nObject = len( pt_list )
	nData = len( pt_list[0] )
	nManifoldDim = pt_list[0][ 0 ].nDim

	var_mu = 0	
	var_est = 0

	for n in range( nObject ):
		mean_area_s = 0
		mean_radius = 0

		for i in range( nData ):
			mean_area_s += ( pt_list[ n ][ i ].pt[ 1 ].pt[ 0 ] / float( nData ) )
			pt_list[ n ][ i ].UpdateMeanRadius()

			mean_radius += ( pt_list[n][ i ].meanRadius / float( nData ) )

		print( "Mean Area" )
		print( mean_area_s )
		print( "Mean Radius" )
		print( mean_radius )

		# Calculate intrinsic mean 
		print( "Calculating Frechet Mean... " )

		mu = meanArray[ n ]
		mu.UpdateMeanRadius()

		print( "Calculating Variance..." )

		# Variance w.r.t the mean 
		for i in range( nData ):
			tVec_mu_to_y_i = mu.LogMap( pt_list[ n ][ i ] )
			tVec_mu_to_y_i.SetMeanRadius( mean_radius )
			tVec_mu_to_y_i.SetMeanScale( np.sqrt( mean_area_s ) * (1.0 / 3.0 ) )

			var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

		# Explained Variance w.r.t esitmated geodesic
		print( "Calculating Variance w.r.t Estimated....")

		for i in range( nData ):
			t_i = t_list[ i ]

			# Tangent Vector * time
			tVec_at_t_i = tangent[ n ].ScalarMultiply( t_i ) 

			est_pt_at_t_i = base[ n ].ExponentialMap( tVec_at_t_i )
			est_pt_at_t_i.UpdateMeanRadius()

			tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ n ][ i ] )
			tVec_est_to_y_i.SetMeanRadius( mean_radius )
			tVec_est_to_y_i.SetMeanScale( np.sqrt( mean_area_s ) * (1.0 / 3.0 ) )

			var_est += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	R2 = ( 1 - ( var_est / var_mu ) )
	print( "Data Variance w.r.t Frechet Mean" ) 
	print( var_mu )

	print( "Data Variance w.r.t Estimated Trend" ) 
	print( var_est ) 	

	return R2


def RootMeanSquaredError( t_list, pt_list, base, tangent ):
	if base.Type == "Sphere":
		return RootMeanSquaredError_Sphere( t_list, pt_list, base, tangent )
	elif base.Type == "PositiveReal":
		return RootMeanSquaredError_PosReal( t_list, pt_list, base, tangent )
	elif base.Type == "Euclidean":
		return RootMeanSquaredError_Euclidean( t_list, pt_list, base, tangent )
	elif base.Type == "CMRep":
		return RootMeanSquaredError_CMRep( t_list, pt_list, base, tangent )
	elif base.Type == "CMRep_Abstract":
		return RootMeanSquaredError_CMRep_Abstract( t_list, pt_list, base, tangent )
	else:
		print( "Manifold Type Unknown" )
		return -1 

def RootMeanSquaredError_Sphere( t_list, pt_list, base, tangent ):
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# RMSE w.r.t esitmated geodesic
	rmse = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		rmse += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	rmse = np.sqrt( rmse )
	return rmse

def RootMeanSquaredError_PosReal( t_list, pt_list, base, tangent ):
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# RMSE w.r.t esitmated geodesic
	rmse = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.pos_real_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		rmse += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	rmse = np.sqrt( rmse )

	return rmse

def RootMeanSquaredError_Euclidean( t_list, pt_list, base, tangent ):
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# RMSE w.r.t esitmated geodesic
	rmse = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.euclidean_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			tVec_at_t_i.tVector[ d ] = ( tangent.tVector[ d ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		rmse += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	rmse = np.sqrt( rmse )

	return rmse

def RootMeanSquaredError_CMRep( t_list, pt_list, base, tangent ):
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# RMSE w.r.t esitmated geodesic
	rmse = 0

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.cmrep_tVec( nManifoldDim )

		for d in range( nManifoldDim ):			
			tVec_at_t_i.tVector[ d ][ 0 ].tVector[ 0 ]  = ( tangent.tVector[ d ][ 0 ].tVector[ 0 ] * t_i )
			tVec_at_t_i.tVector[ d ][ 0 ].tVector[ 1 ]  = ( tangent.tVector[ d ][ 0 ].tVector[ 1 ] * t_i )
			tVec_at_t_i.tVector[ d ][ 0 ].tVector[ 2 ]  = ( tangent.tVector[ d ][ 0 ].tVector[ 2 ] * t_i )

			tVec_at_t_i.tVector[ d ][ 1 ].tVector[ 0 ]  = ( tangent.tVector[ d ][ 1 ].tVector[ 0 ] * t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		rmse += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	rmse = np.sqrt( rmse )

	return rmse

def RootMeanSquaredError_CMRep_Abstract( t_list, pt_list, base, tangent ):
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	# RMSE w.r.t esitmated geodesic
	rmse = 0

	mean_area_s = 0
	mean_radius = 0

	for i in range( nData ):
		mean_area_s += ( pt_list[ i ].pt[ 1 ].pt[ 0 ] / float( nData ) )
		pt_list[ i ].UpdateMeanRadius()
		mean_radius += ( pt_list[ i ].meanRadius / float( nData ) )

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.cmrep_abstract_tVec( nManifoldDim )

		for j in range( 4 ):
			tVec_at_t_i.tVector[ j ] = tangent.tVector[ j ].ScalarMultiply( t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )
		tVec_est_to_y_i.SetMeanScale( mean_area_s ** (1.0 / 3.0 ) )
		tVec_est_to_y_i.SetMeanRadius( mean_radius )

		rmse += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	return np.sqrt( rmse )


def RootMeanSquaredError_CMRep_Abstract_Array( t_list, pt_list, base, tangent ): 
	nObject = len( pt_list )
	nData = len( pt_list[0] )
	nManifoldDim = pt_list[0][ 0 ].nDim

	rmse = 0	

	for n in range( nObject ):
		mean_area_s = 0
		mean_radius = 0

		for i in range( nData ):
			mean_area_s += ( pt_list[ n ][ i ].pt[ 1 ].pt[ 0 ] / float( nData ) )
			pt_list[ n ][ i ].UpdateMeanRadius()

			mean_radius += ( pt_list[n][ i ].meanRadius / float( nData ) )


		for i in range( nData ):
			t_i = t_list[ i ]

			# Tangent Vector * time
			tVec_at_t_i = tangent[ n ].ScalarMultiply( t_i ) 

			est_pt_at_t_i = base[ n ].ExponentialMap( tVec_at_t_i )
			est_pt_at_t_i.UpdateMeanRadius()

			tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ n ][ i ] )
			tVec_est_to_y_i.SetMeanRadius( mean_radius )
			tVec_est_to_y_i.SetMeanScale(  mean_area_s ** (1.0 / 3.0 ) )			

			rmse += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	rmse = np.sqrt( rmse )

	return rmse


def R2Statistics_CMRep_Atom( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim
	R2_Atom = []
	R2_pos_atom = [] 
	R2_rad_atom = [] 

	# Calculate intrinsic mean 
	for i in range( nManifoldDim ):
		pt_list_pos_i = []
		pt_list_rad_i = []

		for j in range( nData ):
			pt_list_pos_i.append( pt_list[ j ].pt[ i ][ 0 ] )
			pt_list_rad_i.append( pt_list[ j ].pt[ i ][ 1 ] )

		t_list_pos_i = list( t_list )
		t_list_rad_i = list( t_list )		

		base_pos_i = base.pt[ i ][ 0 ]
		tangent_pos_i = tangent.tVector[ i ][ 0 ]

		base_rad_i = base.pt[ i ][ 1 ]
		tangent_rad_i = tangent.tVector[ i ][ 1 ]

		R2_pos_i = R2Statistics( t_list_pos_i, pt_list_pos_i, base_pos_i, tangent_pos_i )
		R2_rad_i = R2Statistics( t_list_rad_i, pt_list_rad_i, base_rad_i, tangent_rad_i ) 

		R2_pos_atom.append( R2_pos_i )
		R2_rad_atom.append( R2_rad_i )

	R2_atom = [ R2_pos_atom, R2_rad_atom ]

	return R2_atom


def R2Statistics_CMRep_Abstract_Atom( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim
	R2_Atom = []
	R2_Center = 0
	R2_Scale = 0

	RMSE_Atom = []
	RMSE_Center = 0
	RMSE_Scale = 0

	# Center - Global, Euclidean
	pt_list_center = []
	t_list_center = list( t_list )

	for i in range( nData ):
		pt_list_center.append( pt_list[ i ].pt[ 0 ] )

	base_center = base.pt[ 0 ]
	tangent_center = tangent.tVector[ 0 ] 

	## R2
	R2_center = R2Statistics( t_list_center, pt_list_center, base_center, tangent_center )
	
 	## RMSE
	RMSE_Center = RootMeanSquaredError( t_list_center, pt_list_center, base_center, tangent_center )

	# Scale - Global, Positive Real
	pt_list_scale = []
	t_list_scale = list( t_list )

	for i in range( nData ):
		pt_list_scale.append( pt_list[ i ].pt[ 1 ] )

	print( "RMSE Scale Check" )
	print( "41st Atom" )	
	print( pt_list_scale[ 40 ].pt[ 0 ] )


	base_scale = base.pt[ 1 ]
	tangent_scale = tangent.tVector[ 1 ] 

	## R2 
	R2_scale = R2Statistics( t_list_scale, pt_list_scale, base_scale, tangent_scale )

	## RMSE 
	RMSE_scale = RootMeanSquaredError( t_list_scale, pt_list_scale, base_scale, tangent_scale )

	# Postion Abstract - Global, Sphere
	pt_list_pos_abst = []
	t_list_pos_abst = list( t_list )

	for i in range( nData ):
		pt_list_pos_abst.append( pt_list[ i ].pt[ 2 ] )

	base_pos_abst = base.pt[ 2 ]
	tangent_pos_abst = tangent.tVector[ 2 ] 

	## R2
	R2_pos_abst = R2Statistics( t_list_pos_abst, pt_list_pos_abst, base_pos_abst, tangent_pos_abst )

	## RMSE
	RMSE_pos_abst = RootMeanSquaredError( t_list_pos_abst, pt_list_pos_abst, base_pos_abst, tangent_pos_abst ) 

	# Relative Position - Local, Euclidean
	pt_list_pos_abst = []
	t_list_pos_abst = list( t_list )

	for i in range( nData ):
		pt_list_pos_abst.append( pt_list[ i ].pt[ 2 ] )

	base_pos_abst = base.pt[ 2 ]
	tangent_pos_abst = tangent.tVector[ 2 ] 

	## Calculate a Frechet mean of relative postions
	mu_pos_abstr = FrechetMean( pt_list_pos_abst ) 

	H_sub = HelmertSubmatrix( nManifoldDim )

	H_sub_T = H_sub.T

	## Frechet Mean : Relative Positions on a 3(n-1)-1 sphere
	mu_pos_abstr_sphere_matrix = np.array( mu_pos_abstr.pt ).reshape( -1, 3 )

	## Frechet Mean : Relative Positions on Euclidean
	mu_pos_abstr_euclidean_matrix = np.dot( H_sub_T, mu_pos_abstr_sphere_matrix )

	## Estimated Trajectory 
	geodesic_trend_euclidean_arr = [] 
	data_euclidean_arr = []

	for i in range( nData ):
		t_i = t_list_pos_abst[ i ]

		## Estimated Points from Sphere to Euclidean
		tVec_at_t_i = tangent_pos_abst.ScalarMultiply( t_i )
		est_pt_at_t_i = base_pos_abst.ExponentialMap( tVec_at_t_i )

		est_pt_at_t_i_sphere_matrix = np.array( est_pt_at_t_i.pt ).reshape( -1, 3 )
		est_pt_at_t_i_euclidean_matrix = np.dot( H_sub_T, est_pt_at_t_i_sphere_matrix )

		geodesic_trend_euclidean_arr.append( est_pt_at_t_i_euclidean_matrix )

		## Data points from Sphere to Euclidean
		data_i = pt_list_pos_abst[ i ] 

		data_i_sphere_matrix = np.array( data_i.pt ).reshape( -1, 3 )
		data_i_euclidean_matrix = np.dot( H_sub_T, data_i_sphere_matrix )

		data_euclidean_arr.append( data_i_euclidean_matrix )	

	## Calculate atom-wise locational R^2 on Euclidean metric
	## R2
	R2_Pos_Euclidean_Atom = [] 
	
	## RMSE
	RMSE_Pos_Euclidean_Atom = []

	for d in range( nManifoldDim ):
		var_mu_d = 0
		var_est_d = 0

		for i in range( nData ):
			# Data 
			pt_i_d = data_euclidean_arr[ i ][ d, : ] 

			# Mean
			mu_i_d = mu_pos_abstr_euclidean_matrix[ d, : ] 

			# Estimated 
			est_i_d = geodesic_trend_euclidean_arr[ i ][ d, : ] 

			sqDist_mu_i_d = np.linalg.norm( np.subtract( pt_i_d, mu_i_d ) ) ** 2

			sqDist_est_i_d = np.linalg.norm( np.subtract( pt_i_d, est_i_d ) ) ** 2 

			var_mu_d += sqDist_mu_i_d

			var_est_d += sqDist_est_i_d

		R2_d = ( 1 - ( var_est_d / var_mu_d ) ) 

		R2_Pos_Euclidean_Atom.append( R2_d )
		RMSE_Pos_Euclidean_Atom.append( np.sqrt( var_est_d ) )

	# Radius - Local, Positive Real : log-Euclidean
	## R2
	R2_Rad_PosReal_Atom = []

	## RMSE
	RMSE_Rad_PosReal_Atom = []

	pt_list_rad = []
	t_list_rad = list( t_list )

	for i in range( nData ):
		pt_list_rad.append( pt_list[ i ].pt[ 3 ] )

	base_rad = base.pt[ 3 ]
	tangent_rad = tangent.tVector[ 3 ] 

	## Calculate a Frechet mean of Radius
	mu_rad = FrechetMean( pt_list_rad ) 

	## Estimated Trend Trajectory
	geodesic_trend_rad_arr = [] 

	for i in range( nData ):
		t_i = t_list_rad[ i ]

		## Estimated Points from Sphere to Euclidean
		tVec_at_t_i = tangent_rad.ScalarMultiply( t_i )
		est_pt_at_t_i = base_rad.ExponentialMap( tVec_at_t_i )

		geodesic_trend_rad_arr.append( est_pt_at_t_i )

	for d in range( nManifoldDim ):
		var_mu_d = 0
		var_est_d = 0 

		for i in range( nData ):
			# Data 
			pt_i_d = pt_list_rad[ i ].pt[ d ] 

			# Mean
			mu_i_d = mu_rad.pt[ d ]

			# Estimated 
			est_i_d = geodesic_trend_rad_arr[ i ].pt[ d ] 

			# Sq. distance to the Frechet mean 
			sqDist_mu_i_d = ( np.log( pt_i_d ) - np.log( mu_i_d ) ) ** 2 

			# Sq. distance to the estimated trajectory
			sqDist_est_i_d = ( np.log( pt_i_d ) - np.log( est_i_d ) ) ** 2 

			var_mu_d += sqDist_mu_i_d

			var_est_d += sqDist_est_i_d

		R2_rad_d = ( 1 - ( var_est_d / var_mu_d ) )

		R2_Rad_PosReal_Atom.append( R2_rad_d )
		RMSE_Rad_PosReal_Atom.append( np.sqrt( var_est_d / float( nData ) ) ) 


	# All R2 Statistics
	R2_atom = [ R2_center, R2_scale, R2_pos_abst, R2_Pos_Euclidean_Atom, R2_Rad_PosReal_Atom ]
	RMSE_Atom = [ RMSE_Center, RMSE_scale, RMSE_pos_abst, RMSE_Pos_Euclidean_Atom, RMSE_Rad_PosReal_Atom ]

	return R2_atom, RMSE_Atom


def R2Statistics_CMRep_Abstract_Normal_Atom( t_list, pt_list, mu, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim

	R2_Atom = []
	R2_Center = 0
	R2_Scale = 0

	RMSE_Atom = []
	RMSE_Center = 0
	RMSE_Scale = 0

	# Center - Global, Euclidean
	pt_list_center = []
	t_list_center = list( t_list )

	for i in range( nData ):
		pt_list_center.append( pt_list[ i ].pt[ 0 ] )

	base_center = base.pt[ 0 ]
	tangent_center = tangent.tVector[ 0 ] 

	## R2
	R2_center = R2Statistics_Mu( t_list_center, pt_list_center, base_center, tangent_center, mu.pt[ 0 ] )
	
 	## RMSE
	RMSE_Center = RootMeanSquaredError( t_list_center, pt_list_center, base_center, tangent_center )

	# Scale - Global, Positive Real
	pt_list_scale = []
	t_list_scale = list( t_list )

	for i in range( nData ):
		pt_list_scale.append( pt_list[ i ].pt[ 1 ] )

	print( "RMSE Scale Check" )
	print( "41st Atom" )	
	print( pt_list_scale[ 40 ].pt[ 0 ] )

	base_scale = base.pt[ 1 ]
	tangent_scale = tangent.tVector[ 1 ] 

	## R2 
	R2_scale = R2Statistics_Mu( t_list_scale, pt_list_scale, base_scale, tangent_scale, mu.pt[ 1 ] )

	## RMSE 
	RMSE_scale = RootMeanSquaredError( t_list_scale, pt_list_scale, base_scale, tangent_scale )

	# Postion Abstract - Global, Sphere
	pt_list_pos_abst = []
	t_list_pos_abst = list( t_list )

	for i in range( nData ):
		pt_list_pos_abst.append( pt_list[ i ].pt[ 2 ] )

	base_pos_abst = base.pt[ 2 ]
	tangent_pos_abst = tangent.tVector[ 2 ] 

	## R2
	R2_pos_abst = R2Statistics_Mu( t_list_pos_abst, pt_list_pos_abst, base_pos_abst, tangent_pos_abst, mu.pt[ 2 ] )

	## RMSE
	RMSE_pos_abst = RootMeanSquaredError( t_list_pos_abst, pt_list_pos_abst, base_pos_abst, tangent_pos_abst ) 

	# Radius - Local, Positive Real : log-Euclidean
	## R2
	R2_Rad_PosReal_Atom = []

	## RMSE
	RMSE_Rad_PosReal_Atom = []

	pt_list_rad = []
	t_list_rad = list( t_list )

	for i in range( nData ):
		pt_list_rad.append( pt_list[ i ].pt[ 3 ] )

	base_rad = base.pt[ 3 ]
	tangent_rad = tangent.tVector[ 3 ] 

	## Calculate a Frechet mean of Radius
	mu_rad = mu.pt[ 3 ]

	for d in range( nManifoldDim ):
		t_list_rad_d = list( t_list ) 

		pt_list_rad_d = []

		for i in range( nData ):
			rad_d_i = manifolds.pos_real( 1 )
			rad_d_i.SetPoint( pt_list_rad[ i ].pt[ d ] )

			pt_list_rad_d.append( rad_d_i )

		base_rad_d = manifolds.pos_real( 1 )
		base_rad_d.SetPoint( [ base_rad.pt[ d ] ] )

		tangent_rad_d = manifolds.pos_real_tVec( 1 )
		tangent_rad_d.SetTangentVector( tangent_rad.tVector[ d ] )

		mu_rad_d = manifolds.pos_real( 1 )
		mu_rad_d.SetPoint( [ mu_rad.pt[ d ] ] )

		R2_rad_d = R2Statistics_Mu( t_list_rad_d, pt_list_rad_d, base_rad_d, tangent_rad_d, mu_rad_d )

		R2_Rad_PosReal_Atom.append( R2_rad_d )		

		RMSE_rad_d = RootMeanSquaredError( t_list_rad_d, pt_list_rad_d, base_rad_d, tangent_rad_d )  

		RMSE_Rad_PosReal_Atom.append( RMSE_rad_d )		

	# Boundary Normal 1 - Local, S^2
	## R2
	R2_Normal1_Sphere_Atom = []

	## RMSE
	RMSE_Normal1_Sphere_Atom = []

	pt_list_normal1 = []
	t_list_normal1 = list( t_list )

	for i in range( nData ):
		pt_list_normal1.append( pt_list[ i ].pt[ 4 ] )

	base_normal1 = base.pt[ 4 ]
	tangent_normal1 = tangent.tVector[ 4 ] 

	mu_normal1 = mu.pt[ 4 ]

	for d in range( nManifoldDim ):
		t_list_normal1_d = list( t_list ) 

		pt_list_normal1_d = []

		for i in range( nData ):
			pt_list_normal1_d.append( pt_list_normal1[ i ][ d ] )

		base_normal1_d = manifolds.sphere( 3 )
		base_normal1_d.SetPoint( base_normal1[ d ].pt )

		tangent_normal1_d = manifolds.sphere_tVec( 3 )
		tangent_normal1_d.SetTangentVector( tangent_normal1[ d ].tVector )

		mu_normal1_d = manifolds.sphere( 3 )
		mu_normal1_d.SetPoint( mu_normal1[ d ].pt )

		R2_normal1_d = R2Statistics_Mu( t_list_normal1_d, pt_list_normal1_d, base_normal1_d, tangent_normal1_d, mu_normal1_d )

		R2_Normal1_Sphere_Atom.append( R2_normal1_d )		

		RMSE_normal1_d = RootMeanSquaredError( t_list_normal1_d, pt_list_normal1_d, base_normal1_d, tangent_normal1_d )  

		RMSE_Normal1_Sphere_Atom.append( RMSE_normal1_d )		


	# Boundary Normal 2 - Local, S^2
	## R2
	R2_Normal2_Sphere_Atom = []

	## RMSE
	RMSE_Normal2_Sphere_Atom = []

	pt_list_normal2 = []
	t_list_normal2 = list( t_list )

	for i in range( nData ):
		pt_list_normal2.append( pt_list[ i ].pt[ 5 ] )

	base_normal2 = base.pt[ 5 ]
	tangent_normal2 = tangent.tVector[ 5 ] 

	mu_normal2 = mu.pt[ 5 ]

	for d in range( nManifoldDim ):
		t_list_normal2_d = list( t_list ) 

		pt_list_normal2_d = []

		for i in range( nData ):
			pt_list_normal2_d.append( pt_list_normal2[ i ][ d ] )

		base_normal2_d = manifolds.sphere( 3 )
		base_normal2_d.SetPoint( base_normal2[ d ].pt )

		tangent_normal2_d = manifolds.sphere_tVec( 3 )
		tangent_normal2_d.SetTangentVector( tangent_normal2[ d ].tVector )

		mu_normal2_d = manifolds.sphere( 3 )
		mu_normal2_d.SetPoint( mu_normal2[ d ].pt )

		R2_normal2_d = R2Statistics_Mu( t_list_normal2_d, pt_list_normal2_d, base_normal2_d, tangent_normal2_d, mu_normal2_d )

		R2_Normal2_Sphere_Atom.append( R2_normal2_d )		

		RMSE_normal2_d = RootMeanSquaredError( t_list_normal2_d, pt_list_normal2_d, base_normal2_d, tangent_normal2_d )  

		RMSE_Normal2_Sphere_Atom.append( RMSE_normal2_d )		


	# All R2 Statistics
	R2_atom = [ R2_center, R2_scale, R2_pos_abst, R2_Rad_PosReal_Atom, R2_Normal1_Sphere_Atom, R2_Normal2_Sphere_Atom ]
	RMSE_Atom = [ RMSE_Center, RMSE_scale, RMSE_pos_abst, RMSE_Rad_PosReal_Atom, RMSE_Normal1_Sphere_Atom, RMSE_Normal2_Sphere_Atom ]

	return R2_atom, RMSE_Atom


def RootMeanSquaredError_CMRep_Atom( t_list, pt_list, base, tangent ): 
	nData = len( pt_list )
	nManifoldDim = pt_list[ 0 ].nDim
	RMSE_Atom = []
	RMSE_pos_atom = [] 
	RMSE_rad_atom = [] 

	# Calculate intrinsic mean 
	for i in range( nManifoldDim ):
		pt_list_pos_i = []
		pt_list_rad_i = []

		for j in range( nData ):
			pt_list_pos_i.append( pt_list[ j ].pt[ i ][ 0 ] )
			pt_list_rad_i.append( pt_list[ j ].pt[ i ][ 1 ] )

		t_list_pos_i = list( t_list )
		t_list_rad_i = list( t_list )		

		base_pos_i = base.pt[ i ][ 0 ]
		tangent_pos_i = tangent.tVector[ i ][ 0 ]

		base_rad_i = base.pt[ i ][ 1 ]
		tangent_rad_i = tangent.tVector[ i ][ 1 ]

		RMSE_pos_i = RootMeanSquaredError( t_list_pos_i, pt_list_pos_i, base_pos_i, tangent_pos_i )
		RMSE_rad_i = RootMeanSquaredError( t_list_rad_i, pt_list_rad_i, base_rad_i, tangent_rad_i ) 

		RMSE_pos_atom.append( RMSE_pos_i )
		RMSE_rad_atom.append( RMSE_rad_i )

	RMSE_atom = [ RMSE_pos_atom, RMSE_rad_atom ]

	return RMSE_atom



def NullHypothesisTestingPermutationTest( t_list, pt_list, base, tangent, nTrial = 10000, max_iter = 500, stepSize = 0.05, step_tol = 1e-8 ):
	if base.Type == "Sphere":
		return NullHypothesisTestingPermutationTest_Sphere( t_list, pt_list, base, tangent, nTrial, max_iter, stepSize, step_tol )
	elif base.Type == "PositiveReal":
		return NullHypothesisTestingPermutationTest_PosReal( t_list, pt_list, base, tangent, nTrial, max_iter, stepSize, step_tol )
	# elif base.Type == "Euclidean":
	# 	R2Statistics_Euclidean( t_list, pt_list, base, tangent )
	else:
		print( "Manifold Type Unknown" )
		return -1 

def NullHypothesisTestingPermutationTest_Sphere( t_list, pt_list, base, tangent, nTrial = 10000, max_iter = 500, stepSize = 0.05, step_tol = 1e-8 ):
	# Estimated R2
	R2_est = R2Statistics( t_list, pt_list, base, tangent )

	cnt_greater_R2 = 0 

	for i in range( nTrial ):
		t_list_permuted = list( t_list )
		shuffle( t_list_permuted )

		base_i, tangent_i = GeodesicRegression( t_list_permuted, pt_list, max_iter, stepSize, step_tol, False )

		R2_i = R2Statistics( t_list_permuted, pt_list, base_i, tangent_i )

		if R2_i > R2_est:
			cnt_greater_R2 += 1

	return float( cnt_greater_R2 ) / float( nTrial )


def NullHypothesisTestingPermutationTest_PosReal( t_list, pt_list, base, tangent, nTrial = 10000, max_iter = 500, stepSize = 0.05, step_tol = 1e-8 ):
	# Estimated R2
	R2_est = R2Statistics( t_list, pt_list, base, tangent )

	cnt_greater_R2 = 0 

	for i in range( nTrial ):
		t_list_permuted = list( t_list )
		shuffle( t_list_permuted )

		base_i, tangent_i = GeodesicRegression( t_list_permuted, pt_list, max_iter, stepSize, step_tol, False )

		R2_i = R2Statistics( t_list_permuted, pt_list, base_i, tangent_i )

		if R2_i > R2_est:
			cnt_greater_R2 += 1

	return float( cnt_greater_R2 ) / float( nTrial )


##########################################################################################
###               Multivariate Anchor Point Linearized Geodesic Regression             ###
##########################################################################################

def MultivariateLinearizedGeodesicRegression( X, Y, VG, max_iter = 500, stepSize = 0.05, step_tol = 0.01, useFrechetMeanAnchor = False, verbose=False ):
	if pt_list[ 0 ].Type == "Sphere":
		return MultivariateLinearizedGeodesicRegression_Sphere( X, Y, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	# elif pt_list[ 0 ].Type == "PositiveReal":
	# 	return LinearizedGeodesicRegression_PosReal( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	# elif pt_list[ 0 ].Type == "Euclidean":
	# 	return LinearizedGeodesicRegression_Euclidean( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	# elif pt_list[ 0 ].Type == "CMRep":
	# 	return LinearizedGeodesicRegression_CMRep( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	# elif pt_list[ 0 ].Type == "CMRep_Abstract":
	# 	return LinearizedGeodesicRegression_CMRep_Abstract( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	else:
		print( "Manifold type is not known" )
		print( "Or a function is not ready, mb" )		
		return -1

def MultivariateLinearizedGeodesicRegression_Sphere( X, Y, VG, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

		print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
		print( "No. Observations : " + str( len( X ) ) )

	nData = len( Y )
	nParam = len( X[ 0 ] )

	# Continuous variable such as age should be the last entry of independent variables
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	# Initialize an anchor point 	
	if useFrechetMeanAnchor:
		p_anchor = FrechetMean( Y ) 
	else:
		t_min_idx = np.argmin( t_list )		

		p_anchor = Y[ t_min_idx ]

	nManifoldDim = p_anchor.nDim 

	# Initial intercept point
	init_Interp = manifolds.sphere( nManifoldDim )

	# Initial set of tangent vectors
	init_tVec_arr = [] 

	for i in range( nParam ):
		init_tVec_arr.append( manifolds.sphere_tVec( nManifoldDim ) )

	base = init_Interp
	tangent_arr = init_tVec_arr

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prev_tVec_arr = tangent_arr

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( Y[ j ] )

			for d in range( nManifoldDim ):
				w_list[d].append( tVec_j.tVector[d] )

		estModel_list = []

		for d in range( nManifoldDim ):
			X_sm = sm.add_constant( X )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, X_sm )
			# est_d = LS_model_d.fit(method='qr')

			est_d = LS_model_d.fit()
			estModel_list.append( est_d )

			if verbose:
				print( est_d.summary() )

		# Intercept point
		v_to_base_on_p_anchor = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		print( "Anchor point to intercept" )
		print( v_to_base_on_p_anchor.tVector )

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		new_tVec_arr = []

		for par in range( nParam ):
			v_tangent_on_p_anchor_param = manifolds.sphere_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

			newTangent_param = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor_param )	
			new_tVec_arr.append( newTangent_param )

		# Calculate energy to check if the model was minimized
		energy = 0

		for n in range( nData ):
			target = Y[ n ] 

			current_tangent_VG_intercept = manifolds.sphere_tVec( nManifoldDim ) 
			current_tangent_VG_slope = manifolds.sphere_tVec( nManifoldDim )  

			for d in range( nManifoldDim ):
				current_tangent_VG_slope.tVector[ d ] = 0
				current_tangent_VG_intercept.tVector[ d ] = 0


			for par in range( nParam ): 
				# Intercept
				if VG[ par ] == 0:
					for d in range( nManifoldDim ):
						current_tangent_VG_intercept.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )
				# Slope
				elif VG[ par ] == 1:
					for d in range( nManifoldDim ):
						current_tangent_VG_slope.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

			intercept_n = newBase.ExponentialMap( current_tangent_VG_intercept )

			slope_n = newBase.ParallelTranslateAtoB( newBase, intercept_n, current_tangent_VG_slope )

			estimate_n = intercept_n.ExponentialMap( slope_n )

			et = estimate_n.LogMap( target )

			# Energy of the tangential error
			energy += et.normSquared()


		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )
			break;
		else:
			prevBase = newBase
			prev_tVec_arr = new_tVec_arr
			p_anchor = newBase
			base = newBase
			tangent_arr = new_tVec_arr
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent_arr


def MultivariateLinearizedGeodesicRegression_Sphere_Additive( X, Y, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

		print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
		print( "No. Observations : " + str( len( X ) ) )

	nData = len( Y )
	nParam = len( X[ 0 ] )

	# Continuous variable such as age should be the last entry of independent variables
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	# Initialize an anchor point 	
	if useFrechetMeanAnchor:
		p_anchor = FrechetMean( Y ) 
	else:
		t_min_idx = np.argmin( t_list )		

		p_anchor = Y[ t_min_idx ]

	nManifoldDim = p_anchor.nDim 

	# Initial intercept point
	init_Interp = manifolds.sphere( nManifoldDim )

	# Initial set of tangent vectors
	init_tVec_arr = [] 

	for i in range( nParam ):
		init_tVec_arr.append( manifolds.sphere_tVec( nManifoldDim ) )

	base = init_Interp
	tangent_arr = init_tVec_arr

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prev_tVec_arr = tangent_arr

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( Y[ j ] )

			for d in range( nManifoldDim ):
				w_list[d].append( tVec_j.tVector[d] )

		estModel_list = []

		for d in range( nManifoldDim ):
			X_sm = sm.add_constant( X )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, X_sm )
			# est_d = LS_model_d.fit(method='qr')

			est_d = LS_model_d.fit()
			estModel_list.append( est_d )

			if verbose:
				print( est_d.summary() )

		# Intercept point
		v_to_base_on_p_anchor = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		print( "Anchor poin t to intercept" )
		print( v_to_base_on_p_anchor.tVector )

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		new_tVec_arr = []

		for par in range( nParam ):
			v_tangent_on_p_anchor_param = manifolds.sphere_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

			newTangent_param = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor_param )	
			new_tVec_arr.append( newTangent_param )

		# Calculate energy to check if the model was minimized
		energy = 0

		for n in range( nData ):
			target = Y[ n ] 

			current_tangent = manifolds.sphere_tVec( nManifoldDim ) 

			for d in range( nManifoldDim ):
				current_tangent.tVector[ d ] = 0


			for par in range( nParam ): 
				for d in range( nManifoldDim ):
					current_tangent.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )
			
			estimate_n = newBase.ExponentialMap( current_tangent )

			et = estimate_n.LogMap( target )

			# Energy of the tangential error
			energy += et.normSquared()


		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )
			break;
		else:
			prevBase = newBase
			prev_tVec_arr = new_tVec_arr
			p_anchor = newBase
			base = newBase
			tangent_arr = new_tVec_arr
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )

		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent_arr

def MultivariateLinearizedGeodesicRegression_Intercept_Sphere( X, Y, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )

	nData = len( Y )
	nParam = len( X[ 0 ] )

	# Anchor point is chosen by the last entry of covariates
	# Continuous variable such as a genetic disease score should be the last entry of covariates
	# If data don't have a continuous covariates, the last entry can be a categorical covariate
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	# Set an anchor point
	t_min_idx = np.argmin( t_list )		
	p_anchor = Y[ t_min_idx ]

	nManifoldDim = p_anchor.nDim 

	# Initial intercept point
	init_Interp = manifolds.sphere( nManifoldDim )

	# Initial set of tangent vectors
	init_tVec_arr = [] 

	for i in range( nParam ):
		init_tVec_arr.append( manifolds.sphere_tVec( nManifoldDim ) )

	base = init_Interp
	tangent_arr = init_tVec_arr

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prev_tVec_arr = tangent_arr

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( Y[ j ] )

			for d in range( nManifoldDim ):
				w_list[d].append( tVec_j.tVector[d] )

		estModel_list = []

		for d in range( nManifoldDim ):
			print( "X")
			print( X )

			X_sm = sm.add_constant( X )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, X_sm )
			# est_d = LS_model_d.fit(method='qr')

			est_d = LS_model_d.fit()
			estModel_list.append( est_d )

			if verbose:
				print( est_d.summary() )

		# Intercept point
		v_to_base_on_p_anchor = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		print( "Anchor point to intercept" )
		print( v_to_base_on_p_anchor.tVector )

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		new_tVec_arr = []

		for par in range( nParam ):
			v_tangent_on_p_anchor_param = manifolds.sphere_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

			newTangent_param = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor_param )	
			new_tVec_arr.append( newTangent_param )

		# Calculate energy to check if the model was minimized
		energy = 0

		for n in range( nData ):
			target = Y[ n ] 

			current_tangent_VG_intercept = manifolds.sphere_tVec( nManifoldDim ) 
			current_tangent_VG_slope = manifolds.sphere_tVec( nManifoldDim )  

			for d in range( nManifoldDim ):
				current_tangent_VG_slope.tVector[ d ] = 0
				current_tangent_VG_intercept.tVector[ d ] = 0

			tangent_t_n = manifolds.sphere_tVec( nManifoldDim )


			for par in range( nParam ): 
				for d in range( nManifoldDim ):
					tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

			estimate_n = newBase.ExponentialMap( tangent_t_n )

			et = estimate_n.LogMap( target )

			# Energy of the tangential error
			energy += et.normSquared()

		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )
			break;
		else:
			prevBase = newBase
			prev_tVec_arr = new_tVec_arr
			p_anchor = newBase
			base = newBase
			tangent_arr = new_tVec_arr
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent_arr


def MultivariateLinearizedGeodesicRegression_Slope_Sphere( X, Y, beta0, p0_list, tVec_intercept_arr, cov_intercept_list, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	if len( X ) == 0:
		nManifoldDim = beta0.nDim

		slope_tVec = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			slope_tVec.tVector[ d ] = 0

		print( len( Y ) )

		for i in range( len( Y ) ):
			Y_i = Y [ i ]

			if i == 0:
				Y_i_tilde = Y_i
			else:				
				beta_tVec_f_i = manifolds.sphere_tVec( nManifoldDim )

				for tt in range( len( cov_intercept_list[ i ] ) ):
					est_beta_tt = tVec_intercept_arr[ tt ]

					for kk in range( nManifoldDim ):
						beta_tVec_f_i.tVector[ kk ] += ( est_beta_tt.tVector[ kk ] * cov_intercep_list[ i ][ tt ] )

				f_i = beta0.ExponentialMap( est_beta_tt )				

				Y_i_at_f_i = p0_list[ i ].ParallelTranslateAtoB( p0_list[i], f_i, Y_i ) 
				Y_i_tilde = Y_i_at_f_i.ParallelTranslateAtoB( f_i, beta0, Y_i )

			print( "Y_i")
			print( Y_i.tVector )			
			print( "Y_i_tilde")
			print( Y_i_tilde.tVector )	

			for d in range( nManifoldDim ):
				slope_tVec.tVector[ d ] += ( Y_i_tilde.tVector[ d ] / float( len( Y ) ) )

		init_slope_tVec = slope_tVec

		# Gradient Descent with eps
		eps = 0.0001
		stepSize = 0.01
		stepTol = 1e-8 
		resTol = 1e-6
		nIter = 500

		prev_energy = 0

		for i in range( len( Y ) ):
			beta_tVec_f_i = manifolds.sphere_tVec( nManifoldDim )

			for tt in range( len( cov_intercept_list[ i ] ) ):
				est_beta_tt = tVec_intercept_arr[ tt ]

				for kk in range( nManifoldDim ):
					beta_tVec_f_i.tVector[ kk ] += ( est_beta_tt.tVector[ kk ] * cov_intercep_list[ i ][ tt ] )

			f_i = beta0.ExponentialMap( est_beta_tt )				

			slope_at_f_i = beta0.ParallelTranslateAtoB( beta0, f_i, slope_tVec )
			slope_at_p_i = beta0.ParallelTranslateAtoB( f_i, p0_list[ i ], slope_at_f_i )

			prev_energy_i = 0

			for d in range( nManifoldDim ):
				prev_energy_i += ( slope_at_p_i.tVector[ d ] - Y_i.tVector[ d ] )**2.0

			prev_energy += prev_energy_i

		energy_arr = []

		for k in range( nIter ):
			slope_tVec_updated = manifolds.sphere_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ]

			# Calculate Gradient
			dE = [ 0, 0, 0 ]

			energy_k = 0

			# Calculate FDM
			for d in range( nManifoldDim ):
				slope_pos_eps = manifolds.sphere_tVec( nManifoldDim )
				slope_neg_eps = manifolds.sphere_tVec( nManifoldDim )

				for dd in range( nManifoldDim ):
					slope_pos_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 
					slope_neg_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 

				slope_pos_eps.tVector[ d ] = slope_tVec.tVector[ d ] + eps
				slope_neg_eps.tVector[ d ] = slope_tVec.tVector[ d ] - eps

				for i in range( len( Y ) ):
					Y_i = Y[ i ]

					slope_parT_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

					slope_pos_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_pos_eps )
					slope_neg_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_neg_eps )

					grad_slope_parT_fdm = manifolds.sphere_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						grad_slope_parT_fdm.tVector[ dd ] = float( slope_pos_eps_at_p_i.tVector[ dd ] - slope_neg_eps_at_p_i.tVector[ dd ] ) / float( 2.0 * eps )

					print( "slope_pos_eps" ) 
					print( slope_pos_eps.tVector )
					print( "slope_neg_eps" ) 
					print( slope_neg_eps.tVector )

					print( "slope_pos_eps_p_i" ) 
					print( slope_pos_eps_at_p_i.tVector )
					print( "slope_neg_eps_p_i" ) 
					print( slope_neg_eps_at_p_i.tVector )

					print( "FDM tVector" )
					print( grad_slope_parT_fdm.tVector ) 


					slope_parT_minus_Y_i = manifolds.sphere_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						slope_parT_minus_Y_i.tVector[ dd ] = slope_parT_p_i.tVector[ dd ] - Y_i.tVector[ dd ]

					dE[ d ] += grad_slope_parT_fdm.InnerProduct( slope_parT_minus_Y_i )


				print( "dE[ d ] " )				
				print( dE[ d ] )

				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ] - ( stepSize * dE[ d ] )

			# Calculate Energy
			for i in range( len( Y ) ):
				Y_i = Y[ i ]
				slope_tVec_updated_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec_updated )

				energy_k_i = 0
				for d in range( nManifoldDim ):
					energy_k_i += ( slope_tVec_updated_at_p_i.tVector[ d ] - Y_i.tVector[ d ] ) ** 2

				energy_k += energy_k_i

			if energy_k > prev_energy:
				print( "Iteration : " + str( k + 1 ) )
				print( "Energy Increased : Halve step size")
				print( "Prev. Residual Energy" )
				print( prev_energy )

				energy_k = prev_energy

				energy_arr.append( energy_k )

				stepSize = stepSize / 2 
			else:
				print( "Iteration : " + str( k + 1 ) )
				print( "Residual Energy" )
				print( energy_k )
							
				stepSize = stepSize * 1.5
				slope_tVec = slope_tVec_updated
				prev_energy = energy_k
				energy_arr.append( energy_k )

			if energy_k < resTol:
				print( "Energy Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if stepSize < stepTol:
				slope_tVec = slope_tVec_updated
				print( "Step Size Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if k == nIter- 1:
				slope_tVec = slope_tVec_updated

		print( "Initial Slope" ) 
		print( init_slope_tVec.tVector )
		print( "Updated Slope" )
		print( slope_tVec.tVector )
		print( "Residual Energy" )
		print( energy_k )

		tangent_arr = []
		tangent_arr.append( slope_tVec )

		plt.figure()
		plt.plot( np.linspace( 1, k+1, num=k+1 ), energy_arr )
		plt.show()

		return tangent_arr

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )
	nData = len( Y )
	nParam = len( X[ 0 ] )

	# Anchor point is chosen by the last entry of covariates
	# Continuous variable such as a genetic disease score should be the last entry of covariates
	# If data don't have a continuous covariates, the last entry can be a categorical covariate
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	print( t_list )

	p_anchor = beta0

	nManifoldDim = p_anchor.nDim 

	tVec_list = []
	w_list = []

	for d in range( nManifoldDim ):
		w_list.append( [] )		

	for j in range( nData ):
		Y_j = Y[ j ]
		# Parallel translate a group-wise tangent vector to population-level intercept 
		beta_tVec_f_i = manifolds.sphere_tVec( nManifoldDim )

		for tt in range( len( cov_intercept_list[ j ] ) ):
			est_beta_tt = tVec_intercept_arr[ tt ]

			for kk in range( nManifoldDim ):
				beta_tVec_f_i.tVector[ kk ] += ( est_beta_tt.tVector[ kk ] * cov_intercept_list[ j ][ tt ] )

		f_j = beta0.ExponentialMap( est_beta_tt )				
		Y_j_at_f_j = p0_list[ j ].ParallelTranslateAtoB( p0_list[ j ], f_j, Y_j ) 
		Y_j_tilde = f_j.ParallelTranslateAtoB( f_j, beta0, Y_j_at_f_j )

		tVec_j = Y_j_tilde

		for d in range( nManifoldDim ):
			w_list[d].append( tVec_j.tVector[d] )

	estModel_list = []

	for d in range( nManifoldDim ):
		X_sm = sm.add_constant( X )
		w_d_np = np.asarray( w_list[ d ] )
		LS_model_d = sm.OLS( w_d_np, X_sm )
		# est_d = LS_model_d.fit(method='qr')

		est_d = LS_model_d.fit()
		estModel_list.append( est_d )

		# if verbose:
		# 	print( est_d.summary() )

	# Intercept point
	v_t = manifolds.sphere_tVec( nManifoldDim )

	for d in range( nManifoldDim ):
		v_t.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

	# print( "Anchor point to intercept" )
	# print( v_to_base_on_p_anchor.tVector )

	# newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
	new_tVec_arr = []

	for par in range( nParam ):
		v_tangent_on_p_anchor_param = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

		new_tVec_arr.append( v_tangent_on_p_anchor_param )

	# Append time-wise slope tangent vector at the last
	new_tVec_arr.append( v_t )
	tangent_arr = new_tVec_arr 

	# # Calculate energy to check if the model was minimized
	# energy = 0

	# for n in range( nData ):
	# 	target = Y[ n ] 

	# 	tangent_t_n = manifolds.sphere_tVec( nManifoldDim )

	# 	for par in range( nParam ): 
	# 		for d in range( nManifoldDim ):
	# 			tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

	# 	estimate_n = p_anchor.ExponentialMap( tangent_t_n )

	# 	et = estimate_n.LogMap( target )

	# 	# Energy of the tangential error
	# 	energy += et.normSquared()

	# 	tangent_arr = new_tVec_arr
	# 	if verbose:
	# 		print( "==================================" )
	# 		print( "Residual Energy " ) 
	# 		print( energy )
	# 		print( "==================================" )

	return tangent_arr


def MultivariateLinearizedGeodesicRegression_Slope_DirectTransport_Sphere( X, Y, beta0, p0_list, tVec_intercept_arr, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	if len( X ) == 0:
		nManifoldDim = beta0.nDim

		slope_tVec = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			slope_tVec.tVector[ d ] = 0

		print( len( Y ) )

		for i in range( len( Y ) ):
			Y_i = Y [ i ]

			if i == 0:
				Y_i_tilde = Y_i
			else:
				Y_i_tilde = p0_list[ i ].ParallelTranslateAtoB( p0_list[i], beta0, Y_i )

			print( "Y_i")
			print( Y_i.tVector )			
			print( "Y_i_tilde")
			print( Y_i_tilde.tVector )	

			for d in range( nManifoldDim ):
				slope_tVec.tVector[ d ] += ( Y_i_tilde.tVector[ d ] / float( len( Y ) ) )

		init_slope_tVec = slope_tVec

		# Gradient Descent with eps
		eps = 0.0001
		stepSize = 0.01
		stepTol = 1e-8 
		resTol = 1e-6
		nIter = 500

		prev_energy = 0

		for i in range( len( Y ) ):
			slope_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

			prev_energy_i = 0

			for d in range( nManifoldDim ):
				prev_energy_i += ( slope_at_p_i.tVector[ d ] - Y_i.tVector[ d ] )**2.0

			prev_energy += prev_energy_i

		energy_arr = []

		for k in range( nIter ):
			slope_tVec_updated = manifolds.sphere_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ]

			# Calculate Gradient
			dE = [ 0, 0, 0 ]

			energy_k = 0

			# Calculate FDM
			for d in range( nManifoldDim ):
				slope_pos_eps = manifolds.sphere_tVec( nManifoldDim )
				slope_neg_eps = manifolds.sphere_tVec( nManifoldDim )

				for dd in range( nManifoldDim ):
					slope_pos_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 
					slope_neg_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 

				slope_pos_eps.tVector[ d ] = slope_tVec.tVector[ d ] + eps
				slope_neg_eps.tVector[ d ] = slope_tVec.tVector[ d ] - eps

				for i in range( len( Y ) ):
					Y_i = Y[ i ]

					slope_parT_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

					slope_pos_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_pos_eps )
					slope_neg_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_neg_eps )

					grad_slope_parT_fdm = manifolds.sphere_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						grad_slope_parT_fdm.tVector[ dd ] = float( slope_pos_eps_at_p_i.tVector[ dd ] - slope_neg_eps_at_p_i.tVector[ dd ] ) / float( 2.0 * eps )

					print( "slope_pos_eps" ) 
					print( slope_pos_eps.tVector )
					print( "slope_neg_eps" ) 
					print( slope_neg_eps.tVector )

					print( "slope_pos_eps_p_i" ) 
					print( slope_pos_eps_at_p_i.tVector )
					print( "slope_neg_eps_p_i" ) 
					print( slope_neg_eps_at_p_i.tVector )

					print( "FDM tVector" )
					print( grad_slope_parT_fdm.tVector ) 


					slope_parT_minus_Y_i = manifolds.sphere_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						slope_parT_minus_Y_i.tVector[ dd ] = slope_parT_p_i.tVector[ dd ] - Y_i.tVector[ dd ]

					dE[ d ] += grad_slope_parT_fdm.InnerProduct( slope_parT_minus_Y_i )


				print( "dE[ d ] " )				
				print( dE[ d ] )

				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ] - ( stepSize * dE[ d ] )

			# Calculate Energy
			for i in range( len( Y ) ):
				Y_i = Y[ i ]
				slope_tVec_updated_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec_updated )

				energy_k_i = 0
				for d in range( nManifoldDim ):
					energy_k_i += ( slope_tVec_updated_at_p_i.tVector[ d ] - Y_i.tVector[ d ] ) ** 2

				energy_k += energy_k_i

			if energy_k > prev_energy:
				print( "Iteration : " + str( k + 1 ) )
				print( "Energy Increased : Halve step size")
				print( "Prev. Residual Energy" )
				print( prev_energy )

				energy_k = prev_energy

				energy_arr.append( energy_k )

				stepSize = stepSize / 2 
			else:
				print( "Iteration : " + str( k + 1 ) )
				print( "Residual Energy" )
				print( energy_k )
							
				stepSize = stepSize * 1.5
				slope_tVec = slope_tVec_updated
				prev_energy = energy_k
				energy_arr.append( energy_k )

			if energy_k < resTol:
				print( "Energy Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if stepSize < stepTol:
				slope_tVec = slope_tVec_updated
				print( "Step Size Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if k == nIter- 1:
				slope_tVec = slope_tVec_updated

		print( "Initial Slope" ) 
		print( init_slope_tVec.tVector )
		print( "Updated Slope" )
		print( slope_tVec.tVector )
		print( "Residual Energy" )
		print( energy_k )

		tangent_arr = []
		tangent_arr.append( slope_tVec )

		plt.figure()
		plt.plot( np.linspace( 1, k+1, num=k+1 ), energy_arr )
		plt.show()

		return tangent_arr

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )
	nData = len( Y )
	nParam = len( X[ 0 ] )

	# Anchor point is chosen by the last entry of covariates
	# Continuous variable such as a genetic disease score should be the last entry of covariates
	# If data don't have a continuous covariates, the last entry can be a categorical covariate
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	print( t_list )

	p_anchor = beta0

	nManifoldDim = p_anchor.nDim 

	tVec_list = []
	w_list = []

	for d in range( nManifoldDim ):
		w_list.append( [] )		

	for j in range( nData ):
		# Parallel translate a group-wise tangent vector to population-level intercept 
		tVec_j = p0_list[ j ].ParallelTranslateAtoB( p0_list[ j ], p_anchor, Y[ j ] )

		for d in range( nManifoldDim ):
			w_list[d].append( tVec_j.tVector[d] )

	estModel_list = []

	for d in range( nManifoldDim ):
		X_sm = sm.add_constant( X )
		w_d_np = np.asarray( w_list[ d ] )
		LS_model_d = sm.OLS( w_d_np, X_sm )
		# est_d = LS_model_d.fit(method='qr')

		est_d = LS_model_d.fit()
		estModel_list.append( est_d )

		# if verbose:
		# 	print( est_d.summary() )

	# Intercept point
	v_t = manifolds.sphere_tVec( nManifoldDim )

	for d in range( nManifoldDim ):
		v_t.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

	# print( "Anchor point to intercept" )
	# print( v_to_base_on_p_anchor.tVector )

	# newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
	new_tVec_arr = []

	for par in range( nParam ):
		v_tangent_on_p_anchor_param = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

		new_tVec_arr.append( v_tangent_on_p_anchor_param )

	# Append time-wise slope tangent vector at the last
	new_tVec_arr.append( v_t )
	tangent_arr = new_tVec_arr 

	# # Calculate energy to check if the model was minimized
	# energy = 0

	# for n in range( nData ):
	# 	target = Y[ n ] 

	# 	tangent_t_n = manifolds.sphere_tVec( nManifoldDim )

	# 	for par in range( nParam ): 
	# 		for d in range( nManifoldDim ):
	# 			tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

	# 	estimate_n = p_anchor.ExponentialMap( tangent_t_n )

	# 	et = estimate_n.LogMap( target )

	# 	# Energy of the tangential error
	# 	energy += et.normSquared()

	# 	tangent_arr = new_tVec_arr
	# 	if verbose:
	# 		print( "==================================" )
	# 		print( "Residual Energy " ) 
	# 		print( energy )
	# 		print( "==================================" )

	return tangent_arr

def MultivariateLinearizedGeodesicRegression_Sphere_BottomUp( t_list, pt_list, cov_intercept_list, cov_slope_list=[], max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# The numbers
	nGroup = len( t_list )

	nData_group = [] 
	for i in range( nGroup ):
		nData_group.append( len( t_list[ i ] ) )

	nParam_int = len( cov_intercept_list[ 0 ] )

	nParam_slope = 0	
	if not len( cov_slope_list ) == 0:
		nParam_slope = len( cov_slope_list[ 0 ] )

	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

		print( "No. Group : " + str( nGroup ) )

		for i in range( nGroup ):
			print( "Group " + str( i + 1 ) + " : " + str( nData_group[ i ] ) + " Obs." )
		print( "No. Covariates for Intercept: " + str( nParam_int ) )
		print( "No. Covariates for Slope: " + str( nParam_slope ) )

	# Group-wise intercept, slope tangent vector, covariates (intercept/slope), time 
	p0_group_list = [] # 1-D Array N x 1
	v_group_list = []  # 1-D Array N x 1 
	cov_intercept_group_list = [] # 2-D Array N x C_int
	cov_slope_group_list = [] # 2-D Array N x C_slope
	t_group_list = []   # 2-D Array N x O

	for g in range( nGroup ):
		t_list_g = t_list[ g ]
		pt_list_g = pt_list[ g ] 

		p0_g, v_g = LinearizedGeodesicRegression_Sphere( t_list_g, pt_list_g )

		print( "v_g.tVector" )		
		print( v_g.tVector )

		p0_group_list.append( p0_g )
		v_group_list.append( v_g )

		cov_intercept_group_list.append( cov_intercept_list[ g ] )

		if not len( cov_slope_list ) == 0:
			cov_slope_group_list.append( cov_slope_list[ g ] )

	##############################################	
	## Solve Intercepts Points w.r.t Covariates ##
	##############################################
	beta0, tangent_intercept_arr = MultivariateLinearizedGeodesicRegression_Intercept_Sphere( cov_intercept_group_list, p0_group_list, verbose=verbose )

	##############################################	
	## Solve Tangent Vectors w.r.t Covariates ##
	##############################################

	print( len ( cov_slope_group_list ) )
	print( len ( v_group_list ) )

	print( "cov_slope_group_list" )
	print( cov_slope_group_list )

	print( "v_group_list" )
	print( v_group_list[ 0 ].tVector )
	print( v_group_list[ 1 ].tVector )

	tangent_slope_arr = MultivariateLinearizedGeodesicRegression_Slope_Sphere( cov_slope_group_list, v_group_list, beta0, p0_group_list, tangent_intercept_arr, cov_intercept_group_list, verbose=verbose )

	return beta0, tangent_intercept_arr, tangent_slope_arr


def MultivariateLinearizedGeodesicRegression_Slope_Sphere_PoorSasaki( X, Y, beta0, p0_list, tVec_intercept_arr, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	if len( X ) == 0:
		nManifoldDim = beta0.nDim

		slope_tVec = manifolds.sphere_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			slope_tVec.tVector[ d ] = 0

		print( len( Y ) )

		L = 1000

		beta0_l_1 = beta0
		v0_l_1 = manifolds.sphere_tVec( nManifoldDim )

		for i in range( len( Y ) ):
			Y_i = Y[ i ]

			Y_i_l_1 = Y_i.ScalarMultiply( 1.0 / float( L ) )

			Y_i_l_1_parT = p0_list[ i ].ParallelTranslateAtoB( p0_list[ i ], beta0_l_1, Y_i_l_1 )

			for d in range( nManifoldDim ):
				v0_l_1.tVector[ d ] += ( Y_i_l_1_parT.tVector[ d ] / float( len( Y ) ) )
		
		g0_list = [ beta0_l_1 ] 
		v0_list = [ v0_l_1 ]
		t0_list = [ 0 ]

		for l in range( L ):
			beta0_l = beta0_l_1.ExponentialMap( v0_l_1 )

			g0_list.append( beta0_l ) 
			t0_list.append( float( l + 1 ) / float( L ) )

			v0_l = manifolds.sphere_tVec( nManifoldDim )

			for i in range( len( Y ) ):
				Y_i = Y[ i ]

				p_i_l = p0_list[ i ].ExponentialMap( Y_i.ScalarMultiply( float( l + 1 ) / float( L ) ) )


				Y_i_l = p0_list[ i ].ParallelTranslateAtoB( p0_list[ i ], p_i_l, Y_i.ScalarMultiply( 1.0 / float( L ) ) )

				Y_i_l_parT = p_i_l.ParallelTranslateAtoB( p_i_l, beta0, Y_i_l )

				for d in range( nManifoldDim ):
					v0_l.tVector[ d ] += ( Y_i_l_parT.tVector[ d ] / float( len( Y ) ) )

			v0_list.append( v0_l )

			beta0_l_1 = beta0_l
			v0_l_1 = v0_l

		p, v0 = LinearizedGeodesicRegression_Sphere( t0_list, g0_list )

		tangent_arr = [ ]
		tangent_arr.append( v0 )

		return p, tangent_arr

	return 0

def MultivariateLinearizedGeodesicRegression_Sphere_BottomUp_PoorSasaki( t_list, pt_list, cov_intercept_list, cov_slope_list=[], max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# The numbers
	nGroup = len( t_list )

	nData_group = [] 
	for i in range( nGroup ):
		nData_group.append( len( t_list[ i ] ) )

	nParam_int = len( cov_intercept_list[ 0 ][ 0 ] )

	nParam_slope = 0	
	if not len( cov_slope_list ) == 0:
		nParam_slope = len( cov_slope_list[ 0 ][ 0 ] )

	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

		print( "No. Group : " + str( nGroup ) )

		for i in range( nGroup ):
			print( "Group " + str( i + 1 ) + " : " + str( nData_group[ i ] ) + " Obs." )
		print( "No. Covariates for Intercept: " + str( nParam_int ) )
		print( "No. Covariates for Slope: " + str( nParam_slope ) )


	# Group-wise intercept, slope tangent vector, covariates (intercept/slope), time 
	p0_group_list = [] # 1-D Array N x 1
	v_group_list = []  # 1-D Array N x 1 
	cov_intercept_group_list = [] # 2-D Array N x C_int
	cov_slope_group_list = [] # 2-D Array N x C_slope
	t_group_list = []   # 2-D Array N x O

	for g in range( nGroup ):
		t_list_g = t_list[ g ]
		pt_list_g = pt_list[ g ] 

		p0_g, v_g = LinearizedGeodesicRegression_Sphere( t_list_g, pt_list_g )

		print( "v_g.tVector" )		
		print( v_g.tVector )

		p0_group_list.append( p0_g )
		v_group_list.append( v_g )

		cov_intercept_group_list.append( cov_intercept_list[ g ][ 0 ] )

		if not len( cov_slope_list ) == 0:
			cov_slope_group_list.append( cov_slope_list[ g ][ 0 ] )

	##############################################	
	## Solve Intercepts Points w.r.t Covariates ##
	##############################################

	beta0, tangent_intercept_arr = MultivariateLinearizedGeodesicRegression_Intercept_Sphere( cov_intercept_group_list, p0_group_list, verbose=verbose )

	##############################################	
	## Solve Tangent Vectors w.r.t Covariates ##
	##############################################

	print( len (cov_slope_group_list ) )
	print( len (v_group_list ) )
	
	beta0, tangent_slope_arr = MultivariateLinearizedGeodesicRegression_Slope_Sphere_PoorSasaki( cov_slope_group_list, v_group_list, beta0, p0_group_list, tangent_intercept_arr, verbose=verbose )

	return beta0, tangent_intercept_arr, tangent_slope_arr

##############################################################
##  			2D Scale Kendall Shape space 				##
##############################################################
def MultivariateLinearizedGeodesicRegression_ScaleKendall2D_BottomUp( t_list, pt_list, cov_intercept_list, cov_slope_list=[], max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	pt_shape_list = []
	pt_scale_list = []

	for i in range( len( pt_list ) ):
		pt_shape_list_i = []
		pt_scale_list_i = []

		for j in range( len( pt_list[ i ] ) ):
			pt_scale_list_i.append( pt_list[ i ][ j ].pt[ 0 ] )
			pt_shape_list_i.append( pt_list[ i ][ j ].pt[ 1 ] )

		pt_scale_list.append( pt_scale_list_i )
		pt_shape_list.append( pt_shape_list_i )

	beta0_scale, tangent_scale_intercept_arr, tangent_scale_slope_arr = MultivariateLinearizedGeodesicRegression_Euclidean_BottomUp( t_list, pt_scale_list, cov_intercept_list, cov_slope_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	beta0_kShape, tangent_kShape_intercept_arr, tangent_kShape_slope_arr = MultivariateLinearizedGeodesicRegression_Kendall2D_BottomUp( t_list, pt_shape_list, cov_intercept_list, cov_slope_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )

	beta0 = manifolds.scale_kendall2D( beta0_kShape.nPt )
	beta0.SetPoint( [ beta0_scale, beta0_kShape ] )

	tangent_intercept_arr = []
	tangent_slope_arr = []

	for i in range( len( tangent_kShape_intercept_arr ) ):
		tangent_i = manifolds.scale_kendall2D_tVec( tangent_kShape_intercept_arr[ i ].nPt )
		tangent_i.SetTangentVector( [ tangent_scale_intercept_arr[ i ], tangent_kShape_intercept_arr[ i ] ] )
		tangent_intercept_arr.append( tangent_i )

	for i in range( len( tangent_kShape_slope_arr ) ):
		tangent_i = manifolds.scale_kendall2D_tVec( tangent_kShape_slope_arr[ i ].nPt )
		tangent_i.SetTangentVector( [ tangent_scale_slope_arr[ i ], tangent_kShape_slope_arr[ i ] ] )
		tangent_slope_arr.append( tangent_i )

	return beta0, tangent_intercept_arr, tangent_slope_arr


##############################################################
##  				2D Kendall Shape space 					##
##############################################################
def MultivariateLinearizedGeodesicRegression_Kendall2D_BottomUp( t_list, pt_list, cov_intercept_list, cov_slope_list=[], max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# The numbers
	nGroup = len( t_list )

	nData_group = [] 
	for i in range( nGroup ):
		nData_group.append( len( t_list[ i ] ) )

	nParam_int = len( cov_intercept_list[ 0 ] )

	nParam_slope = 0	
	if not len( cov_slope_list ) == 0:
		nParam_slope = len( cov_slope_list[ 0 ] )

	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

		print( "No. Group : " + str( nGroup ) )

		for i in range( nGroup ):
			print( "Group " + str( i + 1 ) + " : " + str( nData_group[ i ] ) + " Obs." )
		print( "No. Covariates for Intercept: " + str( nParam_int ) )
		print( "No. Covariates for Slope: " + str( nParam_slope ) )

	# Group-wise intercept, slope tangent vector, covariates (intercept/slope), time 
	p0_group_list = [] # 1-D Array N x 1
	v_group_list = []  # 1-D Array N x 1 
	cov_intercept_group_list = [] # 2-D Array N x C_int
	cov_slope_group_list = [] # 2-D Array N x C_slope
	t_group_list = []   # 2-D Array N x O

	for g in range( nGroup ):
		t_list_g = t_list[ g ]
		pt_list_g = pt_list[ g ] 

		p0_g, v_g = LinearizedGeodesicRegression( t_list_g, pt_list_g )

		p0_group_list.append( p0_g )
		v_group_list.append( v_g )
		cov_intercept_group_list.append( cov_intercept_list[ g ] )

		if not len( cov_slope_list ) == 0:
			cov_slope_group_list.append( cov_slope_list[ g ] )

		# # Check R2
		# mean_g = FrechetMean( pt_list[ g ] )

		# sqDist_SG_sum = 0 
		# sqVar_sum = 0

		# for i in range( len( pt_list[ g ] ) ):
		# 	p_i = pt_list_g[ i ]
		# 	t_i = t_list_g[ i ]

		# 	slope_t_i = v_g.ScalarMultiply( t_i )
		# 	est_p_i = p0_g.ExponentialMap( slope_t_i )

		# 	tVec_est_p_i_to_p_i = est_p_i.LogMap( p_i )

		# 	sqDist_i = tVec_est_p_i_to_p_i.normSquared() 

		# 	sqDist_SG_sum += sqDist_i

		# 	tVec_mean_to_p_n = mean_g.LogMap( p_i ) 

		# 	sqVar_n = tVec_mean_to_p_n.normSquared()

		# 	sqVar_sum += sqVar_n

		# R2_SG = 1 - ( sqDist_SG_sum / sqVar_sum )

		# print( "Subject : " + str( g ) )
		# print( str( nData_group[ g ] ) + " Obs." )
		# print( R2_SG )


	##############################################	
	## Solve Intercepts Points w.r.t Covariates ##
	##############################################
	beta0, tangent_intercept_arr = MultivariateLinearizedGeodesicRegression_Intercept_Kendall2D( cov_intercept_group_list, p0_group_list, verbose=verbose )

	##############################################	
	## Solve Tangent Vectors w.r.t Covariates ##
	##############################################
	tangent_slope_arr = MultivariateLinearizedGeodesicRegression_Slope_Kendall2D( cov_slope_group_list, v_group_list, beta0, p0_group_list, tangent_intercept_arr, cov_intercept_group_list, verbose=verbose )

	return beta0, tangent_intercept_arr, tangent_slope_arr

def MultivariateLinearizedGeodesicRegression_Intercept_Kendall2D( X, Y, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )

	nData = len( Y )
	nParam = len( X[ 0 ] )

	if nParam == 0:
		base = FrechetMean( Y )
		tangent_arr = [] 		

		return base, tangent_arr

	# Anchor point is chosen by the last entry of covariates
	# Continuous variable such as a genetic disease score should be the last entry of covariates
	# If data don't have a continuous covariates, the last entry can be a categorical covariate
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	# Set an anchor point
	t_min_idx = np.argmin( t_list )		
	p_anchor = Y[ t_min_idx ]

	nManifoldDim = p_anchor.nPt 

	# Initial intercept point
	init_Interp = manifolds.kendall2D( nManifoldDim )

	# Initial set of tangent vectors
	init_tVec_arr = [] 

	for i in range( nParam ):
		init_tVec_arr.append( manifolds.kendall2D_tVec( nManifoldDim ) )

	base = init_Interp
	tangent_arr = init_tVec_arr

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prev_tVec_arr = tangent_arr

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for k in range( 2 ):
			for d in range( nManifoldDim ):
				w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( Y[ j ] )

			for k in range( 2 ):
				for d in range( nManifoldDim ):
					w_list[ k * nManifoldDim + d].append( tVec_j.tVector[k, d] )

		estModel_list = []

		for k in range( 2 ):
			for d in range( nManifoldDim ):
				X_sm = sm.add_constant( X )
				w_d_np = np.asarray( w_list[ k * nManifoldDim + d ] )
				LS_model_d = sm.OLS( w_d_np, X_sm )
				# est_d = LS_model_d.fit(method='qr')

				est_d = LS_model_d.fit()
				estModel_list.append( est_d )

				if verbose:
					print( est_d.summary() )

		# Intercept point
		v_to_base_on_p_anchor = manifolds.kendall2D_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_to_base_on_p_anchor.tVector[ 0, d ] = estModel_list[ d ].params[ 0 ] 
			v_to_base_on_p_anchor.tVector[ 1, d ] = estModel_list[ nManifoldDim + d ].params[ 0 ] 

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		new_tVec_arr = []

		for par in range( nParam ):
			v_tangent_on_p_anchor_param = manifolds.kendall2D_tVec( nManifoldDim )

			for k in range( 2 ):
				for d in range( nManifoldDim ):
					v_tangent_on_p_anchor_param.tVector[ k, d ] = estModel_list[ k * nManifoldDim + d ].params[ par + 1 ]

			newTangent_param = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor_param )	
			new_tVec_arr.append( newTangent_param )

		# Calculate energy to check if the model was minimized
		energy = 0

		for n in range( nData ):
			target = Y[ n ] 

			current_tangent_VG_intercept = manifolds.kendall2D_tVec( nManifoldDim ) 
			current_tangent_VG_slope = manifolds.kendall2D_tVec( nManifoldDim )  

			tangent_t_n = manifolds.kendall2D_tVec( nManifoldDim )

			for par in range( nParam ):
				for k in range( 2 ):
					for d in range( nManifoldDim ):
						tangent_t_n.tVector[ k, d ] += ( new_tVec_arr[ par ].tVector[ k, d ] * X[ n ][ par ] )

			estimate_n = newBase.ExponentialMap( tangent_t_n )

			et = estimate_n.LogMap( target )

			# Energy of the tangential error
			energy += et.normSquared()

		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )
			break;
		else:
			prevBase = newBase
			prev_tVec_arr = new_tVec_arr
			p_anchor = newBase
			base = newBase
			tangent_arr = new_tVec_arr
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent_arr

def MultivariateLinearizedGeodesicRegression_Slope_Kendall2D( X, Y, beta0, p0_list, tVec_intercept_arr, cov_intercept_list, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	if len( X ) == 0 or len( X[ 0 ] ) == 0 :		
		nManifoldDim = beta0.nPt

		slope_tVec = manifolds.kendall2D_tVec( nManifoldDim )
		print( len( Y ) )

		for i in range( len( Y ) ):
			Y_i = Y [ i ]

			if i == 0:
				Y_i_tilde = Y_i
			else:
				Y_i_tilde = p0_list[ i ].ParallelTranslateAtoB( p0_list[i], beta0, Y_i )

			print( "Y_i")
			print( Y_i.tVector )			
			print( "Y_i_tilde")
			print( Y_i_tilde.tVector )	

			for k in range( 2 ):
				for d in range( nManifoldDim ):
					slope_tVec.tVector[ k, d ] += ( Y_i_tilde.tVector[ k, d ] / float( len( Y ) ) )

		init_slope_tVec = slope_tVec

		# Gradient Descent with eps
		eps = 0.0001
		stepSize = 0.01
		stepTol = 1e-8 
		resTol = 1e-6
		nIter = 500

		prev_energy = 0

		for i in range( len( Y ) ):
			slope_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

			prev_energy_i = 0

			for k in range( 2 ):
				for d in range( nManifoldDim ):
					prev_energy_i += ( slope_at_p_i.tVector[ k, d ] - Y_i.tVector[ k, d ] )**2.0

			prev_energy += prev_energy_i

		energy_arr = []

		for k in range( nIter ):
			slope_tVec_updated = manifolds.kendall2D_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ]

			# Calculate Gradient
			dE = np.zeros( 2, nManifoldDim )

			energy_k = 0

			# Calculate FDM
			for kkk in range( 2 ):
				for d in range( nManifoldDim ):
					slope_pos_eps = manifolds.kendall2D_tVec( nManifoldDim )
					slope_neg_eps = manifolds.kendall2D_tVec( nManifoldDim )

					for kk in range( 2 ):
						for dd in range( nManifoldDim ):
							slope_pos_eps.tVector[ kk, dd ] = slope_tVec.tVector[ kk, dd ] 
							slope_neg_eps.tVector[ kk, dd ] = slope_tVec.tVector[ kk, dd ] 

					slope_pos_eps.tVector[ kkk, d ] = slope_tVec.tVector[ kkk, d ] + eps
					slope_neg_eps.tVector[ kkk, d ] = slope_tVec.tVector[ kkk, d ] - eps

					for i in range( len( Y ) ):
						Y_i = Y[ i ]

						slope_parT_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

						slope_pos_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_pos_eps )
						slope_neg_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_neg_eps )

						grad_slope_parT_fdm = manifolds.kendall2D_tVec( nManifoldDim )

						for kk in range( 2 ):
							for dd in range( nManifoldDim ):
								grad_slope_parT_fdm.tVector[ kk, dd ] = float( slope_pos_eps_at_p_i.tVector[ kk, dd ] - slope_neg_eps_at_p_i.tVector[ kk, dd ] ) / float( 2.0 * eps )

						print( "slope_pos_eps" ) 
						print( slope_pos_eps.tVector )
						print( "slope_neg_eps" ) 
						print( slope_neg_eps.tVector )

						print( "slope_pos_eps_p_i" ) 
						print( slope_pos_eps_at_p_i.tVector )
						print( "slope_neg_eps_p_i" ) 
						print( slope_neg_eps_at_p_i.tVector )

						print( "FDM tVector" )
						print( grad_slope_parT_fdm.tVector ) 


						slope_parT_minus_Y_i = manifolds.kendall2D_tVec( nManifoldDim )

						for kk in range( 2 ):
							for dd in range( nManifoldDim ):
								slope_parT_minus_Y_i.tVector[ kk, dd ] = slope_parT_p_i.tVector[ kk, dd ] - Y_i.tVector[ kk, dd ]


						dE[ kkk, d ] += grad_slope_parT_fdm.InnerProduct( slope_parT_minus_Y_i )


					print( "dE[ kkk, d ] " )				
					print( dE[ kkk, d ] )

					slope_tVec_updated.tVector[ kkk, d ] = slope_tVec.tVector[ kkk, d ] - ( stepSize * dE[ kkk, d ] )

			# Calculate Energy
			for i in range( len( Y ) ):
				Y_i = Y[ i ]
				slope_tVec_updated_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec_updated )

				energy_k_i = 0

				for k in range( 2 ):
					for d in range( nManifoldDim ):
						energy_k_i += ( slope_tVec_updated_at_p_i.tVector[ k, d ] - Y_i.tVector[ k, d ] ) ** 2

				energy_k += energy_k_i

			if energy_k > prev_energy:
				print( "Iteration : " + str( k + 1 ) )
				print( "Energy Increased : Halve step size")
				print( "Prev. Residual Energy" )
				print( prev_energy )

				energy_k = prev_energy

				energy_arr.append( energy_k )

				stepSize = stepSize / 2 
			else:
				print( "Iteration : " + str( k + 1 ) )
				print( "Residual Energy" )
				print( energy_k )
							
				stepSize = stepSize * 1.5
				slope_tVec = slope_tVec_updated
				prev_energy = energy_k
				energy_arr.append( energy_k )

			if energy_k < resTol:
				print( "Energy Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if stepSize < stepTol:
				slope_tVec = slope_tVec_updated
				print( "Step Size Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if k == nIter- 1:
				slope_tVec = slope_tVec_updated

		print( "Initial Slope" ) 
		print( init_slope_tVec.tVector )
		print( "Updated Slope" )
		print( slope_tVec.tVector )
		print( "Residual Energy" )
		print( energy_k )

		tangent_arr = []
		tangent_arr.append( slope_tVec )

		plt.figure()
		plt.plot( np.linspace( 1, k+1, num=k+1 ), energy_arr )
		plt.show()

		return tangent_arr

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )
	nData = len( Y )
	nParam = len( X[ 0 ] )
	
	p_anchor = beta0
	nManifoldDim = p_anchor.nPt 

	w_list = []

	for k in range( 2 ):
		for d in range( nManifoldDim ):
			w_list.append( [] )		

	for j in range( nData ):


		Y_j = Y[ j ]
		# Parallel translate a group-wise tangent vector to population-level intercept 
		beta_tVec_f_i = manifolds.kendall2D_tVec( nManifoldDim )

		for tt in range( len( cov_intercept_list[ j ] ) ):
			est_beta_tt = tVec_intercept_arr[ tt ]

			for kk in range( 2 ):
				for dd in range( nManifoldDim ):
					beta_tVec_f_i.tVector[ kk, dd ] += ( est_beta_tt.tVector[ kk, dd ] * cov_intercept_list[ j ][ tt ] )

		f_j = beta0.ExponentialMap( est_beta_tt )				
		Y_j_at_f_j = p0_list[ j ].ParallelTranslateAtoB( p0_list[ j ], f_j, Y_j ) 
		Y_j_tilde = f_j.ParallelTranslateAtoB( f_j, beta0, Y_j_at_f_j )

		tVec_j = Y_j_tilde

		# Parallel translate a group-wise tangent vector to population-level intercept 
		# tVec_j = p0_list[ j ].ParallelTranslateAtoB( p0_list[ j ], p_anchor, Y[ j ] )

		for k in range( 2 ):
			for d in range( nManifoldDim ):
				w_list[ k * nManifoldDim + d ].append( tVec_j.tVector[ k, d ] )

	estModel_list = []

	for k in range( 2 ):
		for d in range( nManifoldDim ):
			X_sm = sm.add_constant( X )
			w_d_np = np.asarray( w_list[ k * nManifoldDim + d ] )
			LS_model_d = sm.OLS( w_d_np, X_sm )

			est_d = LS_model_d.fit()
			estModel_list.append( est_d )

		# if verbose:
		# 	print( est_d.summary() )

	# base slope for t
	v_t = manifolds.kendall2D_tVec( nManifoldDim )

	for k in range( 2 ):
		for d in range( nManifoldDim ):
			v_t.tVector[ k, d ] = estModel_list[ k * nManifoldDim + d ].params[ 0 ] 

	new_tVec_arr = []

	for par in range( nParam ):
		v_tangent_on_p_anchor_param = manifolds.kendall2D_tVec( nManifoldDim )

		for k in range( 2 ):
			for d in range( nManifoldDim ):
				v_tangent_on_p_anchor_param.tVector[ k, d ] = estModel_list[ k * nManifoldDim + d ].params[ par + 1 ]

		new_tVec_arr.append( v_tangent_on_p_anchor_param )

	# Append time-wise slope tangent vector at the last
	new_tVec_arr.append( v_t )
	tangent_arr = new_tVec_arr 

	# # Calculate energy to check if the model was minimized
	# energy = 0

	# for n in range( nData ):
	# 	target = Y[ n ] 

	# 	tangent_t_n = manifolds.sphere_tVec( nManifoldDim )

	# 	for par in range( nParam ): 
	# 		for d in range( nManifoldDim ):
	# 			tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

	# 	estimate_n = p_anchor.ExponentialMap( tangent_t_n )

	# 	et = estimate_n.LogMap( target )

	# 	# Energy of the tangential error
	# 	energy += et.normSquared()

	# 	tangent_arr = new_tVec_arr
	# 	if verbose:
	# 		print( "==================================" )
	# 		print( "Residual Energy " ) 
	# 		print( energy )
	# 		print( "==================================" )

	return tangent_arr

def MultivariateLinearizedGeodesicRegression_Slope_DirectKendall2D( X, Y, beta0, p0_list, tVec_intercept_arr, cov_intercept_list, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	if len( X ) == 0 or len( X[ 0 ] ) == 0 :		
		nManifoldDim = beta0.nPt

		slope_tVec = manifolds.kendall2D_tVec( nManifoldDim )
		print( len( Y ) )

		for i in range( len( Y ) ):
			Y_i = Y [ i ]

			if i == 0:
				Y_i_tilde = Y_i
			else:
				Y_i_tilde = p0_list[ i ].ParallelTranslateAtoB( p0_list[i], beta0, Y_i )

			print( "Y_i")
			print( Y_i.tVector )			
			print( "Y_i_tilde")
			print( Y_i_tilde.tVector )	

			for k in range( 2 ):
				for d in range( nManifoldDim ):
					slope_tVec.tVector[ k, d ] += ( Y_i_tilde.tVector[ k, d ] / float( len( Y ) ) )

		init_slope_tVec = slope_tVec

		# Gradient Descent with eps
		eps = 0.0001
		stepSize = 0.01
		stepTol = 1e-8 
		resTol = 1e-6
		nIter = 500

		prev_energy = 0

		for i in range( len( Y ) ):
			slope_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

			prev_energy_i = 0

			for k in range( 2 ):
				for d in range( nManifoldDim ):
					prev_energy_i += ( slope_at_p_i.tVector[ k, d ] - Y_i.tVector[ k, d ] )**2.0

			prev_energy += prev_energy_i

		energy_arr = []

		for k in range( nIter ):
			slope_tVec_updated = manifolds.kendall2D_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ]

			# Calculate Gradient
			dE = np.zeros( 2, nManifoldDim )

			energy_k = 0

			# Calculate FDM
			for kkk in range( 2 ):
				for d in range( nManifoldDim ):
					slope_pos_eps = manifolds.kendall2D_tVec( nManifoldDim )
					slope_neg_eps = manifolds.kendall2D_tVec( nManifoldDim )

					for kk in range( 2 ):
						for dd in range( nManifoldDim ):
							slope_pos_eps.tVector[ kk, dd ] = slope_tVec.tVector[ kk, dd ] 
							slope_neg_eps.tVector[ kk, dd ] = slope_tVec.tVector[ kk, dd ] 

					slope_pos_eps.tVector[ kkk, d ] = slope_tVec.tVector[ kkk, d ] + eps
					slope_neg_eps.tVector[ kkk, d ] = slope_tVec.tVector[ kkk, d ] - eps

					for i in range( len( Y ) ):
						Y_i = Y[ i ]

						slope_parT_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

						slope_pos_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_pos_eps )
						slope_neg_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_neg_eps )

						grad_slope_parT_fdm = manifolds.kendall2D_tVec( nManifoldDim )

						for kk in range( 2 ):
							for dd in range( nManifoldDim ):
								grad_slope_parT_fdm.tVector[ kk, dd ] = float( slope_pos_eps_at_p_i.tVector[ kk, dd ] - slope_neg_eps_at_p_i.tVector[ kk, dd ] ) / float( 2.0 * eps )

						print( "slope_pos_eps" ) 
						print( slope_pos_eps.tVector )
						print( "slope_neg_eps" ) 
						print( slope_neg_eps.tVector )

						print( "slope_pos_eps_p_i" ) 
						print( slope_pos_eps_at_p_i.tVector )
						print( "slope_neg_eps_p_i" ) 
						print( slope_neg_eps_at_p_i.tVector )

						print( "FDM tVector" )
						print( grad_slope_parT_fdm.tVector ) 


						slope_parT_minus_Y_i = manifolds.kendall2D_tVec( nManifoldDim )

						for kk in range( 2 ):
							for dd in range( nManifoldDim ):
								slope_parT_minus_Y_i.tVector[ kk, dd ] = slope_parT_p_i.tVector[ kk, dd ] - Y_i.tVector[ kk, dd ]


						dE[ kkk, d ] += grad_slope_parT_fdm.InnerProduct( slope_parT_minus_Y_i )


					print( "dE[ kkk, d ] " )				
					print( dE[ kkk, d ] )

					slope_tVec_updated.tVector[ kkk, d ] = slope_tVec.tVector[ kkk, d ] - ( stepSize * dE[ kkk, d ] )

			# Calculate Energy
			for i in range( len( Y ) ):
				Y_i = Y[ i ]
				slope_tVec_updated_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec_updated )

				energy_k_i = 0

				for k in range( 2 ):
					for d in range( nManifoldDim ):
						energy_k_i += ( slope_tVec_updated_at_p_i.tVector[ k, d ] - Y_i.tVector[ k, d ] ) ** 2

				energy_k += energy_k_i

			if energy_k > prev_energy:
				print( "Iteration : " + str( k + 1 ) )
				print( "Energy Increased : Halve step size")
				print( "Prev. Residual Energy" )
				print( prev_energy )

				energy_k = prev_energy

				energy_arr.append( energy_k )

				stepSize = stepSize / 2 
			else:
				print( "Iteration : " + str( k + 1 ) )
				print( "Residual Energy" )
				print( energy_k )
							
				stepSize = stepSize * 1.5
				slope_tVec = slope_tVec_updated
				prev_energy = energy_k
				energy_arr.append( energy_k )

			if energy_k < resTol:
				print( "Energy Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if stepSize < stepTol:
				slope_tVec = slope_tVec_updated
				print( "Step Size Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if k == nIter- 1:
				slope_tVec = slope_tVec_updated

		print( "Initial Slope" ) 
		print( init_slope_tVec.tVector )
		print( "Updated Slope" )
		print( slope_tVec.tVector )
		print( "Residual Energy" )
		print( energy_k )

		tangent_arr = []
		tangent_arr.append( slope_tVec )

		plt.figure()
		plt.plot( np.linspace( 1, k+1, num=k+1 ), energy_arr )
		plt.show()

		return tangent_arr

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )
	nData = len( Y )
	nParam = len( X[ 0 ] )
	
	p_anchor = beta0
	nManifoldDim = p_anchor.nPt 

	w_list = []

	for k in range( 2 ):
		for d in range( nManifoldDim ):
			w_list.append( [] )		

	for j in range( nData ):
		# Parallel translate a group-wise tangent vector to population-level intercept 
		tVec_j = p0_list[ j ].ParallelTranslateAtoB( p0_list[ j ], p_anchor, Y[ j ] )

		for k in range( 2 ):
			for d in range( nManifoldDim ):
				w_list[ k * nManifoldDim + d ].append( tVec_j.tVector[ k, d ] )

	estModel_list = []

	for k in range( 2 ):
		for d in range( nManifoldDim ):
			X_sm = sm.add_constant( X )
			w_d_np = np.asarray( w_list[ k * nManifoldDim + d ] )
			LS_model_d = sm.OLS( w_d_np, X_sm )

			est_d = LS_model_d.fit()
			estModel_list.append( est_d )

		# if verbose:
		# 	print( est_d.summary() )

	# base slope for t
	v_t = manifolds.kendall2D_tVec( nManifoldDim )

	for k in range( 2 ):
		for d in range( nManifoldDim ):
			v_t.tVector[ k, d ] = estModel_list[ k * nManifoldDim + d ].params[ 0 ] 

	new_tVec_arr = []

	for par in range( nParam ):
		v_tangent_on_p_anchor_param = manifolds.kendall2D_tVec( nManifoldDim )

		for k in range( 2 ):
			for d in range( nManifoldDim ):
				v_tangent_on_p_anchor_param.tVector[ k, d ] = estModel_list[ k * nManifoldDim + d ].params[ par + 1 ]

		new_tVec_arr.append( v_tangent_on_p_anchor_param )

	# Append time-wise slope tangent vector at the last
	new_tVec_arr.append( v_t )
	tangent_arr = new_tVec_arr 

	# # Calculate energy to check if the model was minimized
	# energy = 0

	# for n in range( nData ):
	# 	target = Y[ n ] 

	# 	tangent_t_n = manifolds.sphere_tVec( nManifoldDim )

	# 	for par in range( nParam ): 
	# 		for d in range( nManifoldDim ):
	# 			tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

	# 	estimate_n = p_anchor.ExponentialMap( tangent_t_n )

	# 	et = estimate_n.LogMap( target )

	# 	# Energy of the tangential error
	# 	energy += et.normSquared()

	# 	tangent_arr = new_tVec_arr
	# 	if verbose:
	# 		print( "==================================" )
	# 		print( "Residual Energy " ) 
	# 		print( energy )
	# 		print( "==================================" )

	return tangent_arr

#################################################################################
### 						 	Positive Real Numbers	  					  ###
#################################################################################
def MultivariateLinearizedGeodesicRegression_PosReal_BottomUp( t_list, pt_list, cov_intercept_list, cov_slope_list=[], max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# The numbers
	nGroup = len( t_list )

	nData_group = [] 
	for i in range( nGroup ):
		nData_group.append( len( t_list[ i ] ) )

	nParam_int = len( cov_intercept_list[ 0 ] )

	nParam_slope = 0	
	if not len( cov_slope_list ) == 0:
		nParam_slope = len( cov_slope_list[ 0 ] )

	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

		print( "No. Group : " + str( nGroup ) )

		for i in range( nGroup ):
			print( "Group " + str( i + 1 ) + " : " + str( nData_group[ i ] ) + " Obs." )
		print( "No. Covariates for Intercept: " + str( nParam_int ) )
		print( "No. Covariates for Slope: " + str( nParam_slope ) )

	# Group-wise intercept, slope tangent vector, covariates (intercept/slope), time 
	p0_group_list = [] # 1-D Array N x 1
	v_group_list = []  # 1-D Array N x 1 
	cov_intercept_group_list = [] # 2-D Array N x C_int
	cov_slope_group_list = [] # 2-D Array N x C_slope
	t_group_list = []   # 2-D Array N x O

	for g in range( nGroup ):
		t_list_g = t_list[ g ]
		pt_list_g = pt_list[ g ] 

		p0_g, v_g = LinearizedGeodesicRegression( t_list_g, pt_list_g )

		p0_group_list.append( p0_g )
		v_group_list.append( v_g )
		cov_intercept_group_list.append( cov_intercept_list[ g ] )

		if not len( cov_slope_list ) == 0:
			cov_slope_group_list.append( cov_slope_list[ g ] )

		# # Check R2
		# mean_g = FrechetMean( pt_list[ g ] )

		# sqDist_SG_sum = 0 
		# sqVar_sum = 0

		# for i in range( len( pt_list[ g ] ) ):
		# 	p_i = pt_list_g[ i ]
		# 	t_i = t_list_g[ i ]

		# 	slope_t_i = v_g.ScalarMultiply( t_i )
		# 	est_p_i = p0_g.ExponentialMap( slope_t_i )

		# 	tVec_est_p_i_to_p_i = est_p_i.LogMap( p_i )

		# 	sqDist_i = tVec_est_p_i_to_p_i.normSquared() 

		# 	sqDist_SG_sum += sqDist_i

		# 	tVec_mean_to_p_n = mean_g.LogMap( p_i ) 

		# 	sqVar_n = tVec_mean_to_p_n.normSquared()

		# 	sqVar_sum += sqVar_n

		# R2_SG = 1 - ( sqDist_SG_sum / sqVar_sum )

		# print( "Subject : " + str( g ) )
		# print( str( nData_group[ g ] ) + " Obs." )
		# print( R2_SG )


	##############################################	
	## Solve Intercepts Points w.r.t Covariates ##
	##############################################
	beta0, tangent_intercept_arr = MultivariateLinearizedGeodesicRegression_Intercept_PosReal( cov_intercept_group_list, p0_group_list, verbose=verbose )

	##############################################	
	## Solve Tangent Vectors w.r.t Covariates ##
	##############################################
	tangent_slope_arr = MultivariateLinearizedGeodesicRegression_Slope_PosReal( cov_slope_group_list, v_group_list, beta0, p0_group_list, tangent_intercept_arr, verbose=verbose )

	return beta0, tangent_intercept_arr, tangent_slope_arr

def MultivariateLinearizedGeodesicRegression_Intercept_PosReal( X, Y, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )

	nData = len( Y )
	nParam = len( X[ 0 ] )

	if nParam == 0:
		base = FrechetMean( Y )
		tangent_arr = [] 		

		return base, tangent_arr

	# Anchor point is chosen by the last entry of covariates
	# Continuous variable such as a genetic disease score should be the last entry of covariates
	# If data don't have a continuous covariates, the last entry can be a categorical covariate
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	# Set an anchor point
	t_min_idx = np.argmin( t_list )		
	p_anchor = Y[ t_min_idx ]

	nManifoldDim = p_anchor.nDim 

	# Initial intercept point
	init_Interp = manifolds.pos_real( nManifoldDim )

	# Initial set of tangent vectors
	init_tVec_arr = [] 

	for i in range( nParam ):
		init_tVec_arr.append( manifolds.pos_real_tVec( nManifoldDim ) )

	base = init_Interp
	tangent_arr = init_tVec_arr

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prev_tVec_arr = tangent_arr

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( Y[ j ] )

			for d in range( nManifoldDim ):
				w_list[ d ].append( tVec_j.tVector[ d ] )

		estModel_list = []

		for d in range( nManifoldDim ):
			X_sm = sm.add_constant( X )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, X_sm )
			# est_d = LS_model_d.fit(method='qr')

			est_d = LS_model_d.fit()
			estModel_list.append( est_d )

			if verbose:
				print( est_d.summary() )

		# Intercept point
		v_to_base_on_p_anchor = manifolds.pos_real_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		new_tVec_arr = []

		for par in range( nParam ):
			v_tangent_on_p_anchor_param = manifolds.pos_real_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

			newTangent_param = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor_param )	
			new_tVec_arr.append( newTangent_param )

		# Calculate energy to check if the model was minimized
		energy = 0

		for n in range( nData ):
			target = Y[ n ] 

			current_tangent_VG_intercept = manifolds.pos_real_tVec( nManifoldDim ) 
			current_tangent_VG_slope = manifolds.pos_real_tVec( nManifoldDim )  

			tangent_t_n = manifolds.pos_real_tVec( nManifoldDim )

			for par in range( nParam ):
				for d in range( nManifoldDim ):
					tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

			estimate_n = newBase.ExponentialMap( tangent_t_n )

			et = estimate_n.LogMap( target )

			# Energy of the tangential error
			energy += et.normSquared()

		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )
			break;
		else:
			prevBase = newBase
			prev_tVec_arr = new_tVec_arr
			p_anchor = newBase
			base = newBase
			tangent_arr = new_tVec_arr
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent_arr

def MultivariateLinearizedGeodesicRegression_Slope_PosReal( X, Y, beta0, p0_list, tVec_intercept_arr, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	if len( X ) == 0 or len( X[ 0 ] ) == 0 :		
		nManifoldDim = beta0.nDim

		slope_tVec = manifolds.pos_real_tVec( nManifoldDim )
		print( len( Y ) )

		for i in range( len( Y ) ):
			Y_i = Y [ i ]

			if i == 0:
				Y_i_tilde = Y_i
			else:
				Y_i_tilde = p0_list[ i ].ParallelTranslateAtoB( p0_list[i], beta0, Y_i )

			print( "Y_i")
			print( Y_i.tVector )			
			print( "Y_i_tilde")
			print( Y_i_tilde.tVector )	

			for d in range( nManifoldDim ):
				slope_tVec.tVector[ d ] += ( Y_i_tilde.tVector[ d ] / float( len( Y ) ) )

		init_slope_tVec = slope_tVec

		# Gradient Descent with eps
		eps = 0.0001
		stepSize = 0.01
		stepTol = 1e-8 
		resTol = 1e-6
		nIter = 500

		prev_energy = 0

		for i in range( len( Y ) ):
			slope_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

			prev_energy_i = 0

			for d in range( nManifoldDim ):
				prev_energy_i += ( slope_at_p_i.tVector[ d ] - Y_i.tVector[ d ] )**2.0

			prev_energy += prev_energy_i

		energy_arr = []

		for k in range( nIter ):
			slope_tVec_updated = manifolds.pos_real_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ]

			# Calculate Gradient
			dE = np.zeros( nManifoldDim )

			energy_k = 0

			# Calculate FDM
			for d in range( nManifoldDim ):
				slope_pos_eps = manifolds.pos_real_tVec( nManifoldDim )
				slope_neg_eps = manifolds.pos_real_tVec( nManifoldDim )

				for dd in range( nManifoldDim ):
					slope_pos_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 
					slope_neg_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 

				slope_pos_eps.tVector[ d ] = slope_tVec.tVector[ d ] + eps
				slope_neg_eps.tVector[ d ] = slope_tVec.tVector[ d ] - eps

				for i in range( len( Y ) ):
					Y_i = Y[ i ]

					slope_parT_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

					slope_pos_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_pos_eps )
					slope_neg_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_neg_eps )

					grad_slope_parT_fdm = manifolds.pos_real_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						grad_slope_parT_fdm.tVector[ dd ] = float( slope_pos_eps_at_p_i.tVector[ dd ] - slope_neg_eps_at_p_i.tVector[ dd ] ) / float( 2.0 * eps )

					print( "slope_pos_eps" ) 
					print( slope_pos_eps.tVector )
					print( "slope_neg_eps" ) 
					print( slope_neg_eps.tVector )

					print( "slope_pos_eps_p_i" ) 
					print( slope_pos_eps_at_p_i.tVector )
					print( "slope_neg_eps_p_i" ) 
					print( slope_neg_eps_at_p_i.tVector )

					print( "FDM tVector" )
					print( grad_slope_parT_fdm.tVector ) 

					slope_parT_minus_Y_i = manifolds.kendall2D_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						slope_parT_minus_Y_i.tVector[ dd ] = slope_parT_p_i.tVector[ dd ] - Y_i.tVector[ dd ]

					dE[ d ] += grad_slope_parT_fdm.InnerProduct( slope_parT_minus_Y_i )

				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ] - ( stepSize * dE[ d ] )

			# Calculate Energy
			for i in range( len( Y ) ):
				Y_i = Y[ i ]
				slope_tVec_updated_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec_updated )

				energy_k_i = 0

				for d in range( nManifoldDim ):
					energy_k_i += ( slope_tVec_updated_at_p_i.tVector[ d ] - Y_i.tVector[ d ] ) ** 2

				energy_k += energy_k_i

			if energy_k > prev_energy:
				print( "Iteration : " + str( k + 1 ) )
				print( "Energy Increased : Halve step size")
				print( "Prev. Residual Energy" )
				print( prev_energy )

				energy_k = prev_energy

				energy_arr.append( energy_k )

				stepSize = stepSize / 2 
			else:
				print( "Iteration : " + str( k + 1 ) )
				print( "Residual Energy" )
				print( energy_k )
							
				stepSize = stepSize * 1.5
				slope_tVec = slope_tVec_updated
				prev_energy = energy_k
				energy_arr.append( energy_k )

			if energy_k < resTol:
				print( "Energy Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if stepSize < stepTol:
				slope_tVec = slope_tVec_updated
				print( "Step Size Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if k == nIter- 1:
				slope_tVec = slope_tVec_updated

		print( "Initial Slope" ) 
		print( init_slope_tVec.tVector )
		print( "Updated Slope" )
		print( slope_tVec.tVector )
		print( "Residual Energy" )
		print( energy_k )

		tangent_arr = []
		tangent_arr.append( slope_tVec )

		plt.figure()
		plt.plot( np.linspace( 1, k+1, num=k+1 ), energy_arr )
		plt.show()

		return tangent_arr

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )
	nData = len( Y )
	nParam = len( X[ 0 ] )
	
	p_anchor = beta0
	nManifoldDim = p_anchor.nDim 

	w_list = []

	for d in range( nManifoldDim ):
		w_list.append( [] )		

	for j in range( nData ):
		# Parallel translate a group-wise tangent vector to population-level intercept 
		tVec_j = p0_list[ j ].ParallelTranslateAtoB( p0_list[ j ], p_anchor, Y[ j ] )

		for d in range( nManifoldDim ):
			w_list[ d ].append( tVec_j.tVector[ d ] )

	estModel_list = []

	for d in range( nManifoldDim ):
		X_sm = sm.add_constant( X )
		w_d_np = np.asarray( w_list[ d ] )
		LS_model_d = sm.OLS( w_d_np, X_sm )

		est_d = LS_model_d.fit()
		estModel_list.append( est_d )

		# if verbose:
		# 	print( est_d.summary() )

	# base slope for t
	v_t = manifolds.pos_real_tVec( nManifoldDim )

	for d in range( nManifoldDim ):
		v_t.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

	new_tVec_arr = []

	for par in range( nParam ):
		v_tangent_on_p_anchor_param = manifolds.pos_real_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

		new_tVec_arr.append( v_tangent_on_p_anchor_param )

	# Append time-wise slope tangent vector at the last
	new_tVec_arr.append( v_t )
	tangent_arr = new_tVec_arr 

	# # Calculate energy to check if the model was minimized
	# energy = 0

	# for n in range( nData ):
	# 	target = Y[ n ] 

	# 	tangent_t_n = manifolds.sphere_tVec( nManifoldDim )

	# 	for par in range( nParam ): 
	# 		for d in range( nManifoldDim ):
	# 			tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

	# 	estimate_n = p_anchor.ExponentialMap( tangent_t_n )

	# 	et = estimate_n.LogMap( target )

	# 	# Energy of the tangential error
	# 	energy += et.normSquared()

	# 	tangent_arr = new_tVec_arr
	# 	if verbose:
	# 		print( "==================================" )
	# 		print( "Residual Energy " ) 
	# 		print( energy )
	# 		print( "==================================" )

	return tangent_arr

#################################################################################
### 						 	Euclidean Numbers	  					  	  ###
#################################################################################
def MultivariateLinearizedGeodesicRegression_Euclidean_BottomUp( t_list, pt_list, cov_intercept_list, cov_slope_list=[], max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# The numbers
	nGroup = len( t_list )

	nData_group = [] 
	for i in range( nGroup ):
		nData_group.append( len( t_list[ i ] ) )

	nParam_int = len( cov_intercept_list[ 0 ] )

	nParam_slope = 0	
	if not len( cov_slope_list ) == 0:
		nParam_slope = len( cov_slope_list[ 0 ] )

	if verbose:
		print( "=================================================================" )
		print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
		print( "=================================================================" ) 

		print( "No. Group : " + str( nGroup ) )

		for i in range( nGroup ):
			print( "Group " + str( i + 1 ) + " : " + str( nData_group[ i ] ) + " Obs." )
		print( "No. Covariates for Intercept: " + str( nParam_int ) )
		print( "No. Covariates for Slope: " + str( nParam_slope ) )

	# Group-wise intercept, slope tangent vector, covariates (intercept/slope), time 
	p0_group_list = [] # 1-D Array N x 1
	v_group_list = []  # 1-D Array N x 1 
	cov_intercept_group_list = [] # 2-D Array N x C_int
	cov_slope_group_list = [] # 2-D Array N x C_slope
	t_group_list = []   # 2-D Array N x O

	for g in range( nGroup ):
		t_list_g = t_list[ g ]
		pt_list_g = pt_list[ g ] 

		p0_g, v_g = LinearizedGeodesicRegression( t_list_g, pt_list_g )

		p0_group_list.append( p0_g )
		v_group_list.append( v_g )
		cov_intercept_group_list.append( cov_intercept_list[ g ] )

		if not len( cov_slope_list ) == 0:
			cov_slope_group_list.append( cov_slope_list[ g ] )

		# # Check R2
		# mean_g = FrechetMean( pt_list[ g ] )

		# sqDist_SG_sum = 0 
		# sqVar_sum = 0

		# for i in range( len( pt_list[ g ] ) ):
		# 	p_i = pt_list_g[ i ]
		# 	t_i = t_list_g[ i ]

		# 	slope_t_i = v_g.ScalarMultiply( t_i )
		# 	est_p_i = p0_g.ExponentialMap( slope_t_i )

		# 	tVec_est_p_i_to_p_i = est_p_i.LogMap( p_i )

		# 	sqDist_i = tVec_est_p_i_to_p_i.normSquared() 

		# 	sqDist_SG_sum += sqDist_i

		# 	tVec_mean_to_p_n = mean_g.LogMap( p_i ) 

		# 	sqVar_n = tVec_mean_to_p_n.normSquared()

		# 	sqVar_sum += sqVar_n

		# R2_SG = 1 - ( sqDist_SG_sum / sqVar_sum )

		# print( "Subject : " + str( g ) )
		# print( str( nData_group[ g ] ) + " Obs." )
		# print( R2_SG )


	##############################################	
	## Solve Intercepts Points w.r.t Covariates ##
	##############################################
	beta0, tangent_intercept_arr = MultivariateLinearizedGeodesicRegression_Intercept_Euclidean( cov_intercept_group_list, p0_group_list, verbose=verbose )

	##############################################	
	## Solve Tangent Vectors w.r.t Covariates ##
	##############################################
	tangent_slope_arr = MultivariateLinearizedGeodesicRegression_Slope_Euclidean( cov_slope_group_list, v_group_list, beta0, p0_group_list, tangent_intercept_arr, verbose=verbose )

	return beta0, tangent_intercept_arr, tangent_slope_arr

def MultivariateLinearizedGeodesicRegression_Intercept_Euclidean( X, Y, max_iter = 100, stepSize = 0.05, step_tol = 1e-8, useFrechetMeanAnchor = False, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )

	nData = len( Y )
	nParam = len( X[ 0 ] )

	if nParam == 0:
		base = FrechetMean( Y )
		tangent_arr = [] 		

		return base, tangent_arr

	# Anchor point is chosen by the last entry of covariates
	# Continuous variable such as a genetic disease score should be the last entry of covariates
	# If data don't have a continuous covariates, the last entry can be a categorical covariate
	t_list = []

	for i in range( len( X ) ):
		t_list.append( X[ i ][ -1 ] )

	# Set an anchor point
	t_min_idx = np.argmin( t_list )		
	p_anchor = Y[ t_min_idx ]

	nManifoldDim = p_anchor.nDim 

	# Initial intercept point
	init_Interp = manifolds.euclidean( nManifoldDim )

	# Initial set of tangent vectors
	init_tVec_arr = [] 

	for i in range( nParam ):
		init_tVec_arr.append( manifolds.euclidean_tVec( nManifoldDim ) )

	base = init_Interp
	tangent_arr = init_tVec_arr

	# Iteration Parameters
	prevEnergy = 1e10
	prevBase = base
	prev_tVec_arr = tangent_arr

	for i in range( max_iter ):
		tVec_list = []
		w_list = []

		for d in range( nManifoldDim ):
			w_list.append( [] )		

		for j in range( nData ):
			tVec_j = p_anchor.LogMap( Y[ j ] )

			for d in range( nManifoldDim ):
				w_list[ d ].append( tVec_j.tVector[ d ] )

		estModel_list = []

		for d in range( nManifoldDim ):
			X_sm = sm.add_constant( X )
			w_d_np = np.asarray( w_list[ d ] )
			LS_model_d = sm.OLS( w_d_np, X_sm )
			# est_d = LS_model_d.fit(method='qr')

			est_d = LS_model_d.fit()
			estModel_list.append( est_d )

			if verbose:
				print( est_d.summary() )

		# Intercept point
		v_to_base_on_p_anchor = manifolds.euclidean_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_to_base_on_p_anchor.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

		newBase = p_anchor.ExponentialMap( v_to_base_on_p_anchor )
		new_tVec_arr = []

		for par in range( nParam ):
			v_tangent_on_p_anchor_param = manifolds.euclidean_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

			newTangent_param = p_anchor.ParallelTranslateAtoB( p_anchor, newBase, v_tangent_on_p_anchor_param )	
			new_tVec_arr.append( newTangent_param )

		# Calculate energy to check if the model was minimized
		energy = 0

		for n in range( nData ):
			target = Y[ n ] 

			current_tangent_VG_intercept = manifolds.euclidean_tVec( nManifoldDim ) 
			current_tangent_VG_slope = manifolds.euclidean_tVec( nManifoldDim )  

			tangent_t_n = manifolds.euclidean_tVec( nManifoldDim )

			for par in range( nParam ):
				for d in range( nManifoldDim ):
					tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

			estimate_n = newBase.ExponentialMap( tangent_t_n )

			et = estimate_n.LogMap( target )

			# Energy of the tangential error
			energy += et.normSquared()

		if energy >= prevEnergy:
			if verbose:
				print( "=========================" )
				print( "   Energy Increased " ) 
				print ( energy )
				print( "=========================" )
			break;
		else:
			prevBase = newBase
			prev_tVec_arr = new_tVec_arr
			p_anchor = newBase
			base = newBase
			tangent_arr = new_tVec_arr
			prevEnergy = energy
			if verbose:
				print( "==================================" )
				print( str( i ) + "th Iteration " ) 
				print( energy )
				print( "==================================" )
		if stepSize < step_tol:
			if verbose:
				print( "==================================" )
				print( "Step size under tolerance")
				print( "Aborting")
				print( "==================================" )
			break

	return base, tangent_arr

def MultivariateLinearizedGeodesicRegression_Slope_Euclidean( X, Y, beta0, p0_list, tVec_intercept_arr, verbose=True ):
	# if verbose:
	# 	print( "=================================================================" )
	# 	print( "      Linear Regression on Anchor Point Tangent Vector Space    " )
	# 	print( "=================================================================" ) 

	if len( X ) == 0 or len( X[ 0 ] ) == 0 :		
		nManifoldDim = beta0.nDim

		slope_tVec = manifolds.euclidean_tVec( nManifoldDim )
		print( len( Y ) )

		for i in range( len( Y ) ):
			Y_i = Y [ i ]

			if i == 0:
				Y_i_tilde = Y_i
			else:
				Y_i_tilde = p0_list[ i ].ParallelTranslateAtoB( p0_list[i], beta0, Y_i )

			print( "Y_i")
			print( Y_i.tVector )			
			print( "Y_i_tilde")
			print( Y_i_tilde.tVector )	

			for d in range( nManifoldDim ):
				slope_tVec.tVector[ d ] += ( Y_i_tilde.tVector[ d ] / float( len( Y ) ) )

		init_slope_tVec = slope_tVec

		# Gradient Descent with eps
		eps = 0.0001
		stepSize = 0.01
		stepTol = 1e-8 
		resTol = 1e-6
		nIter = 500

		prev_energy = 0

		for i in range( len( Y ) ):
			slope_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

			prev_energy_i = 0

			for d in range( nManifoldDim ):
				prev_energy_i += ( slope_at_p_i.tVector[ d ] - Y_i.tVector[ d ] )**2.0

			prev_energy += prev_energy_i

		energy_arr = []

		for k in range( nIter ):
			slope_tVec_updated = manifolds.euclidean_tVec( nManifoldDim )

			for d in range( nManifoldDim ):
				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ]

			# Calculate Gradient
			dE = np.zeros( nManifoldDim )

			energy_k = 0

			# Calculate FDM
			for d in range( nManifoldDim ):
				slope_pos_eps = manifolds.euclidean_tVec( nManifoldDim )
				slope_neg_eps = manifolds.euclidean_tVec( nManifoldDim )

				for dd in range( nManifoldDim ):
					slope_pos_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 
					slope_neg_eps.tVector[ dd ] = slope_tVec.tVector[ dd ] 

				slope_pos_eps.tVector[ d ] = slope_tVec.tVector[ d ] + eps
				slope_neg_eps.tVector[ d ] = slope_tVec.tVector[ d ] - eps

				for i in range( len( Y ) ):
					Y_i = Y[ i ]

					slope_parT_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec )

					slope_pos_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_pos_eps )
					slope_neg_eps_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_neg_eps )

					grad_slope_parT_fdm = manifolds.euclidean_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						grad_slope_parT_fdm.tVector[ dd ] = float( slope_pos_eps_at_p_i.tVector[ dd ] - slope_neg_eps_at_p_i.tVector[ dd ] ) / float( 2.0 * eps )

					print( "slope_pos_eps" ) 
					print( slope_pos_eps.tVector )
					print( "slope_neg_eps" ) 
					print( slope_neg_eps.tVector )

					print( "slope_pos_eps_p_i" ) 
					print( slope_pos_eps_at_p_i.tVector )
					print( "slope_neg_eps_p_i" ) 
					print( slope_neg_eps_at_p_i.tVector )

					print( "FDM tVector" )
					print( grad_slope_parT_fdm.tVector ) 

					slope_parT_minus_Y_i = manifolds.kendall2D_tVec( nManifoldDim )

					for dd in range( nManifoldDim ):
						slope_parT_minus_Y_i.tVector[ dd ] = slope_parT_p_i.tVector[ dd ] - Y_i.tVector[ dd ]

					dE[ d ] += grad_slope_parT_fdm.InnerProduct( slope_parT_minus_Y_i )

				slope_tVec_updated.tVector[ d ] = slope_tVec.tVector[ d ] - ( stepSize * dE[ d ] )

			# Calculate Energy
			for i in range( len( Y ) ):
				Y_i = Y[ i ]
				slope_tVec_updated_at_p_i = beta0.ParallelTranslateAtoB( beta0, p0_list[ i ], slope_tVec_updated )

				energy_k_i = 0

				for d in range( nManifoldDim ):
					energy_k_i += ( slope_tVec_updated_at_p_i.tVector[ d ] - Y_i.tVector[ d ] ) ** 2

				energy_k += energy_k_i

			if energy_k > prev_energy:
				print( "Iteration : " + str( k + 1 ) )
				print( "Energy Increased : Halve step size")
				print( "Prev. Residual Energy" )
				print( prev_energy )

				energy_k = prev_energy

				energy_arr.append( energy_k )

				stepSize = stepSize / 2 
			else:
				print( "Iteration : " + str( k + 1 ) )
				print( "Residual Energy" )
				print( energy_k )
							
				stepSize = stepSize * 1.5
				slope_tVec = slope_tVec_updated
				prev_energy = energy_k
				energy_arr.append( energy_k )

			if energy_k < resTol:
				print( "Energy Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if stepSize < stepTol:
				slope_tVec = slope_tVec_updated
				print( "Step Size Tolerance")
				print( "# Iteration : " + str( k + 1 ) )
				print( "Initial Slope" ) 
				print( init_slope_tVec.tVector )
				print( "Updated Slope" )
				print( slope_tVec.tVector )
				print( "Residual Energy" )
				print( energy_k )
				break

			if k == nIter- 1:
				slope_tVec = slope_tVec_updated

		print( "Initial Slope" ) 
		print( init_slope_tVec.tVector )
		print( "Updated Slope" )
		print( slope_tVec.tVector )
		print( "Residual Energy" )
		print( energy_k )

		tangent_arr = []
		tangent_arr.append( slope_tVec )

		plt.figure()
		plt.plot( np.linspace( 1, k+1, num=k+1 ), energy_arr )
		plt.show()

		return tangent_arr

	# 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
	# 	print( "No. Observations : " + str( len( X ) ) )
	nData = len( Y )
	nParam = len( X[ 0 ] )
	
	p_anchor = beta0
	nManifoldDim = p_anchor.nDim 

	w_list = []

	for d in range( nManifoldDim ):
		w_list.append( [] )		

	for j in range( nData ):
		# Parallel translate a group-wise tangent vector to population-level intercept 
		tVec_j = p0_list[ j ].ParallelTranslateAtoB( p0_list[ j ], p_anchor, Y[ j ] )

		for d in range( nManifoldDim ):
			w_list[ d ].append( tVec_j.tVector[ d ] )

	estModel_list = []

	for d in range( nManifoldDim ):
		X_sm = sm.add_constant( X )
		w_d_np = np.asarray( w_list[ d ] )
		LS_model_d = sm.OLS( w_d_np, X_sm )

		est_d = LS_model_d.fit()
		estModel_list.append( est_d )

		# if verbose:
		# 	print( est_d.summary() )

	# base slope for t
	v_t = manifolds.euclidean_tVec( nManifoldDim )

	for d in range( nManifoldDim ):
		v_t.tVector[ d ] = estModel_list[ d ].params[ 0 ] 

	new_tVec_arr = []

	for par in range( nParam ):
		v_tangent_on_p_anchor_param = manifolds.euclidean_tVec( nManifoldDim )

		for d in range( nManifoldDim ):
			v_tangent_on_p_anchor_param.tVector[ d ] = estModel_list[ d ].params[ par + 1 ]

		new_tVec_arr.append( v_tangent_on_p_anchor_param )

	# Append time-wise slope tangent vector at the last
	new_tVec_arr.append( v_t )
	tangent_arr = new_tVec_arr 

	# # Calculate energy to check if the model was minimized
	# energy = 0

	# for n in range( nData ):
	# 	target = Y[ n ] 

	# 	tangent_t_n = manifolds.sphere_tVec( nManifoldDim )

	# 	for par in range( nParam ): 
	# 		for d in range( nManifoldDim ):
	# 			tangent_t_n.tVector[ d ] += ( new_tVec_arr[ par ].tVector[ d ] * X[ n ][ par ] )

	# 	estimate_n = p_anchor.ExponentialMap( tangent_t_n )

	# 	et = estimate_n.LogMap( target )

	# 	# Energy of the tangential error
	# 	energy += et.normSquared()

	# 	tangent_arr = new_tVec_arr
	# 	if verbose:
	# 		print( "==================================" )
	# 		print( "Residual Energy " ) 
	# 		print( energy )
	# 		print( "==================================" )

	return tangent_arr


###############################################################
#####					 Miscelleneous		 			  #####
###############################################################
def HelmertSubmatrix( nAtoms ):
	# Create a Helmert submatrix - similarity-invariant
	H = np.zeros( [ nAtoms - 1, nAtoms ] )

	for k in range( nAtoms - 1 ):
		h_k = -np.divide( 1.0, np.sqrt( ( k + 1 ) * ( k + 2 ) ) )
		neg_kh_k = np.multiply( h_k, -( k + 1 ) )
		for h in range( k + 1 ):
			H[ k, h ] = h_k
		H[ k, k + 1 ] = neg_kh_k 

	return H


def HelmertMatrix( nAtoms ):
	# Create a Helmert matrix - similiarity-invariant : First row - Center of Gravity (mass) (uniform mass of points)
	H_full = np.zeors( [ nAtoms, nAtoms ] )

	for h in range( nAtoms ):
		H_full[ 0, h ] = np.divide( 1, np.sqrt( nAtoms ) )

	for k in range( 1, nAtoms, 1 ):
		h_k = -np.divide( 1.0, np.sqrt( ( k ) * ( k + 1 ) ) )
		neg_kh_k = np.multiply( h_k, -k )
		for h in range( k ):
			H_full[ k, h ] = h_k
		H_full[ k, k ] = neg_kh_k 

	return H_full
