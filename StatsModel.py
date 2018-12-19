# MRep Manifold 
import manifolds
import numpy as np

import pylab 

from random import shuffle 


# Stats Model
import statsmodels.api as sm

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
	elif dataList[ 0 ].Type == "CMRep_Abstract":
		return FrechetMean_CMRep_Abstract( dataList, maxIter, tol, stepsize )

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

	mu = FrechetMean( dataList, maxIter, tol, stepSize )

	return mu

def TangentPGA_CMRep_Abstract( dataList, maxIter = 500, tol = 0.001, stepsize=0.01 ):
	nManDim = dataList[ 0 ].nDim
	nData = len( dataList )

	# Intrinsic Mean
	mu = FrechetMean( dataList, maxIter, tol, stepSize )

	# Covariance matrix on a tangent vector space 
	nCenterDim = dataList[ 0 ].pt[ 0 ].nDim
	nScaleDim = dataList[ 0 ].pt[ 1 ].nDim
	nPreShapeDim = dataList[ 0 ].pt[ 2 ].nDim 
	nRadiusDim = dataList[ 0 ].pt[ 3 ].nDim 
	
	# Total Dimension
	nManDim_Cov = nCenterDim + nScaleDime + nPreShapeDim + nRadiusDim

	S = np.zeros( [ nManDim_Cov, nManDim_Cov ] )

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
	# elif dataList[ 0 ].Type == "MRep":
	# 	return FrechetMean_MRep( dataList, maxIter, tol )
	else:
		print( "Manifold type is not known" )
		return -1

def GeodesicRegression_Sphere( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol = 1e-8, verbose=True ):
	nDimManifold = pt_list[ 0 ].nDim
	nData = len( pt_list )

	# Initial point on manifold and tangent vector
	init_Interp = manifolds.sphere( nDimManifold )
	init_tVec = manifolds.sphere_tVec( nDimManifold )

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

# To be updated
def GeodesicRegression_CMRep_Abstract( t_list, pt_list, max_iter = 500, stepSize = 0.01, step_tol =  1e-8, verbose=True ):
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


#############################################################################
###               Anchor Point Linearized Geodesic Regression             ###
#############################################################################

def LinearizedGeodesicRegression( t_list, pt_list, max_iter = 500, stepSize = 0.05, step_tol = 0.01, useFrechetMeanAnchor = False, verbose=False ):
	if pt_list[ 0 ].Type == "Sphere":
		return LinearizedGeodesicRegression_Sphere( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "PositiveReal":
		return LinearizedGeodesicRegression_PosReal( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "Euclidean":
		return LinearizedGeodesicRegression_Euclidean( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "CMRep":
		return LinearizedGeodesicRegression_CMRep( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
	elif pt_list[ 0 ].Type == "CMRep_Abstract":
		return LinearizedGeodesicRegression_CMRep_Abstract( t_list, pt_list, max_iter, stepSize, step_tol, useFrechetMeanAnchor, verbose )
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

	# Variance w.r.t the mean 
	for i in range( nData ):
		tVec_mu_to_y_i = mu.LogMap( pt_list[ i ] )
		tVec_mu_to_y_i.SetMeanRadius( mu.meanRadius )
		var_mu += ( tVec_mu_to_y_i.normSquared() / float( nData ) )

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

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )
		tVec_est_to_y_i.SetMeanRadius( est_pt_at_t_i.meanRadius )

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
		# Calculate intrinsic mean 
		print( "Calculating Frechet Mean... " )

		mu = FrechetMean( pt_list[ n ] )
		mu.UpdateMeanRadius()

		print( "Calculating Variance..." )

		# Variance w.r.t the mean 
		for i in range( nData ):
			tVec_mu_to_y_i = mu.LogMap( pt_list[ n ][ i ] )
			tVec_mu_to_y_i.SetMeanRadius( mu.meanRadius )
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
			tVec_est_to_y_i.SetMeanRadius( est_pt_at_t_i.meanRadius )

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

	for i in range( nData ):
		t_i = t_list[ i ]

		# Tangent Vector * time
		tVec_at_t_i = manifolds.cmrep_abstract_tVec( nManifoldDim )

		for j in range( 4 ):
			tVec_at_t_i.tVector[ j ] = tangent.tVector[ j ].ScalarMultiply( t_i )

		est_pt_at_t_i = base.ExponentialMap( tVec_at_t_i )

		tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ i ] )

		rmse += ( tVec_est_to_y_i.normSquared() / float( nData ) )

	return np.sqrt( rmse )


def RootMeanSquaredError_CMRep_Abstract_Array( t_list, pt_list, base, tangent ): 
	nObject = len( pt_list )
	nData = len( pt_list[0] )
	nManifoldDim = pt_list[0][ 0 ].nDim

	rmse = 0	

	for n in range( nObject ):
		for i in range( nData ):
			t_i = t_list[ i ]

			# Tangent Vector * time
			tVec_at_t_i = tangent[ n ].ScalarMultiply( t_i ) 

			est_pt_at_t_i = base[ n ].ExponentialMap( tVec_at_t_i )
			est_pt_at_t_i.UpdateMeanRadius()

			tVec_est_to_y_i = est_pt_at_t_i.LogMap( pt_list[ n ][ i ] )
			tVec_est_to_y_i.SetMeanRadius( est_pt_at_t_i.meanRadius )

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



'''

def CalculateIntrinsicMean( dataList, ManifoldType='MRep', maxIter=50, tol=0.01 ):
	if ManifoldType == 'MRep':
		return CalculateIntrinsicMean_MRep( dataList, maxIter )
	else:
		print( "Error: Unknown Manifold Type" )
		return -1

def CalculateIntrinsicMean_MRep( dataList, maxIter=50, tol=0.01 ):
	print( "=================================" ) 
	print( "Intrinsic Mean - M-Rep" )
	print( "=================================" )
	# Initialize the intrinsic mean
	mu = MReps.MRep()
	nManDim = dataList[ 0 ].nAtoms
	nData = len( dataList )

	for k in range( nManDim ):
		mu_atom = atom.mrep_atom()

		mu_atom.pos[ 0 ] = dataList[ 0 ].atom_list[k].pos[0]
		mu_atom.pos[ 1 ] = dataList[ 0 ].atom_list[k].pos[1]
		mu_atom.pos[ 2 ] = dataList[ 0 ].atom_list[k].pos[2]

		mu_atom.rad = dataList[ 0 ].atom_list[k].rad

		mu_atom.sphere_comp1.sphere_pt[0] = dataList[ 0 ].atom_list[k].sphere_comp1.sphere_pt[0]
		mu_atom.sphere_comp1.sphere_pt[1] = dataList[ 0 ].atom_list[k].sphere_comp1.sphere_pt[1]
		mu_atom.sphere_comp1.sphere_pt[2] = dataList[ 0 ].atom_list[k].sphere_comp1.sphere_pt[2]

		mu_atom.sphere_comp2.sphere_pt[0] = dataList[ 0 ].atom_list[k].sphere_comp2.sphere_pt[0]
		mu_atom.sphere_comp2.sphere_pt[1] = dataList[ 0 ].atom_list[k].sphere_comp2.sphere_pt[1]
		mu_atom.sphere_comp2.sphere_pt[2] = dataList[ 0 ].atom_list[k].sphere_comp2.sphere_pt[2]

		mu.AppendAtom( mu_atom )	

	for i in range( max_iter ):
		print( "=================================" ) 
		print( str( i ) + "th Iteration" )
		print( "=================================" )

		for k in range( nManDim ):
			dMu_k = atom.mrep_tVec()

			for j in range( nData ):
				Log_mu_M_j_k = mu.atom_list[k].LogMap( dataList[ j ].atom_list[ k ] )

				dMu_k.tVector[ 0 ][0] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][0] )
				dMu_k.tVector[ 0 ][1] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][1] )
				dMu_k.tVector[ 0 ][2] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][2] )

				dMu_k.tVector[ 1 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 1 ] )

				dMu_k.tVector[ 2 ].tVector[ 0 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 2 ].tVector[ 0 ] )
				dMu_k.tVector[ 2 ].tVector[ 1 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 2 ].tVector[ 1 ] )
				dMu_k.tVector[ 2 ].tVector[ 2 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 2 ].tVector[ 2 ] )

				dMu_k.tVector[ 3 ].tVector[ 0 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 3 ].tVector[ 0 ] )
				dMu_k.tVector[ 3 ].tVector[ 1 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 3 ].tVector[ 1 ] )
				dMu_k.tVector[ 3 ].tVector[ 2 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 3 ].tVector[ 2 ] )
				
			Mu_k = mu.atom_list[ k ].ExponentialMap( dMu_k )
			mu.atom_list[ k ] = Mu_k
	return mu

###################################################
##########  Principal Geodesic Analysis  ##########
###################################################
def PGA( dataList, mu, ManifoldType="MRep" ):
	if ManifoldType == "MRep":
		return PGA_MRep( dataList, mu )
	else:
		print( "Unknown Manifold Type" )
		return -1


def PGA_MRep( dataList, mu, nPC=2 ):
	print( "=====================================" ) 
	print( " Principal Geodesic Analysis - M-Rep " )
	print( "=====================================" )


	S = np.zeros( [ 10 * nManDim,  10 * nManDim ] ) 

	for j in range( nData ):
		u_j_arr = []

		for i in range( nManDim ):
			u_j = mu.atom_list[ i ].LogMap( mRepDataList[ j ].atom_list[ i ] )

			u_j_arr.append( [ u_j.tVector[0][0], u_j.tVector[0][1], u_j.tVector[0][2], u_j.tVector[1], u_j.tVector[2].tVector[0], u_j.tVector[2].tVector[1], u_j.tVector[2].tVector[2], u_j.tVector[3].tVector[0], u_j.tVector[3].tVector[1], u_j.tVector[3].tVector[2] ] )

		u_j_mat = np.asmatrix( u_j_arr )
		u_j_mat = u_j_mat.flatten()
		u_j_u_j_t = np.dot( u_j_mat.T, u_j_mat )

		# print( u_j_mat.shape )
		# print( u_j_u_j_t )
	 
		S = np.add( S, np.multiply( 1.0/nData, u_j_u_j_t ) )


	[ w, v ] = np.linalg.eig( S )

	w_sortIdx = np.abs( w ).argsort()
	w = w[ w_sortIdx[ ::-1] ]
	v = v[ :, w_sortIdx[ ::-1 ] ] 
	w = np.real( w ) 
	v = np.real( v ) 

	return w[0:nPC], v[:, 0:nPC]

###################################################
##########  Principal Geodesic Analysis  ##########
##########  Miscelleneous                ##########
###################################################
def PGA_Reconstruction( dataList, v, mu, ManifoldType="MRep" ):
	if ManifoldType == "MRep":
		return PGA_Reconstruction_MRep( dataList, v, mu )
	else:
		print( "Unknown Manifold Type" )
		return -1

def PGA_Reconstruction_MRep( dataList, v, mu ):
	print( "================================================" ) 
	print( " Principal Geodesic Analysis Projection - M-Rep " )
	print( "================================================" )

	nPC = v.shape[1]
	nData = len( dataList )
	nManDim = dataList[ 0 ].nAtoms
	residual_percent_mean = 0

	w_est = np.zeros( [ nData, nPC ] )

	for j in range( nData ):
		u_j_arr = []
		meanRad = 0

		for i in range( nManDim ):
			u_j = mu.atom_list[ i ].LogMap( mRepDataList[ j ].atom_list[ i ] )
			u_j_arr.append( [ u_j.tVector[0][0], u_j.tVector[0][1], u_j.tVector[0][2], u_j.tVector[1], u_j.tVector[2].tVector[0], u_j.tVector[2].tVector[1], u_j.tVector[2].tVector[2], u_j.tVector[3].tVector[0], u_j.tVector[3].tVector[1], u_j.tVector[3].tVector[2] ] )
			meanRad += ( ( 1.0 / nManDim ) * mu.atom_list[ i ].rad )

		u_j_mat = np.asmatrix( u_j_arr )
		u_j_arr = u_j_mat.flatten()
		u_j_arr_res = u_j_arr

		for k in nPC:
			w_k_j = np.dot( u_j_arr_res, v[ :, k ] )
			u_j_arr_res = np.subtract( u_j_arr, np.multiply( w_1_j.flatten(), v[ :, 0 ] ).T )	
			w_est[ j, k ] = w_k_j

		u_j_arr_recon = np.zeros( [ 1, v.shape[0] ] ) 

		for k in nPC: 
			u_j_arr_recon = np.add( u_j_arr_recon, np.multiply( w_est[ j, k ], v[ :, 1 ] ).T )

		u_j_arr_res = np.subtract( u_j_arr, u_j_arr_recon )

		residual = 0
		u_j_norm = 0

		for i in range( 4 ):
			if i == 0:
				pos_x_arr_u_j = u_j_arr[ :, ::10 ].flatten()
				pos_y_arr_u_j = u_j_arr[ :, 1::10 ].flatten()
				pos_z_arr_u_j = u_j_arr[ :, 2::10 ].flatten() 
				pos_arr_u_j = np.asarray( [ pos_x_arr_u_j, pos_y_arr_u_j, pos_z_arr_u_j ] ).flatten()

				pos_x_arr_u_j_res = u_j_arr_res[ :, ::10 ]
				pos_y_arr_u_j_res = u_j_arr_res[ :, 1::10 ]
				pos_z_arr_u_j_res = u_j_arr_res[ :, 2::10 ]
				pos_arr_u_j_res = np.asarray( [ pos_x_arr_u_j_res, pos_y_arr_u_j_res, pos_z_arr_u_j_res ] ).flatten()

				residual += ( np.linalg.norm( pos_arr_u_j_res ) ** 2 )
				u_j_norm += ( np.linalg.norm( pos_arr_u_j ) ** 2 )

			elif i == 1:
				rho_arr_u_j = u_j_arr[ :, 3::10 ].flatten()
				rho_arr_u_j_res = u_j_arr_res[ :, 3::10 ].flatten()

				residual += ( ( meanRad * np.linalg.norm( rho_arr_u_j_res ) ) ** 2 )
				u_j_norm += ( ( meanRad * np.linalg.norm( rho_arr_u_j ) ) ** 2 )

			elif i == 2:
				sphere_pt1_x_arr_u_j = u_j_arr[ :, 4::10 ]
				sphere_pt1_y_arr_u_j = u_j_arr[ :, 5::10 ]
				sphere_pt1_z_arr_u_j = u_j_arr[ :, 6::10 ]
				sphere_pt1_arr_u_j = np.asarray( [ sphere_pt1_x_arr_u_j, sphere_pt1_y_arr_u_j, sphere_pt1_z_arr_u_j ] ).flatten()

				sphere_pt1_x_arr_u_j_res = u_j_arr_res[ :, 4::10 ]
				sphere_pt1_y_arr_u_j_res = u_j_arr_res[ :, 5::10 ]
				sphere_pt1_z_arr_u_j_res = u_j_arr_res[ :, 6::10 ]
				sphere_pt1_arr_u_j_res = np.asarray( [ sphere_pt1_x_arr_u_j_res, sphere_pt1_y_arr_u_j_res, sphere_pt1_z_arr_u_j_res ] ).flatten()

				residual += ( ( meanRad * np.linalg.norm( sphere_pt1_arr_u_j_res ) ) ** 2 )
				u_j_norm += ( ( meanRad * np.linalg.norm( sphere_pt1_arr_u_j ) ) ** 2 )

			elif i == 3:
				sphere_pt2_x_arr_u_j = u_j_arr[ :, 7::10 ]
				sphere_pt2_y_arr_u_j = u_j_arr[ :, 8::10 ]
				sphere_pt2_z_arr_u_j = u_j_arr[ :, 9::10 ]
				sphere_pt2_arr_u_j = np.asarray( [ sphere_pt2_x_arr_u_j, sphere_pt2_y_arr_u_j, sphere_pt2_z_arr_u_j ] ).flatten()

				sphere_pt2_x_arr_u_j_res = u_j_arr_res[ :, 7::10 ]
				sphere_pt2_y_arr_u_j_res = u_j_arr_res[ :, 8::10 ]
				sphere_pt2_z_arr_u_j_res = u_j_arr_res[ :, 9::10 ]
				sphere_pt2_arr_u_j_res = np.asarray( [ sphere_pt2_x_arr_u_j_res, sphere_pt2_y_arr_u_j_res, sphere_pt2_z_arr_u_j_res ] ).flatten()

				residual += ( ( meanRad * np.linalg.norm( sphere_pt2_arr_u_j_res ) ) ** 2 )
				u_j_norm += ( ( meanRad * np.linalg.norm( sphere_pt2_arr_u_j ) ) ** 2 )

		residual_percent = np.divide( residual, u_j_norm )
		residual_percent_mean += ( residual_percent / nData )

	print( "Mean Residual : " + str( residual_percent_mean ) )
	return w_est
'''
