# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds
import numpy as np

# Visualization
import vtk

# Stats Model
import StatsModel as sm 


# Pandas
import pandas as pd
import matplotlib.pyplot as plt

# Ground Truth
nManifoldDim = 1 

p_interp = manifolds.pos_real( nManifoldDim )
v_slope = manifolds.pos_real_tVec( nManifoldDim )

p_interp_vec = 2.0

p_interp.SetPoint( p_interp_vec )
v_slope.SetTangentVector( 2.0 )

## Random Ground Truth Generation
# random_interp = np.random.rand(3)
# random_interp_n = np.divide( random_interp, np.linalg.norm( random_interp ) )

# random_tangent_vector = np.random.rand(3)
# random_scale = np.random.rand(1) * 2
# random_tangent_vector = np.multiply( random_tangent_vector, random_scale )

# p_interp.SetSpherePt( random_interp_n )
# v_slope.SetTangentVector( random_tangent_vector )

# Generating sphere atoms distributed over time perturbed by Gaussian random
# Time
t0 = 0
t1 = 1
nTimePt = 30

# Generate a random point on the manifold
nData = 100
dim = 1
sigma = 0.05

pt_list = []
pt_val_list = []
t_list = []

for n in range( nData ):
	time_pt = np.random.uniform( t0, t1 )
	# time_pt = ( t1 - t0 ) * n / nData + t0

	# Generate a random Gaussians with polar Box-Muller Method
	rand_pt = 0

	r2 = 0
	x = 0
	y = 0

	while( r2 > 1.0 or r2 == 0 ):
		x = ( 2.0 * np.random.rand() - 1.0 )
		y = ( 2.0 * np.random.rand() - 1.0 )
		r2 = x * x + y * y 

	gen_rand_no = sigma * y * np.sqrt( -2.0 * np.log( r2 ) / r2 )
	rand_pt = gen_rand_no
	print( rand_pt )

	# Set Random Vector to Tangent Vector - ListToTangent
	rand_tVec = manifolds.pos_real_tVec( nManifoldDim )
	rand_tVec.SetTangentVector( rand_pt )

	v_t = manifolds.pos_real_tVec( nManifoldDim )
	v_t.SetTangentVector( np.multiply( v_slope.tVector, time_pt ).tolist() )
	mean = p_interp.ExponentialMap( v_t )

	# print( "Mean At Time : " + str( time_pt ) )	
	# print( mean.sphere_pt )

	# Projected Tangent to Mean Point
	rand_tVec_projected = mean.ProjectTangent( mean, rand_tVec )

	# print( "Random Tangent" )
	# print( rand_tVec.tVector )

	# print( "Projected Random Tangent" )
	# print( rand_tVec_projected.tVector )

	# Perturbed point at time_pt 
	pt_perturbed = mean.ExponentialMap( rand_tVec_projected )

	# print( "Perturbed pt At Time : " + str( time_pt ) )	
	# print( pt_perturbed.sphere_pt )

	pt_list.append( pt_perturbed )
	pt_val_list.append( pt_perturbed.pt )
	t_list.append( time_pt )


##########################
#  Anchored Point Linear Regression  #
##########################
print( "=============================================" )
print( " Geodesic Regression Model " )
print( "=============================================" ) 

# Gradient Descent Parameters
step_size = 0.1
max_iter = 500
step_tol = 1e-8

base, tangent = sm.GeodesicRegression( t_list, pt_list, max_iter, step_size, step_tol )

AReg_p_interp = base
AReg_tVec_v_p = tangent

print( "=============================================" )
print( " Validations " )
print( "=============================================" ) 

print( "True P" )
print( p_interp.pt )
print( "True V" )
print( v_slope.tVector )

print( "Estimated P" )
print( base.pt )
print( "Estimated V" ) 
print( tangent.tVector )

R2 = sm.R2Statistics( t_list, pt_list, base, tangent )

print( "R2 Statistics" )
print( R2 )

RMSE = sm.RootMeanSquaredError( t_list, pt_list, base, tangent )

print( "RMSE" )
print( RMSE )

# Permutation Test
nTrial = 10000

print( "======================================" )
print( "Random Permutation Testing........    " )
print( ( "# of Trials : %d", nTrial )  )
print( "======================================" )

P_val = sm.NullHypothesisTestingPermutationTest( t_list, pt_list, base, tangent, nTrial, max_iter, step_size, step_tol )

print( "P-Value" )
print( P_val )


######################################
#   		Visualization     		 #
######################################
plt.figure()
colors = [ [ 1, 0, 0 ], [ 1.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 1.0, 0, 1.0 ], [ 0, 0, 1.0 ] ] 
est_colors =[ [ 0, 1, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  

trend_pt_list = []
trend_t_list = []

est_pt_list = []
est_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	v_t = manifolds.pos_real_tVec( nManifoldDim )
	v_t.SetTangentVector( np.multiply( v_slope.tVector, time_pt ).tolist() )
	mean = p_interp.ExponentialMap( v_t )

	trend_pt_list.append( mean.pt )
	trend_t_list.append( time_pt )


	v_est_t = manifolds.pos_real_tVec( nManifoldDim )
	v_est_t.SetTangentVector( np.multiply( tangent.tVector, time_pt ).tolist() )
	est_pt_t = base.ExponentialMap( v_est_t )

	est_pt_list.append( est_pt_t.pt )
	est_t_list.append( time_pt ) 

plt.plot( trend_t_list, trend_pt_list, c=colors[ 0 ] )
plt.scatter( t_list, pt_val_list, c=colors[ 0 ] )
plt.plot( est_t_list, est_pt_list, c=est_colors[ 0 ] )

plt.show()

