# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds
import numpy as np

# Visualization
import vtk

# Stats Model
import statsmodels.api as sm

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
	rand_tVec = manifolds.pos_real_tVec()
	rand_tVec.SetTangentVector( rand_pt )

	v_t = manifolds.pos_real_tVec() 
	v_t.SetTangentVector( v_slope.tVector * time_pt )
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
print( " Linearized Model on Anchored Tangent Space" )
print( "=============================================" ) 

# # Calculate Intrinsic Mean
# print( "====================================" )
# print( "Calculate Intrinsic Mean" )
# print( "====================================" ) 
# max_iter = 100
# tol = 0.1

# # Initialize
# mu = manifolds.sphere_atom()
# max_iter = 100

# for i in range( max_iter ):
# 	print( "=================================" ) 
# 	print( str( i ) + "th Iteration" )
# 	print( "=================================" )

# 	dMu_k = manifolds.sphere_tVec()

# 	for g_i in range( 5 ):
# 		pt_list = pt_group_list[ g_i ]

# 		for j in range( nData ):
# 			Log_mu_M_j_k = mu.LogMap( pt_list[ j ] )

# 			dMu_k.tVector[ 0 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ] )
# 			dMu_k.tVector[ 1 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 1 ] )
# 			dMu_k.tVector[ 2 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 2 ] )
				
# 			Mu_k = mu.ExponentialMap( dMu_k )
# 			mu = Mu_k

mu = pt_list[0]

print( mu.pt )

print( "====================================" )
print( " Project data to Intrinsic Mean" )
print( "====================================" ) 

tVec_list = []
w_1 = [] 
time_list = []


for j in range( nData ):
	tVec_j = mu.LogMap( pt_list[ j ] )

	u_j_arr = []
	u_j_arr = [ tVec_j.tVector ] 

	w_1_j = tVec_j.tVector
	w_1.append( w_1_j )

	time_list.append( t_list[ j ] )


print( "======================================" )
print( " Linear Regression on Tangent Vectors " )
print( "======================================" ) 

t_list_sm = sm.add_constant( t_list )

LS_model1 = sm.OLS( w_1, t_list_sm )
est1 = LS_model1.fit()
print( est1.summary() )

print( "=========================================================================" )
print( " Mapping Linear Regression Result on A-Tangent Space Results to Manifold" )
print( "=========================================================================" ) 

nTimePt = 100
est_trend_pt_list_add = []
est_trend_pt_vis_list_add = []
est_trend_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0
	
	est1_fitted = np.add( np.multiply( time_pt, est1.params[1] ), est1.params[0] )

	tVec_t = manifolds.pos_real_tVec()
	tVec_t.tVector = est1_fitted

	pt_t = mu.ExponentialMap( tVec_t )
	est_trend_pt_list_add.append( pt_t )
	est_trend_pt_vis_list_add.append( pt_t.pt )

	est_trend_t_list.append( time_pt )


AReg_tVec_v_wp = manifolds.pos_real_tVec()
AReg_tVec_v_wp.tVector = est1.params[0]

AReg_tVec_v_w = manifolds.pos_real_tVec()
AReg_tVec_v_w.tVector = est1.params[1]

AReg_p_interp = mu.ExponentialMap( AReg_tVec_v_wp )
AReg_tVec_v_p = mu.ParallelTranslateAtoB( mu, AReg_p_interp, AReg_tVec_v_w ) 

print( "=====================================" )
print( "            Ground Truth ")
print( "=====================================" )

print( "True P" )
print( p_interp.pt )
print( "True V" )
print( v_slope.tVector )

print( "============================================" )
print( "   Anchor Point LME Model Results ")
print( "============================================" )

print( "Estimated P" )
print( AReg_p_interp.pt )
print( "Estimated V" ) 
print( AReg_tVec_v_p.tVector )





# Validations 
# Measure Geodesic Distance and Approximated Distance of data from the ground truth geodesic 
diff_mean = 0
diff_max = -1000
diff_min = 10000
diff_list = []

approx_dist_mean = 0 
approx_dist_max = -1000
approx_dist_min = 10000
approx_dist_list = []

geo_dist_mean = 0 
geo_dist_max = -1000
geo_dist_min = 10000
geo_dist_list = []

print( "=================================================" )
print( " Comparison : Geodesic Distance vs Approx Dist. ")
print( "=================================================" )

for n in range( nData ):
	# Data
	pt_n = pt_list[ n ]
	t_n = t_list[ n ]

	# A corresponding point on a ground truth geodesic
	v_t = manifolds.pos_real_tVec() 
	v_t.SetTangentVector( v_slope.tVector * t_n )

	mean = p_interp.ExponentialMap( v_t )

	# Geodesic Distance
	geo_dist = mean.norm( mean.LogMap( pt_n ) )

	# Approximated Distance
	mu_tVec_approx = manifolds.pos_real_tVec()
	mu_tVec_approx_list = mu.LogMap( mean ).tVector - mu.LogMap( pt_n ).tVector
	mu_tVec_approx.SetTangentVector( mu_tVec_approx_list )

	approx_dist = mean.norm( mu_tVec_approx )

	diff = np.abs( geo_dist - approx_dist )

	diff_mean += ( diff / nData )

	if diff_max < diff:
		diff_max = diff
	if diff_min > diff:
		diff_min = diff

	diff_list.append( diff )

	approx_dist_mean += ( approx_dist / nData )

	if approx_dist_max < approx_dist:
		approx_dist_max = approx_dist

	if approx_dist_min > approx_dist:
		approx_dist_min = approx_dist

	approx_dist_list.append( approx_dist )


	geo_dist_mean += ( geo_dist / nData )

	if geo_dist_max < geo_dist:
		geo_dist_max = geo_dist

	if geo_dist_min > geo_dist:
		geo_dist_min = geo_dist

	geo_dist_list.append( geo_dist )


# Difference
print( "Difference" )
print( "Std" )
print( np.std( diff_list ) )
print( "Mean" )
print( diff_mean )
print( "Max" )
print( diff_max )
print( "Min" )
print( diff_min )

# Approximated Distance
print( "Approximated Distance" )
print( "Std" )
print( np.std( approx_dist_list ) )
print( "Mean" )
print( approx_dist_mean )
print( "Max" )
print( approx_dist_max )
print( "Min" )
print( approx_dist_min )


# Geodesic Distance
print( "Geodesic Distance" )
print( "Std" )
print( np.std( geo_dist_list ) )
print( "Mean" )
print( geo_dist_mean )
print( "Max" )
print( geo_dist_max )
print( "Min" )
print( geo_dist_min )


######################################
#   		Visualization     		 #
######################################

plt.figure()
colors = [ [ 1, 0, 0 ], [ 1.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 1.0, 0, 1.0 ], [ 0, 0, 1.0 ] ] 
est_colors =[ [ 0, 1, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  

trend_pt_list = []
trend_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	v_t = manifolds.pos_real_tVec() 
	v_t.SetTangentVector( v_slope.tVector * time_pt )
	mean = p_interp.ExponentialMap( v_t )

	trend_pt_list.append( mean.pt )
	trend_t_list.append( time_pt )

plt.plot( trend_t_list, trend_pt_list, c=colors[ 0 ] )
plt.scatter( t_list, pt_val_list, c=colors[ 0 ] )

plt.plot( est_trend_t_list, est_trend_pt_vis_list_add, c=est_colors[ 0 ] )

plt.show()

