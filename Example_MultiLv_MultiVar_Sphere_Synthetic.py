################################################################################################
################################################################################################
# Synthetic Example - MLMG Model                                         					   #
# Manifold - Sphere Manifold 																   #
# Ind. Variables - age, sex, CAG repeat length 												   #	
#																							   #
# Model - y = ( beta_0 + beta_1 s + beta_2 c ) + gamma_0 t + gamma_1 st + gamma_2 ct		   #	
# Model explanation - sex affects an intercept point and slope 								   #	
# 					  CAG repeat length affects an intercept,								   #
# 					  a shape changes in a common direction and magnitude for each sex,        #
#                     but initial baseline shapes of male and female are different,			   #
#                     CAG repeat lenght affects initial baseline shapes						   #
################################################################################################
################################################################################################

import manifolds 
import numpy as np
import StatsModel as sm

import matplotlib.pyplot as plt
# Visualization
import vtk

# Parameters
nDimManifold = 3

nSubject = 500

# General Data Noise Parameter
pt_sigma = 0.0001

# Individual Noise 
a_sigma = 0.005
b_sigma = 0.001

# Time
t0 = 20.0
t1 = 70.0

# CAG Repeath Length Range
c0 = 0.0
c1 = 10.0

# Sex
s0 = 0
s1 = 1

# Time Interval 
t_int0 = 0.5
t_int1 = 2

# Number of Observations
nObs0 = 2
nObs1 = 5

# # Visualize group level grount truth
# Curve Visualization Parameter 
nLineTimePt = 100

# Ground Truth
# Intercept Coefficeints
beta0 = manifolds.sphere( nDimManifold )
beta1 = manifolds.sphere_tVec( nDimManifold )
beta2 = manifolds.sphere_tVec( nDimManifold )

# Intercept Point
beta0.SetPoint( [ 1.0, 0.0, 0.0 ] )

# A tangent vector for different sex
beta1.SetTangentVector( [ 0.0, 0.2, 0.0 ] )

# A tangent vector for CAG repeat length
# beta2.SetTangentVector( [ 0.0, 0.05, 0.0 ] )
beta2.SetTangentVector( [ 0.0, 0.0, 0.15 ] )

# beta2.SetTangentVector( [ 0.0, 0.0, -0.2 ] )

# Slope Coefficients
gamma0 = manifolds.sphere_tVec( nDimManifold )
gamma1 = manifolds.sphere_tVec( nDimManifold )
gamma2 = manifolds.sphere_tVec( nDimManifold )

# A slope tangent vector for age
gamma0.SetTangentVector( [ 0.0, 0.0, 0.01 ] )

# A slope tangent vector for sex and age
gamma1.SetTangentVector( [ 0.0, 0.005, -0.002 ] )

# A slope tangent vector for CAG and age
gamma2.SetTangentVector( [ 0.0, 0.002, 0.0 ] )

# # A tangent vector for different sex
# beta1.SetTangentVector( [ 0.0, 0.2, 0.0 ] )

# # A tangent vector for CAG repeat length
# beta2.SetTangentVector( [ 0.0, 0.05, 0.0 ] )

# # A slope tangent vector for age
# beta3.SetTangentVector( [ 0.0, 0.0, 0.01 ] )

# # A slope tangent vector for sex and age
# beta4.SetTangentVector( [ 0.0, 0.005, -0.002 ] )

# # A slope tangent vector for CAG and age
# beta5.SetTangentVector( [ 0.0, 0.001, 0.0 ] )


# paramCurve_t0 = 0.0
# paramCurve_t1 = 1.0

nCAG_Lines = 5

group_vtk_list = []

group_vtk_list_s1 = []

group_c_list_s1 = []

# Group 1 - s = 0
c_step = ( c1 - c0 ) / float( nCAG_Lines )

for c in range( nCAG_Lines ):
	c_pt =  ( c + 0.5 ) * c_step + c0 

	group_c_list_s1.append( c_pt )

	group_geodesic_pt_list = []

	beta2_c = beta2.ScalarMultiply( c_pt )

	p0_c = beta0.ExponentialMap( beta2_c )

	gamma0_tilde = beta0.ParallelTranslateAtoB( beta0, p0_c, gamma0 )
	gamma2_tilde = beta0.ParallelTranslateAtoB( beta0, p0_c, gamma2 )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( gamma0_tilde.tVector[ 0 ] + gamma2_tilde.tVector[ 0 ] * c_pt ) * time_pt, ( gamma0_tilde.tVector[ 1 ] + gamma2_tilde.tVector[ 1 ] * c_pt ) * time_pt, ( gamma0_tilde.tVector[ 2 ] + gamma2_tilde.tVector[ 2 ] * c_pt ) * time_pt ] )

		p_t = p0_c.ExponentialMap( v_t )
		group_geodesic_pt_list.append( p_t )

	group_geodesic_vtk = vtk.vtkPolyData()
	group_geodesic_pts = vtk.vtkPoints()

	for t in range( len( group_geodesic_pt_list ) ):
		group_geodesic_pts.InsertNextPoint( group_geodesic_pt_list[ t ].pt[ 0 ], group_geodesic_pt_list[ t ].pt[ 1 ], group_geodesic_pt_list[ t ].pt[ 2 ] )

	group_geodesic_line = vtk.vtkCellArray()
	for t in range( len( group_geodesic_pt_list ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		group_geodesic_line.InsertNextCell( line_i )

	group_geodesic_vtk.SetPoints( group_geodesic_pts )
	group_geodesic_vtk.SetLines( group_geodesic_line )

	group_vtk_list_s1.append( group_geodesic_vtk )

# Group 2 - s = 1
group_vtk_list_s2 = []

group_c_list_s2 = []
c_step = ( c1 - c0 ) / float( nCAG_Lines )

for c in range( nCAG_Lines ):
	c_pt = ( c + 0.5 ) * c_step + c0 

	group_c_list_s2.append( c_pt )

	group_geodesic_pt_list2 = []

	beta_c = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		beta_c.tVector[ k ] = beta1.tVector[ k ] + beta2.tVector[ k ] * c_pt

	p_1_c = beta0.ExponentialMap( beta_c )

	gamma0_at_p_1_c = beta0.ParallelTranslateAtoB( beta0, p_1_c, gamma0 )
	gamma1_at_p_1_c = beta0.ParallelTranslateAtoB( beta0, p_1_c, gamma1 )
	gamma2_at_p_1_c = beta0.ParallelTranslateAtoB( beta0, p_1_c, gamma2 )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( gamma0_at_p_1_c.tVector[ 0 ] + gamma1_at_p_1_c.tVector[ 0 ] + gamma2_at_p_1_c.tVector[ 0 ] * c_pt ) * time_pt, ( gamma0_at_p_1_c.tVector[ 1 ] + gamma1_at_p_1_c.tVector[ 1 ] + gamma2_at_p_1_c.tVector[ 1 ] * c_pt ) * time_pt, ( gamma0_at_p_1_c.tVector[ 2 ] + gamma1_at_p_1_c.tVector[ 2 ] + gamma2_at_p_1_c.tVector[ 2 ] * c_pt ) * time_pt ] )

		p_t = p_1_c.ExponentialMap( v_t )
		group_geodesic_pt_list2.append( p_t )

	group_geodesic_vtk2 = vtk.vtkPolyData()
	group_geodesic_pts2 = vtk.vtkPoints()

	for t in range( len( group_geodesic_pt_list2 ) ):
		group_geodesic_pts2.InsertNextPoint( group_geodesic_pt_list2[ t ].pt[ 0 ], group_geodesic_pt_list2[ t ].pt[ 1 ], group_geodesic_pt_list2[ t ].pt[ 2 ] )

	group_geodesic_line2 = vtk.vtkCellArray()
	for t in range( len( group_geodesic_pt_list2 ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		group_geodesic_line2.InsertNextCell( line_i )

	group_geodesic_vtk2.SetPoints( group_geodesic_pts2 )
	group_geodesic_vtk2.SetLines( group_geodesic_line2 )

	group_vtk_list_s2.append( group_geodesic_vtk2 )

group_vtk_list = [ group_vtk_list_s1, group_vtk_list_s2 ]
group_c_list = [ group_c_list_s1, group_c_list_s2 ]

# Point List with Covariates
pt_list = []
t_list = []

cov_int_list = []
cov_slope_list = []

# Comparison - w.o. Covariate
cov_int_list_null = []
cov_slope_list_null = []

# Lists for Visualization
pt_list_all = [] 
t_list_all = []
s_list_all = []
c_list_all = []

c_step = ( c1 - c0 ) / float( nCAG_Lines )
c_pt = ( c + 0.5 ) * c_step + c0 

for i in range( nSubject ):
	pt_list_i = []
	t_list_i = []

	s_i = np.random.random_integers( s0, s1 )
	c_i = np.random.uniform( c0, c1 )

	nObs_i = np.random.random_integers( nObs0, nObs1 )
	
	t0_i = np.random.uniform( t0, t1 )	

	beta_i = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		beta_i.tVector[ k ] = beta1.ScalarMultiply( s_i ).tVector[ k ] + beta2.ScalarMultiply( c_i ).tVector[ k ] 

	f_i = beta0.ExponentialMap( beta_i )

	a_i = sm.GaussianNoisePerturbation( f_i, a_sigma )

	gamma1_s = gamma1.ScalarMultiply( s_i )
	gamma2_c = gamma2.ScalarMultiply( c_i )

	gamma_i = manifolds.sphere_tVec( nDimManifold )
	gamma_pert_i = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		gamma_i.tVector[ k ] = gamma0.tVector[ k ] + gamma1_s.tVector[ k ] + gamma2_c.tVector[ k ]

		if k == 0:
			gamma_pert_i.tVector[ k ] = gamma_i.tVector[ k ]
		else:
			gamma_pert_i.tVector[ k ] = np.random.normal( gamma_i.tVector[ k ], b_sigma )

	b_i_f_i = beta0.ParallelTranslateAtoB( beta0, f_i, gamma_pert_i )
	b_i = f_i.ParallelTranslateAtoB( f_i, a_i, b_i_f_i )

	for j in range( nObs_i ):
		t_ij = 0

		if j == 0:
			t_ij = t0_i
			t_list_i.append( t_ij )			
		else:
			t_int_ij = np.random.uniform( t_int0, t_int1 )

			t_ij = t_list_i[ j - 1 ] + t_int_ij

			t_list_i.append( t_ij )

		b_i_t_ij = b_i.ScalarMultiply( t_ij )

		p_ij = a_i.ExponentialMap( b_i_t_ij )

		p_ij_pert = sm.GaussianNoisePerturbation( p_ij, pt_sigma )

		# # Check no point-wise perturbation
		# p_ij_pert = p_ij

		pt_list_i.append( p_ij_pert )
		
		pt_list_all.append( p_ij_pert )
		t_list_all.append( t_ij ) 
		s_list_all.append( s_i )
		c_list_all.append( c_i )		

	pt_list.append( pt_list_i ) 
	t_list.append( t_list_i ) 	
	cov_int_list.append( [ s_i, c_i ] )
	cov_slope_list.append( [ s_i, c_i ] ) 

	cov_int_list_null.append( [0, 0] )
	cov_slope_list_null.append( [0, 0] ) 


print( len( pt_list ) )
print( len( t_list ) )
print( len( cov_int_list ) )
print( len( cov_slope_list ) )

# cov_int_list = [] 
# cov_slope_list = []


# group_pt_list = group1_pt_list + group2_pt_list
# group_t_list = group1_t_list + group2_t_list
# group_s_list = group1_s_list + group2_s_list

# # group_ind_var_list = [] 

# # for i in range( len( group_pt_list ) ):
# # 	group_ind_var_list.append( [ group_s_list[ i ], group_t_list[ i ] ] )

# t_list = [ group1_t_list, group2_t_list ]
# pt_list = [ group1_pt_list, group2_pt_list ]
# cov_int_list = [ group1_cov_int_list, group2_cov_int_list ]
# cov_slope_list = [ group1_cov_slope_list, group2_cov_slope_list ] 

p0_group_list = []
v_group_list = []

# Individual Group Estimation
for g in range( len( pt_list ) ):
	t_list_g = t_list[ g ]
	pt_list_g = pt_list[ g ] 

	p0_g, v_g = sm.LinearizedGeodesicRegression_Sphere( t_list_g, pt_list_g, verbose=False )

	print( "v_g.tVector" )		
	print( v_g.tVector )

	p0_group_list.append( p0_g )
	v_group_list.append( v_g )

# Individual Estimated Results
ind_est_group_vtk_list = []

for g in range( len( pt_list ) ):
	ind_est_group_geodesic_pt_list = []

	t_0_g = t_list[ g ][ 0 ]
	t_1_g = t_list[ g ][ -1 ]

	for t in range( nLineTimePt ):
		time_pt = ( t_1_g - t_0_g ) * t / nLineTimePt + t_0_g

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ v_group_list[ g ].tVector[ 0 ] * time_pt, v_group_list[ g ].tVector[ 1 ] * time_pt, v_group_list[ g ].tVector[ 2 ] * time_pt ] )

		p_t = p0_group_list[ g ].ExponentialMap( v_t )
		ind_est_group_geodesic_pt_list.append( p_t )

	ind_est_group_geodesic_vtk = vtk.vtkPolyData()
	ind_est_group_geodesic_pts = vtk.vtkPoints()

	for t in range( len( ind_est_group_geodesic_pt_list ) ):
		ind_est_group_geodesic_pts.InsertNextPoint( ind_est_group_geodesic_pt_list[ t ].pt[ 0 ], ind_est_group_geodesic_pt_list[ t ].pt[ 1 ], ind_est_group_geodesic_pt_list[ t ].pt[ 2 ] )

	ind_est_group_geodesic_line = vtk.vtkCellArray()

	for t in range( len( ind_est_group_geodesic_pt_list ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		ind_est_group_geodesic_line.InsertNextCell( line_i )

	ind_est_group_geodesic_vtk.SetPoints( ind_est_group_geodesic_pts )
	ind_est_group_geodesic_vtk.SetLines( ind_est_group_geodesic_line )

	ind_est_group_vtk_list.append( ind_est_group_geodesic_vtk )

# print( len( ind_est_group_vtk_list ) )

# Multi-Step Population Level Estimation

# print( "cov_int_list" ) 
# print( cov_int_list )
# print( "cov_slope_list" ) 
# print( cov_slope_list ) 

est_beta0, tangent_intercept_arr, tangent_slope_arr = sm.MultivariateLinearizedGeodesicRegression_Sphere_BottomUp( t_list, pt_list, cov_int_list, cov_slope_list, max_iter=10, verbose=True )

# print( "tangent_slope_arr[ 0 ].tVector")
# print( tangent_slope_arr[ 0 ].tVector )
# print( "tangent_slope_arr[ 1 ].tVector")
# print( tangent_slope_arr[ 1 ].tVector )
# print( "tangent_slope_arr[ 2 ].tVector")
# print( tangent_slope_arr[ 2 ].tVector )

# print( "tangent_intercept_arr[ 0 ].tVector")
# print( tangent_intercept_arr[0].tVector )
# print( "tangent_intercept_arr[ 1 ].tVector")
# print( tangent_intercept_arr[1].tVector )

# print( "beta0.tVector")
# print( beta0.pt )

base = est_beta0

est_beta1 = manifolds.sphere_tVec( nDimManifold )
est_beta1.SetTangentVector( tangent_intercept_arr[ 0 ].tVector )

est_beta2 = manifolds.sphere_tVec( nDimManifold )
est_beta2.SetTangentVector( tangent_intercept_arr[ 1 ].tVector )
 
est_gamma0 = manifolds.sphere_tVec( nDimManifold )
est_gamma0.SetTangentVector( tangent_slope_arr[ 2 ].tVector )

est_gamma1 = manifolds.sphere_tVec( nDimManifold )
est_gamma1.SetTangentVector( tangent_slope_arr[ 0 ].tVector )

est_gamma2 = manifolds.sphere_tVec( nDimManifold )
est_gamma2.SetTangentVector( tangent_slope_arr[ 1 ].tVector )

# # print( "Beta2 : GT - Estimated Diff" ) 
# # print( np.sqrt( ( est_beta2.tVector[ 0 ] - beta2.tVector[ 0 ] ) ** 2 + ( est_beta2.tVector[ 1 ] - beta2.tVector[ 1 ] ) ** 2 + ( est_beta2.tVector[ 2 ] - beta2.tVector[ 2 ] ) ** 2 ) )

# # beta2_v1_diff = np.sqrt( ( v_group_list[0].tVector[ 0 ] - beta2.tVector[ 0 ] ) ** 2 + ( v_group_list[0].tVector[ 1 ] - beta2.tVector[ 1 ] ) ** 2 + ( v_group_list[0].tVector[ 2 ] - beta2.tVector[ 2 ] ) ** 2 )
# # beta2_v2_diff = np.sqrt( ( v_group_list[1].tVector[ 0 ] - base.ParallelTranslate( est_beta1, beta2 ).tVector[ 0 ] ) ** 2 + ( v_group_list[1].tVector[ 1 ] - base.ParallelTranslate( est_beta1, beta2 ).tVector[ 1 ] ) ** 2 + ( v_group_list[1].tVector[ 2 ] - base.ParallelTranslate( est_beta1, beta2 ).tVector[ 2 ] ) ** 2 )

# # est_beta2_v1_diff = np.sqrt( ( v_group_list[0].tVector[ 0 ] - est_beta2.tVector[ 0 ] ) ** 2 + ( v_group_list[0].tVector[ 1 ] - est_beta2.tVector[ 1 ] ) ** 2 + ( v_group_list[0].tVector[ 2 ] - est_beta2.tVector[ 2 ] ) ** 2 )
# # est_beta2_v2_diff = np.sqrt( ( v_group_list[1].tVector[ 0 ] - base.ParallelTranslate( est_beta1, est_beta2 ).tVector[ 0 ] ) ** 2 + ( v_group_list[1].tVector[ 1 ] - base.ParallelTranslate( est_beta1, est_beta2 ).tVector[ 1 ] ) ** 2 + ( v_group_list[1].tVector[ 2 ] - base.ParallelTranslate( est_beta1, est_beta2 ).tVector[ 2 ] ) ** 2 )

# # print( "beta2 : GT - Group-wise TV Diff" )
# # print( beta2_v1_diff + beta2_v2_diff )
# # print( "beta2 : Est - Group-wise TV Diff" )
# # print( est_beta2_v1_diff + est_beta2_v2_diff )

print( "Beta0 " ) 
print( beta0.pt )
print( est_beta0.pt )

print( "Beta1" )
print( beta1.tVector )
print( est_beta1.tVector )

print( "Beta2" )
print( beta2.tVector )
print( est_beta2.tVector )

print( "gamma0" )
print( gamma0.tVector )
print( est_gamma0.tVector )

print( "gamma1" )
print( gamma1.tVector )
print( est_gamma1.tVector )

print( "gamma2" )
print( gamma2.tVector )
print( est_gamma2.tVector )

# Estimated Results
est_group_vtk_list = []
# Group 1 - s = 0
est_group_vtk_list_s1 = []

# Group 1 - s = 0
c_step = ( c1 - c0 ) / float( nCAG_Lines )

for c in range( nCAG_Lines ):
	c_pt =  ( c + 0.5 ) * c_step + c0 

	group_geodesic_pt_list = []

	beta2_c = est_beta2.ScalarMultiply( c_pt )

	p0_c = est_beta0.ExponentialMap( beta2_c )

	gamma0_tilde = est_beta0.ParallelTranslateAtoB( est_beta0, p0_c, est_gamma0 )
	gamma2_tilde = est_beta0.ParallelTranslateAtoB( est_beta0, p0_c, est_gamma2 )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( gamma0_tilde.tVector[ 0 ] + gamma2_tilde.tVector[ 0 ] * c_pt ) * time_pt, ( gamma0_tilde.tVector[ 1 ] + gamma2_tilde.tVector[ 1 ] * c_pt ) * time_pt, ( gamma0_tilde.tVector[ 2 ] + gamma2_tilde.tVector[ 2 ] * c_pt ) * time_pt ] )

		p_t = p0_c.ExponentialMap( v_t )
		group_geodesic_pt_list.append( p_t )

	group_geodesic_vtk = vtk.vtkPolyData()
	group_geodesic_pts = vtk.vtkPoints()

	for t in range( len( group_geodesic_pt_list ) ):
		group_geodesic_pts.InsertNextPoint( group_geodesic_pt_list[ t ].pt[ 0 ], group_geodesic_pt_list[ t ].pt[ 1 ], group_geodesic_pt_list[ t ].pt[ 2 ] )

	group_geodesic_line = vtk.vtkCellArray()
	for t in range( len( group_geodesic_pt_list ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		group_geodesic_line.InsertNextCell( line_i )

	group_geodesic_vtk.SetPoints( group_geodesic_pts )
	group_geodesic_vtk.SetLines( group_geodesic_line )

	est_group_vtk_list_s1.append( group_geodesic_vtk )

# Group 2 - s = 1
est_group_vtk_list_s2 = []

c_step = ( c1 - c0 ) / float( nCAG_Lines )

for c in range( nCAG_Lines ):
	c_pt = ( c + 0.5 ) * c_step + c0 

	group_geodesic_pt_list2 = []

	# p_1 = est_beta0.ExponentialMap( est_beta1 )

	# beta_2_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_beta2 )

	est_beta2_c = est_beta2.ScalarMultiply( c_pt )

	beta_c = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		beta_c.tVector[ k ] = ( est_beta1.tVector[ k ] + est_beta2_c.tVector[ k ] )

	p_1_c = est_beta0.ExponentialMap( beta_c )

	# gamma0_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_gamma0 )
	gamma0_at_p_1_c = est_beta0.ParallelTranslateAtoB( est_beta0, p_1_c, est_gamma0 )

	# gamma1_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_gamma1 )
	gamma1_at_p_1_c = est_beta0.ParallelTranslateAtoB( est_beta0, p_1_c, est_gamma1 )

	# gamma2_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_gamma2 )
	gamma2_at_p_1_c = est_beta0.ParallelTranslateAtoB( est_beta0, p_1_c, est_gamma2 )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( gamma0_at_p_1_c.tVector[ 0 ] + gamma1_at_p_1_c.tVector[ 0 ] + gamma2_at_p_1_c.tVector[ 0 ] * c_pt ) * time_pt, ( gamma0_at_p_1_c.tVector[ 1 ] + gamma1_at_p_1_c.tVector[ 1 ] + gamma2_at_p_1_c.tVector[ 1 ] * c_pt ) * time_pt, ( gamma0_at_p_1_c.tVector[ 2 ] + gamma1_at_p_1_c.tVector[ 2 ] + gamma2_at_p_1_c.tVector[ 2 ] * c_pt ) * time_pt ] )

		p_t = p_1_c.ExponentialMap( v_t )
		group_geodesic_pt_list2.append( p_t )

	group_geodesic_vtk2 = vtk.vtkPolyData()
	group_geodesic_pts2 = vtk.vtkPoints()

	for t in range( len( group_geodesic_pt_list2 ) ):
		group_geodesic_pts2.InsertNextPoint( group_geodesic_pt_list2[ t ].pt[ 0 ], group_geodesic_pt_list2[ t ].pt[ 1 ], group_geodesic_pt_list2[ t ].pt[ 2 ] )

	group_geodesic_line2 = vtk.vtkCellArray()
	for t in range( len( group_geodesic_pt_list2 ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		group_geodesic_line2.InsertNextCell( line_i )

	group_geodesic_vtk2.SetPoints( group_geodesic_pts2 )
	group_geodesic_vtk2.SetLines( group_geodesic_line2 )

	est_group_vtk_list_s2.append( group_geodesic_vtk2 )

est_group_vtk_list = [ est_group_vtk_list_s1, est_group_vtk_list_s2 ]


## Comparisons 
# Geodesic Regression
p0_geo, v0_geo = sm.LinearizedGeodesicRegression_Sphere( t_list_all, pt_list_all, verbose=False )

georeg_pt_list = []

for t in range( nLineTimePt ):
	time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

	v_t = manifolds.sphere_tVec( nDimManifold )
	v_t.SetTangentVector( [ v0_geo.tVector[ 0 ] * time_pt, v0_geo.tVector[ 1 ] * time_pt, v0_geo.tVector[ 2 ] * time_pt ] )

	p_t = p0_geo.ExponentialMap( v_t )
	georeg_pt_list.append( p_t )

georeg_vtk = vtk.vtkPolyData()
georeg_pts = vtk.vtkPoints()

for t in range( len( georeg_pt_list ) ):
	georeg_pts.InsertNextPoint( georeg_pt_list[ t ].pt[ 0 ], georeg_pt_list[ t ].pt[ 1 ], georeg_pt_list[ t ].pt[ 2 ] )

georeg_line = vtk.vtkCellArray()

for t in range( len( georeg_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, t )
	line_i.GetPointIds().SetId( 1, t + 1 )
	georeg_line.InsertNextCell( line_i )

georeg_vtk.SetPoints( georeg_pts )
georeg_vtk.SetLines( georeg_line )

# MultiLv Geodesic Model without Covariates
est_beta0_null, tangent_intercept_arr_null, tangent_slope_arr_null = sm.MultivariateLinearizedGeodesicRegression_Sphere_BottomUp( t_list, pt_list, cov_int_list_null, cov_slope_list_null, max_iter=10, verbose=True )

est_beta1_null = manifolds.sphere_tVec( nDimManifold )
est_beta1_null.SetTangentVector( tangent_intercept_arr_null[ 0 ].tVector )

est_beta2_null = manifolds.sphere_tVec( nDimManifold )
est_beta2_null.SetTangentVector( tangent_intercept_arr_null[ 1 ].tVector )
 
est_gamma0_null = manifolds.sphere_tVec( nDimManifold )
est_gamma0_null.SetTangentVector( tangent_slope_arr_null[ 2 ].tVector )

est_gamma1_null = manifolds.sphere_tVec( nDimManifold )
est_gamma1_null.SetTangentVector( tangent_slope_arr_null[ 0 ].tVector )

est_gamma2_null = manifolds.sphere_tVec( nDimManifold )
est_gamma2_null.SetTangentVector( tangent_slope_arr_null[ 1 ].tVector )

# Estimated Results
est_group_vtk_list_null = []
# Group 1 - s = 0
est_group_vtk_list_s1_null = []

# Group 1 - s = 0
c_step = ( c1 - c0 ) / float( nCAG_Lines )

for c in range( nCAG_Lines ):
	c_pt =  ( c + 0.5 ) * c_step + c0 

	group_geodesic_pt_list = []

	beta2_c = est_beta2_null.ScalarMultiply( c_pt )

	p0_c = est_beta0_null.ExponentialMap( beta2_c )

	gamma0_tilde = est_beta0.ParallelTranslateAtoB( est_beta0_null, p0_c, est_gamma0_null )
	gamma2_tilde = est_beta0.ParallelTranslateAtoB( est_beta0, p0_c, est_gamma2_null )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( gamma0_tilde.tVector[ 0 ] + gamma2_tilde.tVector[ 0 ] * c_pt ) * time_pt, ( gamma0_tilde.tVector[ 1 ] + gamma2_tilde.tVector[ 1 ] * c_pt ) * time_pt, ( gamma0_tilde.tVector[ 2 ] + gamma2_tilde.tVector[ 2 ] * c_pt ) * time_pt ] )

		p_t = p0_c.ExponentialMap( v_t )
		group_geodesic_pt_list.append( p_t )

	group_geodesic_vtk = vtk.vtkPolyData()
	group_geodesic_pts = vtk.vtkPoints()

	for t in range( len( group_geodesic_pt_list ) ):
		group_geodesic_pts.InsertNextPoint( group_geodesic_pt_list[ t ].pt[ 0 ], group_geodesic_pt_list[ t ].pt[ 1 ], group_geodesic_pt_list[ t ].pt[ 2 ] )

	group_geodesic_line = vtk.vtkCellArray()
	for t in range( len( group_geodesic_pt_list ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		group_geodesic_line.InsertNextCell( line_i )

	group_geodesic_vtk.SetPoints( group_geodesic_pts )
	group_geodesic_vtk.SetLines( group_geodesic_line )

	est_group_vtk_list_s1_null.append( group_geodesic_vtk )

# Group 2 - s = 1
est_group_vtk_list_s2_null = []

for c in range( nCAG_Lines ):
	c_pt = ( c + 0.5 ) * c_step + c0 

	group_geodesic_pt_list2 = []

	est_beta2_c_null = est_beta2_null.ScalarMultiply( c_pt )

	beta_c = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		beta_c.tVector[ k ] = ( est_beta1_null.tVector[ k ] + est_beta2_c_null.tVector[ k ] )

	p_1_c = est_beta0_null.ExponentialMap( beta_c )

	# gamma0_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_gamma0 )
	gamma0_at_p_1_c = est_beta0_null.ParallelTranslateAtoB( est_beta0_null, p_1_c, est_gamma0_null )

	# gamma1_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_gamma1 )
	gamma1_at_p_1_c = est_beta0_null.ParallelTranslateAtoB( est_beta0_null, p_1_c, est_gamma1_null )

	# gamma2_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_gamma2 )
	gamma2_at_p_1_c = est_beta0_null.ParallelTranslateAtoB( est_beta0_null, p_1_c, est_gamma2_null )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( gamma0_at_p_1_c.tVector[ 0 ] + gamma1_at_p_1_c.tVector[ 0 ] + gamma2_at_p_1_c.tVector[ 0 ] * c_pt ) * time_pt, ( gamma0_at_p_1_c.tVector[ 1 ] + gamma1_at_p_1_c.tVector[ 1 ] + gamma2_at_p_1_c.tVector[ 1 ] * c_pt ) * time_pt, ( gamma0_at_p_1_c.tVector[ 2 ] + gamma1_at_p_1_c.tVector[ 2 ] + gamma2_at_p_1_c.tVector[ 2 ] * c_pt ) * time_pt ] )

		p_t = p_1_c.ExponentialMap( v_t )
		group_geodesic_pt_list2.append( p_t )

	group_geodesic_vtk2 = vtk.vtkPolyData()
	group_geodesic_pts2 = vtk.vtkPoints()

	for t in range( len( group_geodesic_pt_list2 ) ):
		group_geodesic_pts2.InsertNextPoint( group_geodesic_pt_list2[ t ].pt[ 0 ], group_geodesic_pt_list2[ t ].pt[ 1 ], group_geodesic_pt_list2[ t ].pt[ 2 ] )

	group_geodesic_line2 = vtk.vtkCellArray()
	for t in range( len( group_geodesic_pt_list2 ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		group_geodesic_line2.InsertNextCell( line_i )

	group_geodesic_vtk2.SetPoints( group_geodesic_pts2 )
	group_geodesic_vtk2.SetLines( group_geodesic_line2 )

	est_group_vtk_list_s2_null.append( group_geodesic_vtk2 )

est_group_vtk_list_null = est_group_vtk_list_s1_null + est_group_vtk_list_s2_null

print( "=====================================================" )
print( "== 					Estimation Done 			   ==" )
print( "=====================================================" )

# MLMG
# R2 Statistics
sqDist_sum = 0
sqVar_sum = 0

p_mean = sm.FrechetMean( pt_list_all ) 

for n in range( len( pt_list_all ) ):
	p_n = pt_list_all[ n ]
	s_n = s_list_all[ n ] 
	c_n = c_list_all[ n ] 

	t_n = t_list_all[ n ] 

	beta_n = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		beta_n.tVector[ k ] = est_beta1.tVector[ k ] * s_n + est_beta2.tVector[ k ] * c_n

	p_1_n = est_beta0.ExponentialMap( beta_n ) 

	gamma0_at_p_1_n = est_beta0_null.ParallelTranslateAtoB( est_beta0, p_1_n, est_gamma0 )
	gamma1_at_p_1_n = est_beta0_null.ParallelTranslateAtoB( est_beta0, p_1_n, est_gamma1 )
	gamma2_at_p_1_n = est_beta0_null.ParallelTranslateAtoB( est_beta0, p_1_n, est_gamma2 )

	gamma_n = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		gamma_n.tVector[ k ] = ( gamma0_at_p_1_n.tVector[ k ] + gamma1_at_p_1_n.tVector[ k ] * s_n + gamma2_at_p_1_n.tVector[ k ] * c_n ) * t_n 

	est_p_n = p_1_n.ExponentialMap( gamma_n )

	tVec_est_p_n_to_p_n = est_p_n.LogMap( p_n ) 

	sqDist_n = tVec_est_p_n_to_p_n.normSquared() 

	sqDist_sum += sqDist_n

	tVec_mean_to_p_n = p_mean.LogMap( p_n ) 

	sqVar_n = tVec_mean_to_p_n.normSquared()

	sqVar_sum += sqVar_n

R2 = 1 - ( sqDist_sum / sqVar_sum )

nData = len( pt_list_all )
nParam = 6 

adjustedR2 = R2 - ( ( 1 - R2 ) * nParam / ( nData - nParam - 1 ) )

# MLSG
# R2 Statistics
sqDist_MLSG_sum = 0
sqVar_MLSG_sum = 0

for n in range( len( pt_list_all ) ):
	p_n = pt_list_all[ n ]
	s_n = s_list_all[ n ] 
	c_n = c_list_all[ n ] 

	t_n = t_list_all[ n ] 

	beta_n = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		beta_n.tVector[ k ] = est_beta1_null.tVector[ k ] * s_n + est_beta2_null.tVector[ k ] * c_n

	p_1_n = est_beta0_null.ExponentialMap( beta_n ) 

	gamma0_at_p_1_n = est_beta0_null.ParallelTranslateAtoB( est_beta0_null, p_1_n, est_gamma0_null )
	gamma1_at_p_1_n = est_beta0_null.ParallelTranslateAtoB( est_beta0_null, p_1_n, est_gamma1_null )
	gamma2_at_p_1_n = est_beta0_null.ParallelTranslateAtoB( est_beta0_null, p_1_n, est_gamma2_null )

	gamma_n = manifolds.sphere_tVec( nDimManifold )

	for k in range( nDimManifold ):
		gamma_n.tVector[ k ] = ( gamma0_at_p_1_n.tVector[ k ] + gamma1_at_p_1_n.tVector[ k ] * s_n + gamma2_at_p_1_n.tVector[ k ] * c_n ) * t_n 

	est_p_n = p_1_n.ExponentialMap( gamma_n )

	tVec_est_p_n_to_p_n = est_p_n.LogMap( p_n ) 

	sqDist_n = tVec_est_p_n_to_p_n.normSquared() 

	sqDist_MLSG_sum += sqDist_n

	tVec_mean_to_p_n = p_mean.LogMap( p_n ) 

	sqVar_n = tVec_mean_to_p_n.normSquared()

	sqVar_MLSG_sum += sqVar_n

R2_MLSG = 1 - ( sqDist_MLSG_sum / sqVar_MLSG_sum )

nParam_MLSG = 2 

adjustedR2_MLSG = R2_MLSG - ( ( 1 - R2_MLSG ) * nParam_MLSG / ( nData - nParam_MLSG - 1 ) )

# SG
# R2 Statistics
sqDist_SG_sum = 0
sqVar_SG_sum = 0

for n in range( len( pt_list_all ) ):
	p_n = pt_list_all[ n ]
	s_n = s_list_all[ n ] 
	c_n = c_list_all[ n ] 

	t_n = t_list_all[ n ] 

	v0_geo_n = v0_geo.ScalarMultiply( t_n )

	est_p_n = p0_geo.ExponentialMap( v0_geo_n )

	tVec_est_p_n_to_p_n = est_p_n.LogMap( p_n ) 

	sqDist_n = tVec_est_p_n_to_p_n.normSquared() 

	sqDist_SG_sum += sqDist_n

	tVec_mean_to_p_n = p_mean.LogMap( p_n ) 

	sqVar_n = tVec_mean_to_p_n.normSquared()

	sqVar_SG_sum += sqVar_n

R2_SG = 1 - ( sqDist_SG_sum / sqVar_SG_sum )

nParam_SG = 2 

adjustedR2_SG = R2_SG - ( ( 1 - R2_SG ) * nParam_SG / ( nData - nParam_SG - 1 ) )

###########################################################
###  	R2 and Random Effects For Intercepts/Slopes 	###
###########################################################
subj_intercept_list = []
subj_slope_list = []
subj_R2_list = []


for i in range( len( pt_list ) ):
	est_base_i, est_slope_i = sm.LinearizedGeodesicRegression( t_list[ i ], pt_list[ i ], max_iter=10, verbose=False )

	subj_intercept_list.append( est_base_i ) 
	# Frechet Mean and R2
	mean_i = sm.FrechetMean( pt_list[ i ], maxIter = 500 )
	
	sqDist_sum_i = 0
	sqVar_sum_i = 0

	for j in range( len( pt_list[ i ] ) ):
		p_ij = pt_list[ i ][ j ]
		t_ij = t_list[ i ][ j ]

		slope_t_ij = est_slope_i.ScalarMultiply( t_ij )
		est_p_ij = est_base_i.ExponentialMap( slope_t_ij )

		tVec_est_p_ij_to_p_ij = est_p_ij.LogMap( p_ij )

		sqDist_ij = tVec_est_p_ij_to_p_ij.normSquared() 

		sqDist_sum_i += sqDist_ij

		mean_to_p_ij = mean_i.LogMap( p_ij ) 		
		sqVar_sum_i += mean_to_p_ij.normSquared()

	R2_i = 1 - ( sqDist_sum_i / sqVar_sum_i )

	subj_R2_list.append( R2_i )

	# Transport est_slope to beta0
	s_i = cov_int_list[ i ][ 0 ]
	c_i = cov_int_list[ i ][ 1 ]

	beta_i = manifolds.sphere_tVec( nDimManifold )

	for j in range( nDimManifold ):
		beta_i.tVector[ j ] = est_beta1.tVector[ j ] * s_i + est_beta2.tVector[ j ] * c_i

	f_i = est_beta0.ExponentialMap( beta_i )

	est_slope_f_i = est_base_i.ParallelTranslateAtoB( est_base_i, f_i, est_slope_i )
	est_slope_beta0 = f_i.ParallelTranslateAtoB( f_i, est_beta0, est_slope_f_i ) 

	subj_slope_list.append( est_slope_beta0.tVector )

print( "Subject-Wise R2 Mean" ) 
print( np.average( subj_R2_list ) )

print( "Subject-Wise R2 STD" ) 
print( np.std( subj_R2_list ) )

# Intercept Model R2
mean_intercept = sm.FrechetMean( subj_intercept_list, maxIter=500 ) 

sqDist_intercept = 0
sqVar_intercept = 0
sqDist_hsg_intercept = 0

for i in range( len( subj_intercept_list ) ):
	est_base_i = subj_intercept_list[ i ]

	s_i = cov_int_list[ i ][ 0 ]
	c_i = cov_int_list[ i ][ 1 ]

	beta_i = manifolds.sphere_tVec( nDimManifold )

	for j in range( nDimManifold ):
		beta_i.tVector[ j ] =  est_beta1.tVector[ j ] * s_i + est_beta2.tVector[ j ] * c_i

	f_i = est_beta0.ExponentialMap( beta_i )

	# HMG
	est_tVec_i = f_i.LogMap( est_base_i )
	sqDist_intercept += est_tVec_i.normSquared()

	mean_tVec_i = mean_intercept.LogMap( est_base_i )
	sqVar_intercept += mean_tVec_i.normSquared()


R2_intercept = 1 - ( sqDist_intercept / sqVar_intercept )

print( "HMG Intercept Model R2" )
print( R2_intercept )

print( "HMG Intercept Random Effects" )
print( np.sqrt( ( sqDist_intercept ) / ( len( subj_intercept_list ) - 1.0 ) ) )

print( "HSG Intercept Random Effects" )
print( np.sqrt( sqVar_intercept / ( len( subj_intercept_list ) - 1.0 ) ) )

# Slope Model R2 
mean_slope_tVector = np.zeros( nDimManifold )

for i in range( len( subj_slope_list ) ):
	mean_slope_tVector += subj_slope_list[ i ] 

mean_slope_tVector = np.divide( mean_slope_tVector, len( subj_slope_list ) )

est_sqDist_slope = 0
sqVar_slope = 0


for i in range( len( subj_slope_list ) ):
	slope_i = subj_slope_list[ i ]

	gamma_i = manifolds.sphere_tVec( nDimManifold )

	s_i = cov_int_list[ i ][ 0 ]
	c_i = cov_int_list[ i ][ 1 ]

	for j in range( nDimManifold ):
		gamma_i.tVector[ j ] = ( est_gamma0.tVector[ j ] + est_gamma1.tVector[ j ] * s_i + est_gamma2.tVector[ j ] * c_i ) 

	gamma_i_mat = gamma_i.tVector

	est_sqDist_slope += ( np.linalg.norm( gamma_i_mat - slope_i ) )**2

	sqVar_slope += ( np.linalg.norm( mean_slope_tVector - slope_i ) )**2 

R2_Slope = 1 - ( est_sqDist_slope / sqVar_slope )

print( "HMG Slope Model R2" )
print( R2_Slope )

print( "HMG Slope Random Effects" )
print( np.sqrt( est_sqDist_slope / ( len( subj_intercept_list ) - 1.0 ) ) )

print( "HSG Slope Random Effects" ) 
print( np.sqrt( sqVar_slope / ( len( subj_intercept_list ) - 1.0 ) ) )

print( "SG" )
print( R2_SG )
print( adjustedR2_SG )

print( "MLSG" )
print( R2_MLSG )
print( adjustedR2_MLSG )

print( "MLMG" )
print( R2 )
print( adjustedR2 )


########################################
#####        Visualization        ######   
########################################

# Visualize a sphere coordinate
sphere = vtk.vtkSphereSource()
sphere.SetThetaResolution( 30 )
sphere.SetPhiResolution( 30 )
sphere.SetRadius( 1.0 )
sphere.SetCenter( 0.0, 0.0, 0.0 )
sphere.SetLatLongTessellation( True )
sphere.Update()

conMapper = vtk.vtkPolyDataMapper()
conMapper.SetInputData( sphere.GetOutput() )
conMapper.ScalarVisibilityOff()
conMapper.Update()

conActor = vtk.vtkActor()
conActor.SetMapper( conMapper )
conActor.GetProperty().SetOpacity( 1 )
conActor.GetProperty().SetColor( 0.9, 0.9, 0.9 )
conActor.GetProperty().SetEdgeColor( 0.4, 0.4, 0.7 )
conActor.GetProperty().EdgeVisibilityOn()
conActor.GetProperty().SetAmbient(0.3)
conActor.GetProperty().SetDiffuse(0.375)
conActor.GetProperty().SetSpecular( 0.2 )
conActor.GetProperty().SetSpecularPower( 10.0 )


# Visualize Data points - Group 1 
group_gt_color = [ [ 1, 0, 0 ], [ 0, 0, 1 ] ] 
group_est_color = [ [ 1, 0, 1 ], [ 0, 1, 1 ] ] 
group_ind_est_color = [ [ 0, 1, 0 ], [ 1, 1, 0 ] ] 

group_gt_point_color = [ [ 1, 0, 0 ], [ 0, 0, 1 ] ] 

# Points - Time
points_t = vtk.vtkPoints()
colors_t = vtk.vtkUnsignedCharArray()
colors_t.SetNumberOfComponents( 3 )
colors_t.SetName( "Colors" )

for i in range( len( pt_list_all ) ):
	t_i = t_list_all[ i ] 

	points_t.InsertNextPoint( pt_list_all[ i ].pt[0], pt_list_all[ i ].pt[1], pt_list_all[ i ].pt[2] )
	color_i_t = [  int( 255.0 * float( t_i - t0 ) / float( t1 - t0 + 10 ) ), int( 255.0 * float( t_i - t0 ) / float( t1 - t0  + 10 ) ), int( 255.0 * float( t_i - t0 ) / float( t1 - t0 + 10 ) ) ] 

	colors_t.InsertNextTuple( color_i_t )

ptsPolyData_t = vtk.vtkPolyData()
ptsPolyData_t.SetPoints( points_t )

vertFilter_t = vtk.vtkVertexGlyphFilter()
vertFilter_t.SetInputData( ptsPolyData_t )
vertFilter_t.Update()

polyData_t = vertFilter_t.GetOutput()
polyData_t.GetPointData().SetScalars( colors_t )

ptsMapper_t = vtk.vtkPolyDataMapper()
ptsMapper_t.SetInputData( polyData_t )

ptsActor_t = vtk.vtkActor()
ptsActor_t.SetMapper( ptsMapper_t )
ptsActor_t.GetProperty().SetPointSize( 8 )
ptsActor_t.GetProperty().SetOpacity( 1.0 )
ptsActor_t.GetProperty().SetRenderPointsAsSpheres( 1 )

# Renderer1
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )

# Visualize Geodesic Regression Results with Time 
# Comparisons
# Geodesic Regression 
geoRegMapper = vtk.vtkPolyDataMapper()
geoRegMapper.SetInputData( georeg_vtk )

geoRegActor = vtk.vtkActor()
geoRegActor.SetMapper( geoRegMapper )
geoRegActor.GetProperty().SetLineWidth( 15 )
geoRegActor.GetProperty().SetColor( [ 1, 0, 1 ] )
geoRegActor.GetProperty().SetOpacity( 0.8 )
geoRegActor.GetProperty().SetRenderLinesAsTubes( 1 )

ren.AddActor( geoRegActor ) 
ren.AddActor( ptsActor_t )

# light = vtk.vtkLight() 
# light.SetFocalPoint( 1.0, 0.0, 0.0 )
# light.SetPosition( 2, 1, 0 )

# ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )
ren.AutomaticLightCreationOn()

renWin.Render()
renIn.Start()

# ren.RemoveActor( geoRegActor )
ren.RemoveActor( ptsActor_t )

# Visualize estimated geodesic without covariates
# Points 
points_c = vtk.vtkPoints()
colors_c = vtk.vtkUnsignedCharArray()
colors_c.SetNumberOfComponents( 3 )
colors_c.SetName( "Colors" )

for i in range( len( pt_list_all ) ):
	points_c.InsertNextPoint( pt_list_all[ i ].pt[0], pt_list_all[ i ].pt[1], pt_list_all[ i ].pt[2] )
	color_i_c = [  255, 255, 0 ] 
	colors_c.InsertNextTuple( color_i_c )

ptsPolyData_c = vtk.vtkPolyData()
ptsPolyData_c.SetPoints( points_c )

vertFilter_c = vtk.vtkVertexGlyphFilter()
vertFilter_c.SetInputData( ptsPolyData_c )
vertFilter_c.Update()

polyData_c = vertFilter_c.GetOutput()
polyData_c.GetPointData().SetScalars( colors_c )

ptsMapper_c = vtk.vtkPolyDataMapper()
ptsMapper_c.SetInputData( polyData_c )

ptsActor_c = vtk.vtkActor()
ptsActor_c.SetMapper( ptsMapper_c )
ptsActor_c.GetProperty().SetPointSize( 8 )
ptsActor_c.GetProperty().SetOpacity( 1.0 )
ptsActor_c.GetProperty().SetRenderPointsAsSpheres( 1 )

estGroupGeodesicMapper_null = vtk.vtkPolyDataMapper()
estGroupGeodesicMapper_null.SetInputData( est_group_vtk_list_null[ 0 ] )

estGroupGeodesicActor_null = vtk.vtkActor()
estGroupGeodesicActor_null.SetMapper( estGroupGeodesicMapper_null )
estGroupGeodesicActor_null.GetProperty().SetLineWidth( 15 )
estGroupGeodesicActor_null.GetProperty().SetColor( [ 0.545, 0.271, 0.075 ] )
estGroupGeodesicActor_null.GetProperty().SetOpacity( 1.0 )
estGroupGeodesicActor_null.GetProperty().SetRenderLinesAsTubes( 1 )

ren.AddActor( estGroupGeodesicActor_null )

# Visualized Individually estimated geodesic
for g in range( len( pt_list ) ): 
	ind_estGroupGeodesicMapper = vtk.vtkPolyDataMapper()
	ind_estGroupGeodesicMapper.SetInputData( ind_est_group_vtk_list[ g ] )

	ind_estGroupGeodesicActor = vtk.vtkActor()
	ind_estGroupGeodesicActor.SetMapper( ind_estGroupGeodesicMapper )
	ind_estGroupGeodesicActor.GetProperty().SetLineWidth( 8 )
	ind_estGroupGeodesicActor.GetProperty().SetColor( [ 1, 1, 1 ] )
	ind_estGroupGeodesicActor.GetProperty().SetOpacity( 0.5 )
	ind_estGroupGeodesicActor.GetProperty().SetRenderLinesAsTubes( 1 )

	ren.AddActor( ind_estGroupGeodesicActor )

ren.AddActor( ptsActor_c )

renWin.Render()
renIn.Start()

ren.RemoveActor( estGroupGeodesicActor_null )
ren.RemoveActor( ptsActor_c ) 
ren.RemoveActor( geoRegActor )

## Visualize MLMG 
points = vtk.vtkPoints()
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents( 3 )
colors.SetName( "Colors" )

for i in range( len( pt_list_all ) ):
	s_i = s_list_all[ i ] 
	c_i = c_list_all[ i ]

	if s_i == 0:
		points.InsertNextPoint( pt_list_all[ i ].pt[0], pt_list_all[ i ].pt[1], pt_list_all[ i ].pt[2] )
		# color_i_s = group_gt_point_color[ int( s_i ) ]
		color_i_s_c = [ 255, int( 255.0 * float( c_i - c0 ) / float( c1 - c0 ) ), 0.0 ] 
		# color_i_s_c = [ 255.0, 0, 255.0 ] 

		colors.InsertNextTuple( color_i_s_c )
	else:
		points.InsertNextPoint( pt_list_all[ i ].pt[0], pt_list_all[ i ].pt[1], pt_list_all[ i ].pt[2] )
		# color_i_s = group_gt_point_color[ int( s_i ) ]
		color_i_s_c = [ 0.0, int( 255.0 * float( c_i - c0 ) / float( c1 - c0 ) ), 255 ] 
		# color_i_s_c = [ 255.0, 0, 255.0 ] 

		colors.InsertNextTuple( color_i_s_c )

ptsPolyData = vtk.vtkPolyData()
ptsPolyData.SetPoints( points )

vertFilter = vtk.vtkVertexGlyphFilter()
vertFilter.SetInputData( ptsPolyData )
vertFilter.Update()

polyData = vertFilter.GetOutput()
polyData.GetPointData().SetScalars( colors )

ptsMapper = vtk.vtkPolyDataMapper()
ptsMapper.SetInputData( polyData )

ptsActor = vtk.vtkActor()
ptsActor.SetMapper( ptsMapper )
ptsActor.GetProperty().SetPointSize( 8 )
# ptsActor.GetProperty().SetColor( group_gt_color[ 0 ] )
ptsActor.GetProperty().SetOpacity( 1.0 )
ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 )

ren.AddActor( ptsActor )

group_geodesic_actor_arr = [] 

# Visualize ground truth group level geodesics
for i in range( len( group_vtk_list ) ):
	# if i == 0:
	# 	continue 
	for j in range( len( group_vtk_list[ i ] ) ):
		c_ij = group_c_list[ i ][ j ]

		if i == 0:
			color_ij = [ 1.0, ( 1.0 * float( c_ij - c0 ) / float( c1 - c0 ) ), 0.0 ] 
		else:
			color_ij = [ 0.0, ( 1.0 * float( c_ij - c0 ) / float( c1 - c0 ) ), 1.0 ]  


		group_geodesic_mapper = vtk.vtkPolyDataMapper()
		group_geodesic_mapper.SetInputData( group_vtk_list[ i ][ j ] )

		group_geodesic_actor = vtk.vtkActor()
		group_geodesic_actor.SetMapper( group_geodesic_mapper )
		group_geodesic_actor.GetProperty().SetColor( color_ij )
		group_geodesic_actor.GetProperty().SetOpacity( 0.8 )
		group_geodesic_actor.GetProperty().SetLineWidth( 12 )
		group_geodesic_actor.GetProperty().SetRenderLinesAsTubes( 1 )

		group_geodesic_actor_arr.append( group_geodesic_actor )

# for i in range( len( group_geodesic_actor_arr ) ):
# 	ren.AddActor( group_geodesic_actor_arr[ i ] )

renWin.Render()
renIn.Start()

for i in range( len( group_geodesic_actor_arr ) ):
	ren.RemoveActor( group_geodesic_actor_arr[ i ] )		

# Visualize estimated geodesic
for i in range( len( est_group_vtk_list ) ) :
	for j in range( len( est_group_vtk_list[ i ] ) ):
		c_ij = group_c_list[ i ][ j ]

		if i == 0:
			color_ij = [ 1.0, ( 1.0 * float( c_ij - c0 ) / float( c1 - c0 ) ), 0.0 ] 
		else:
			color_ij = [ 0.0, ( 1.0 * float( c_ij - c0 ) / float( c1 - c0 ) ), 1.0 ]  

		estGroupGeodesicMapper = vtk.vtkPolyDataMapper()
		estGroupGeodesicMapper.SetInputData( est_group_vtk_list[ i ][ j ] )

		estGroupGeodesicActor = vtk.vtkActor()
		estGroupGeodesicActor.SetMapper( estGroupGeodesicMapper )
		estGroupGeodesicActor.GetProperty().SetLineWidth( 15 )
		estGroupGeodesicActor.GetProperty().SetColor( color_ij )
		estGroupGeodesicActor.GetProperty().SetOpacity( 0.5 )
		estGroupGeodesicActor.GetProperty().SetRenderLinesAsTubes( 1 )

		ren.AddActor( estGroupGeodesicActor )

# Visualize ground truth group level geodesics
for i in range( len( group_vtk_list ) ):
	# if i == 0:
	# 	continue 
	for j in range( len( group_vtk_list[ i ] ) ):
		c_ij = group_c_list[ i ][ j ]

		group_geodesic_mapper = vtk.vtkPolyDataMapper()
		group_geodesic_mapper.SetInputData( group_vtk_list[ i ][ j ] )

		group_geodesic_actor = vtk.vtkActor()
		group_geodesic_actor.SetMapper( group_geodesic_mapper )
		group_geodesic_actor.GetProperty().SetColor( [ 0, 0, 0 ] )
		group_geodesic_actor.GetProperty().SetOpacity( 1 )
		group_geodesic_actor.GetProperty().SetLineWidth( 8 )
		group_geodesic_actor.GetProperty().SetRenderLinesAsTubes( 1 )

		ren.AddActor( group_geodesic_actor )

# light = vtk.vtkLight() 
# light.SetFocalPoint(0,0.6125,1.875)
# light.SetPosition(1,0.875,1.6125)

# ren.AddLight( light )
# ren.SetBackground( 1.0, 1.0, 1.0 )

renWin.Render()
renIn.Start()
