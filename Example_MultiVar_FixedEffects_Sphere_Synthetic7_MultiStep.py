################################################################################################
################################################################################################
# Synthetic Example - Multivariate Fixed Effects Model                                         #
# Manifold - Sphere Manifold 																   #
# Ind. Variables - age, sex, CAG repeat length 												   #	
# Model - y = ( beta_0 + beta_1 s + beta_2 c ) + beta_3 t + beta_4 st + beta_5 ct			   #	
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

nData_1 = 500
nData_2 = 500
nData_g = 100

nData = nData_1 + nData_2

# Data Noise Parameter
pt_sigma = 0.05

# Ground Truth
beta0 = manifolds.sphere( nDimManifold )
beta1 = manifolds.sphere_tVec( nDimManifold )
beta2 = manifolds.sphere_tVec( nDimManifold )
beta3 = manifolds.sphere_tVec( nDimManifold )
beta4 = manifolds.sphere_tVec( nDimManifold )
beta5 = manifolds.sphere_tVec( nDimManifold )

# Intercept Point
# beta0.SetPoint( [ 0.94644084, 0.00146423, -0.32287396 ] )
beta0.SetPoint( [ 1.0, 0.0, 0.0 ] )
# beta0.SetPoint( [ np.sqrt( 0.3 ), np.sqrt( 0.4 ), np.sqrt( 0.3 ) ] )

## Time
t0 = 0.0
t1 = 60.0

## CAG Repeath Length
c0 = 0.0
c1 = 20.0

# A tangent vector for different sex
beta1.SetTangentVector( [ 0.0, 0.2, 0.0 ] )

# A tangent vector for CAG repeat length
beta2.SetTangentVector( [ 0.0, 0.05, 0.0 ] )

# A slope tangent vector for age
beta3.SetTangentVector( [ 0.0, 0.0, 0.01 ] )

# A slope tangent vector for sex and age
beta4.SetTangentVector( [ 0.0, 0.005, -0.002 ] )

# A slope tangent vector for CAG and age
beta5.SetTangentVector( [ 0.0, 0.001, 0.0 ] )

# # Visualize group level grount truth
# Curve Visualization Parameter 
nLineTimePt = 100
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

	beta3_tilde = beta0.ParallelTranslateAtoB( beta0, p0_c, beta3 )
	beta5_tilde = beta0.ParallelTranslateAtoB( beta0, p0_c, beta5 )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( beta3_tilde.tVector[ 0 ] + beta5_tilde.tVector[ 0 ] * c_pt ) * time_pt, ( beta3_tilde.tVector[ 1 ] + beta5_tilde.tVector[ 1 ] * c_pt ) * time_pt, ( beta3_tilde.tVector[ 2 ] + beta5_tilde.tVector[ 2 ] * c_pt ) * time_pt ] )

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

	p_1 = beta0.ExponentialMap( beta1 )

	beta_2_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta2 )

	beta_2_at_p_1_c = beta_2_at_p_1.ScalarMultiply( c_pt )

	p_1_c = p_1.ExponentialMap( beta_2_at_p_1_c )

	beta_3_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta3 )
	beta_3_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_3_at_p_1 )

	beta_4_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta4 )
	beta_4_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_4_at_p_1 )

	beta_5_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta5 )
	beta_5_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_5_at_p_1 )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( beta_3_at_p_1_c.tVector[ 0 ] + beta_4_at_p_1_c.tVector[ 0 ] + beta_5_at_p_1_c.tVector[ 0 ] * c_pt ) * time_pt, ( beta_3_at_p_1_c.tVector[ 1 ] + beta_4_at_p_1_c.tVector[ 1 ] + beta_5_at_p_1_c.tVector[ 1 ] * c_pt ) * time_pt, ( beta_3_at_p_1_c.tVector[ 2 ] + beta_4_at_p_1_c.tVector[ 2 ] + beta_5_at_p_1_c.tVector[ 2 ] * c_pt ) * time_pt ] )

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

# Lists for Visualization
pt_list_all = [] 
t_list_all = []
s_list = []
c_list = []

c_step = ( c1 - c0 ) / float( nCAG_Lines )
c_pt = ( c + 0.5 ) * c_step + c0 

for s_g in range( 2 ):	
	for g in range( nCAG_Lines ):
		c_g = ( g + 0.5 ) * c_step + c0 

		cov_int_g = [ s_g, c_g ]
		cov_slope_g = [ s_g, c_g ]		

		pt_list_g = []
		t_list_g = []

		for i in range( nData_g ):
			# c_i = np.random.random_integers( c0, c1 )
			t_i = np.random.uniform( t0, t1 )
			
			s_list.append( s_g ) 
			c_list.append( c_g )

			cov_int_i = [ s_g, c_g ]
			cov_slope_i = [ s_g, c_g ] 

			t_list_g.append( t_i )

			p_1 = beta0.ExponentialMap( beta1.ScalarMultiply( s_g ) )

			beta_2_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta2 )

			beta_2_at_p_1_c = beta_2_at_p_1.ScalarMultiply( c_g )

			p_1_c = p_1.ExponentialMap( beta_2_at_p_1_c )

			beta_3_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta3 )
			beta_3_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_3_at_p_1 )

			beta_4_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta4 )
			beta_4_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_4_at_p_1 )

			beta_4_at_p_1_c_s = beta_4_at_p_1_c.ScalarMultiply( s_g )

			beta_5_at_p_1 = beta0.ParallelTranslateAtoB( beta0, p_1, beta5 )
			beta_5_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_5_at_p_1 )

			beta_5_at_p_1_c_c = beta_5_at_p_1_c.ScalarMultiply( c_g )

			v_t_i = manifolds.sphere_tVec( nDimManifold )

			for d in range( nDimManifold ):
				v_t_i.tVector[ d ] = ( beta_3_at_p_1_c.tVector[ d ] + beta_4_at_p_1_c_s.tVector[ d ] + beta_5_at_p_1_c_c.tVector[ d ] ) * t_i

			p_i_mean = p_1_c.ExponentialMap( v_t_i )

			p_i_pert = sm.GaussianNoisePerturbation( p_i_mean, pt_sigma )

			pt_list_g.append( p_i_pert )

			t_list_all.append( t_i )
			pt_list_all.append( p_i_pert ) 

		pt_list.append( pt_list_g )
		t_list.append( t_list_g )
		cov_int_list.append( cov_int_g )
		cov_slope_list.append( cov_slope_g )

print( len( pt_list ) )
print( len( t_list ) )
print( len( cov_int_list ) )
print( len( cov_slope_list ) )

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

	p0_g, v_g = sm.LinearizedGeodesicRegression_Sphere( t_list_g, pt_list_g, verbose=True )

	print( "v_g.tVector" )		
	print( v_g.tVector )

	p0_group_list.append( p0_g )
	v_group_list.append( v_g )

# Individual Estimated Results
ind_est_group_vtk_list = []

for g in range( len( pt_list ) ):
	ind_est_group_geodesic_pt_list = []

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

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

print( len( ind_est_group_vtk_list ) )

# Multi-Step Population Level Estimation

print( "cov_int_list" ) 
print( cov_int_list )
print( "cov_slope_list" ) 
print( cov_slope_list ) 

est_beta0, tangent_intercept_arr, tangent_slope_arr = sm.MultivariateLinearizedGeodesicRegression_Sphere_BottomUp( t_list, pt_list, cov_int_list, cov_slope_list, max_iter=10, verbose=True )

print( "tangent_slope_arr[ 0 ].tVector")
print( tangent_slope_arr[ 0 ].tVector )
print( "tangent_slope_arr[ 1 ].tVector")
print( tangent_slope_arr[ 1 ].tVector )
print( "tangent_slope_arr[ 2 ].tVector")
print( tangent_slope_arr[ 2 ].tVector )

print( "tangent_intercept_arr[ 0 ].tVector")
print( tangent_intercept_arr[0].tVector )
print( "tangent_intercept_arr[ 1 ].tVector")
print( tangent_intercept_arr[1].tVector )

print( "beta0.tVector")
print( beta0.pt )

base = est_beta0

est_beta1 = manifolds.sphere_tVec( nDimManifold )
est_beta1.SetTangentVector( tangent_intercept_arr[ 0 ].tVector )

est_beta2 = manifolds.sphere_tVec( nDimManifold )
est_beta2.SetTangentVector( tangent_intercept_arr[ 1 ].tVector )
 
est_beta3 = manifolds.sphere_tVec( nDimManifold )
est_beta3.SetTangentVector( tangent_slope_arr[ 2 ].tVector )

est_beta4 = manifolds.sphere_tVec( nDimManifold )
est_beta4.SetTangentVector( tangent_slope_arr[ 0 ].tVector )

est_beta5 = manifolds.sphere_tVec( nDimManifold )
est_beta5.SetTangentVector( tangent_slope_arr[ 1 ].tVector )

# print( "Beta2 : GT - Estimated Diff" ) 
# print( np.sqrt( ( est_beta2.tVector[ 0 ] - beta2.tVector[ 0 ] ) ** 2 + ( est_beta2.tVector[ 1 ] - beta2.tVector[ 1 ] ) ** 2 + ( est_beta2.tVector[ 2 ] - beta2.tVector[ 2 ] ) ** 2 ) )

# beta2_v1_diff = np.sqrt( ( v_group_list[0].tVector[ 0 ] - beta2.tVector[ 0 ] ) ** 2 + ( v_group_list[0].tVector[ 1 ] - beta2.tVector[ 1 ] ) ** 2 + ( v_group_list[0].tVector[ 2 ] - beta2.tVector[ 2 ] ) ** 2 )
# beta2_v2_diff = np.sqrt( ( v_group_list[1].tVector[ 0 ] - base.ParallelTranslate( est_beta1, beta2 ).tVector[ 0 ] ) ** 2 + ( v_group_list[1].tVector[ 1 ] - base.ParallelTranslate( est_beta1, beta2 ).tVector[ 1 ] ) ** 2 + ( v_group_list[1].tVector[ 2 ] - base.ParallelTranslate( est_beta1, beta2 ).tVector[ 2 ] ) ** 2 )

# est_beta2_v1_diff = np.sqrt( ( v_group_list[0].tVector[ 0 ] - est_beta2.tVector[ 0 ] ) ** 2 + ( v_group_list[0].tVector[ 1 ] - est_beta2.tVector[ 1 ] ) ** 2 + ( v_group_list[0].tVector[ 2 ] - est_beta2.tVector[ 2 ] ) ** 2 )
# est_beta2_v2_diff = np.sqrt( ( v_group_list[1].tVector[ 0 ] - base.ParallelTranslate( est_beta1, est_beta2 ).tVector[ 0 ] ) ** 2 + ( v_group_list[1].tVector[ 1 ] - base.ParallelTranslate( est_beta1, est_beta2 ).tVector[ 1 ] ) ** 2 + ( v_group_list[1].tVector[ 2 ] - base.ParallelTranslate( est_beta1, est_beta2 ).tVector[ 2 ] ) ** 2 )

# print( "beta2 : GT - Group-wise TV Diff" )
# print( beta2_v1_diff + beta2_v2_diff )
# print( "beta2 : Est - Group-wise TV Diff" )
# print( est_beta2_v1_diff + est_beta2_v2_diff )

# print( "Beta0 " ) 
# print( beta0.pt )
# print( est_beta0.pt )

# print( "Beta1" )
# print( beta1.tVector )
# print( est_beta1.tVector )

# print( "Beta2" )
# print( beta2.tVector )
# print( est_beta2.tVector )

# print( "Beta3" )
# print( beta3.tVector )
# print( est_beta3.tVector )

# print( "Beta4" )
# print( beta4.tVector )
# print( est_beta4.tVector )


# Estimated Results
est_group_vtk_list = []
# Group 1 - s = 0
est_group_vtk_list_s1 = []

# Group 1 - s = 0
c_step = ( c1 - c0 ) / float( nCAG_Lines )

for c in range( nCAG_Lines ):
	c_pt =  ( c + 0.5 ) * c_step + c0 

	group_c_list_s1.append( c_pt )

	group_geodesic_pt_list = []

	beta2_c = est_beta2.ScalarMultiply( c_pt )

	p0_c = est_beta0.ExponentialMap( beta2_c )

	beta3_tilde = est_beta0.ParallelTranslateAtoB( est_beta0, p0_c, est_beta3 )
	beta5_tilde = est_beta0.ParallelTranslateAtoB( est_beta0, p0_c, est_beta5 )

	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( beta3_tilde.tVector[ 0 ] + beta5_tilde.tVector[ 0 ] * c_pt ) * time_pt, ( beta3_tilde.tVector[ 1 ] + beta5_tilde.tVector[ 1 ] * c_pt ) * time_pt, ( beta3_tilde.tVector[ 2 ] + beta5_tilde.tVector[ 2 ] * c_pt ) * time_pt ] )

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

	group_c_list_s2.append( c_pt )

	group_geodesic_pt_list2 = []

	p_1 = est_beta0.ExponentialMap( est_beta1 )

	beta_2_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_beta2 )

	beta_2_at_p_1_c = beta_2_at_p_1.ScalarMultiply( c_pt )

	p_1_c = p_1.ExponentialMap( beta_2_at_p_1_c )

	beta_3_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_beta3 )
	beta_3_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_3_at_p_1 )

	beta_4_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_beta4 )
	beta_4_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_4_at_p_1 )

	beta_5_at_p_1 = est_beta0.ParallelTranslateAtoB( est_beta0, p_1, est_beta5 )
	beta_5_at_p_1_c = p_1.ParallelTranslateAtoB( p_1, p_1_c, beta_5_at_p_1 )


	for t in range( nLineTimePt ):
		time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ ( beta_3_at_p_1_c.tVector[ 0 ] + beta_4_at_p_1_c.tVector[ 0 ] + beta_5_at_p_1_c.tVector[ 0 ] * c_pt ) * time_pt, ( beta_3_at_p_1_c.tVector[ 1 ] + beta_4_at_p_1_c.tVector[ 1 ] + beta_5_at_p_1_c.tVector[ 1 ] * c_pt ) * time_pt, ( beta_3_at_p_1_c.tVector[ 2 ] + beta_4_at_p_1_c.tVector[ 2 ] + beta_5_at_p_1_c.tVector[ 2 ] * c_pt ) * time_pt ] )

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

est_group_vtk_list = est_group_vtk_list_s1 + est_group_vtk_list_s2

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
conActor.GetProperty().SetSpecular(0.0)


# Visualize Data points - Group 1 
group_gt_color = [ [ 1, 0, 0 ], [ 0, 0, 1 ] ] 
group_est_color = [ [ 1, 0, 1 ], [ 0, 1, 1 ] ] 
group_ind_est_color = [ [ 0, 1, 0 ], [ 1, 1, 0 ] ] 

group_gt_point_color = [ [ 1, 0, 0 ], [ 0, 0, 1 ] ] 

points = vtk.vtkPoints()

colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents( 3 )
colors.SetName( "Colors" )

for i in range( len( pt_list_all ) ):
	s_i = s_list[ i ] 
	c_i = c_list[ i ]

	if s_i == 0:
		points.InsertNextPoint( pt_list_all[ i ].pt[0], pt_list_all[ i ].pt[1], pt_list_all[ i ].pt[2] )
		# color_i_s = group_gt_point_color[ int( s_i ) ]
		color_i_s_c = [ 255.0, ( 255.0 * float( c_i - c0 ) / float( c1 - c0 ) ), 0.0 ] 
		# color_i_s_c = [ 255.0, 0, 255.0 ] 

		colors.InsertNextTuple( color_i_s_c )
		dd = 0
	else:
		points.InsertNextPoint( pt_list_all[ i ].pt[0], pt_list_all[ i ].pt[1], pt_list_all[ i ].pt[2] )
		# color_i_s = group_gt_point_color[ int( s_i ) ]
		color_i_s_c = [ 0.0, ( 255.0 * float( c_i - c0 ) / float( c1 - c0 ) ), 255.0 ] 
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


# Renderer1
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )

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
		group_geodesic_actor.GetProperty().SetOpacity( 0.7 )
		group_geodesic_actor.GetProperty().SetLineWidth( 15 )
		group_geodesic_actor.GetProperty().SetRenderLinesAsTubes( 1 )

		ren.AddActor( group_geodesic_actor )


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

# Visualize estimated geodesic
for g in range( len( pt_list ) ): 
	estGroupGeodesicMapper = vtk.vtkPolyDataMapper()
	estGroupGeodesicMapper.SetInputData( est_group_vtk_list[ g ] )

	estGroupGeodesicActor = vtk.vtkActor()
	estGroupGeodesicActor.SetMapper( estGroupGeodesicMapper )
	estGroupGeodesicActor.GetProperty().SetLineWidth( 8 )
	estGroupGeodesicActor.GetProperty().SetColor( [ 0, 0, 0 ] )
	estGroupGeodesicActor.GetProperty().SetOpacity( 0.5 )
	estGroupGeodesicActor.GetProperty().SetRenderLinesAsTubes( 1 )

	ren.AddActor( estGroupGeodesicActor )

ren.AddActor( ptsActor )

light = vtk.vtkLight() 
light.SetFocalPoint(0,0.6125,1.875)
light.SetPosition(1,0.875,1.6125)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

renWin.Render()
renIn.Start()
