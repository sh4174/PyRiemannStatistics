# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds 
import numpy as np
import StatsModel as sm

# Visualization
import vtk

# Time
import time

# Color 
from colour import Color 

#####################################################
#				 	Data Generation					#
#####################################################
# Generating sphere manifolds. distributed over time perturbed by Gaussian random
# Time
t0 = 0	
t1 = 2

# Generate a random point on the manifold
# Parameters
nDimManifold = 3
nData = 200
dim = nDimManifold
sigma = 0.15

# Population Bias 
p_bias = manifolds.sphere( nDimManifold )
v_bias = manifolds.sphere_tVec( nDimManifold )

p_bias.SetPoint( [ 0.0, 0.0, 1.0 ] )
v_bias.SetTangentVector( [ 0, np.pi * 0.1, 0 ] )

bias_param_t0 = 0.0
bias_param_t1 = 3.0

nSubj = 10
bias_step = ( bias_param_t1 - bias_param_t0 ) / float( nSubj - 1 )

# Subject-wise Ground Truth
subj_t0 = -0.5
subj_t1 = 0.5

v_subj_slope_base = manifolds.sphere_tVec( nDimManifold )
v_subj_slope_base.SetTangentVector( [ np.pi * 0.1, 0, 0 ] )

nData_Subj = 10
subj_time_step = ( subj_t1 - subj_t0 ) / float( nData_Subj - 1 )

# All Data and time : Cross-Secional Data Set
pt_list_all = []
t_list_all = []

# Collection of Subject-wise data and time : Longitudinal Data Set
pt_list = []
t_list = []
cov_int_list = []
cov_slope_list = []

for i in range( nSubj ):
	bias_t_i = bias_step * float( i ) + bias_param_t0	
	bias_interp = p_bias.ExponentialMap( v_bias.ScalarMultiply( bias_t_i ) ) 
	v_slope_i = p_bias.ParallelTranslateAtoB( p_bias, bias_interp, v_subj_slope_base )

	# Subject-wise Trajectory
	pt_list_i = [] 
	t_list_i = []

	for j in range( nData_Subj ):
		subj_t_ij_for_subj_geodesic = subj_time_step * float( j ) + subj_t0 		

		pt_ij = bias_interp.ExponentialMap( v_slope_i.ScalarMultiply( subj_t_ij_for_subj_geodesic ) )
		t_ij = subj_t_ij_for_subj_geodesic + bias_t_i

		# Project sin
		v_t_sin = manifolds.sphere_tVec( nDimManifold )
		v_t_sin.tVector[ 0 ] = 0
		v_t_sin.tVector[ 1 ] = np.sin( t_ij * np.pi * 4 ) / 20.0 
		v_t_sin.tVector[ 2 ] = 0

		v_t_proj_sin = pt_ij.ProjectTangent( pt_ij, v_t_sin )
		pt_proj_sin_ij = pt_ij.ExponentialMap( v_t_proj_sin )

		pt_list_i.append( pt_proj_sin_ij ) 
		t_list_i.append( t_ij ) 

		pt_list_all.append( pt_proj_sin_ij ) 
		t_list_all.append( t_ij )

	pt_list.append( pt_list_i )
	t_list.append( t_list_i )

	cov_int_list.append( [ 0 ] )
	cov_slope_list.append( [ 0 ] ) 


############################################
#	 Hierarchical Multi-Geodesic Model 	   #
############################################
# 1) Geodesic Regression
#####################################################
# 	Linearized Geodesic Regression for all Data		#
#####################################################
step_size = 0.01
max_iter = 500
step_tol = 1e-8

start_egreg_all_data = time.time() 
base_all_lReg, tangent_all_lReg = sm.LinearizedGeodesicRegression( t_list_all, pt_list_all, max_iter, step_size, step_tol, False, False )
end_egreg_all_data = time.time()

# Display
nDispPts = 100
vis_t0 = -0.5
vis_t1 = 3.5

geoReg_pts = vtk.vtkPoints()
geoReg_lines = vtk.vtkCellArray()
geoReg_data = vtk.vtkPolyData()

for i in range( nDispPts ):
	t_i = vis_t0 + ( ( vis_t1 - vis_t0 ) / float( nDispPts - 1 ) ) * float( i )

	pt_i = base_all_lReg.ExponentialMap( tangent_all_lReg.ScalarMultiply( t_i ) )
	geoReg_pts.InsertNextPoint( pt_i.pt[ 0 ], pt_i.pt[ 1 ], pt_i.pt[ 2 ] )

for i in range( nDispPts - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i ) 
	line_i.GetPointIds().SetId( 1, i + 1 ) 
	
	geoReg_lines.InsertNextCell( line_i )

geoReg_data.SetPoints( geoReg_pts )
geoReg_data.SetLines( geoReg_lines )

geoReg_data.Modified()

geoRegMapper = vtk.vtkPolyDataMapper()
geoRegMapper.SetInputData( geoReg_data )
geoRegMapper.Update()

geoRegActor = vtk.vtkActor()
geoRegActor.SetMapper( geoRegMapper )
geoRegActor.GetProperty().SetLineWidth( 15 )
geoRegActor.GetProperty().SetColor( [ 0, 0, 1 ] )
geoRegActor.GetProperty().SetOpacity( 0.8 )
geoRegActor.GetProperty().SetRenderLinesAsTubes( 1 )


# 2) Hierarchical Multi-Geodesic Regression
#############################################################
# 	Hierarchical Multi-Geodesic Regression for all Data		#
#############################################################
step_size = 0.01
max_iter = 500
step_tol = 1e-8

est_beta0, tangent_intercept_arr, tangent_slope_arr = sm.MultivariateLinearizedGeodesicRegression_Sphere_BottomUp( t_list, pt_list, cov_int_list, cov_slope_list, max_iter=max_iter, verbose=False )

est_gamma0_null = manifolds.sphere_tVec( nDimManifold )
est_gamma0_null.SetTangentVector( tangent_slope_arr[ 1 ].tVector )

est_gamma1_null = manifolds.sphere_tVec( nDimManifold )
est_gamma1_null.SetTangentVector( tangent_slope_arr[ 0 ].tVector )

print( "Gamma0")
print( est_gamma0_null.tVector )
print( "Gamma1")
print( est_gamma1_null.tVector )

# Display
HGeoReg_pts = vtk.vtkPoints()
HGeoReg_lines = vtk.vtkCellArray()
HGeoReg_data = vtk.vtkPolyData()

for i in range( nDispPts ):
	t_i = vis_t0 + ( ( vis_t1 - vis_t0 ) / float( nDispPts - 1 ) ) * float( i )

	pt_i = est_beta0.ExponentialMap( est_gamma0_null.ScalarMultiply( t_i ) )
	HGeoReg_pts.InsertNextPoint( pt_i.pt[ 0 ], pt_i.pt[ 1 ], pt_i.pt[ 2 ] )

for i in range( nDispPts - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i ) 
	line_i.GetPointIds().SetId( 1, i + 1 ) 
	
	HGeoReg_lines.InsertNextCell( line_i )

HGeoReg_data.SetPoints( HGeoReg_pts )
HGeoReg_data.SetLines( HGeoReg_lines )

HGeoReg_data.Modified()

HGeoRegMapper = vtk.vtkPolyDataMapper()
HGeoRegMapper.SetInputData( HGeoReg_data )
HGeoRegMapper.Update()

HGeoRegActor = vtk.vtkActor()
HGeoRegActor.SetMapper( HGeoRegMapper )
HGeoRegActor.GetProperty().SetLineWidth( 15 )
HGeoRegActor.GetProperty().SetColor( [ 0, 1, 0 ] )
HGeoRegActor.GetProperty().SetOpacity( 0.8 )
HGeoRegActor.GetProperty().SetRenderLinesAsTubes( 1 )

# 3) Wrapped Gaussian Process
#############################################################
#		 	Wrapped Gaussian Process for all Data			#
#############################################################
nPt_line = 100
l = 0.1
sigma_f = 0.15

sigma = 0.01

def kernel_k( x, y, sigma_f=1, l=0.5 ):
	return sigma_f * np.exp( -( ( x - y )**2 ) / ( 2 * (l**2) ) )

# # ##################################
# # # 	    Kernel Regression	     #
# # ##################################
# # # Draw weighted means 
# # min_t = np.min( t_list )
# # max_t = np.max( t_list )
# # t_step = float( max_t - min_t ) / float( nPt_line - 1 )

# # # Weighted Average over time
# # wAvg_list = [] 
# # t_list_out = []

# # for i in range( nPt_line ):
# # 	t_i = min_t + ( float( i ) * t_step )
# # 	t_list_out.append( t_i ) 

# # 	w_list_i = [] 

# # 	for j in range( len( data_all_list ) ):
# # 		w_list_i.append( kernel_k( t_i, time_all_list[ j ], sigma_f=sigma_f, l=l ) )

# # 	print( np.sum( w_list_i ) )
# # 	w_list_i = np.divide( w_list_i, np.sum( w_list_i ) )
# # 	print( np.sum( w_list_i ) )

# # 	w_avg_i = sm.WeightedFrechetMean( data_all_list, w_list_i )
# # 	wAvg_list.append( w_avg_i )

# # # Visualization : Kernel Regression - Gaussian Kernel
# # wAvg_Points = vtk.vtkPoints()

# # for i in range( len( wAvg_list ) ):
# # 	wAvg_Points.InsertNextPoint( wAvg_list[ i ].pt[0], wAvg_list[ i ].pt[1], wAvg_list[ i ].pt[2] )

# # wAvg_ptsPolyData = vtk.vtkPolyData()
# # wAvg_ptsPolyData.SetPoints( wAvg_Points )

# # wAvg_vertFilter = vtk.vtkVertexGlyphFilter()
# # wAvg_vertFilter.SetInputData( wAvg_ptsPolyData )
# # wAvg_vertFilter.Update()

# # wAvg_ptsMapper = vtk.vtkPolyDataMapper()
# # wAvg_ptsMapper.SetInputData( wAvg_vertFilter.GetOutput() )

# # # Red - Gaussian Process
# # wAvg_ptsActor = vtk.vtkActor()
# # wAvg_ptsActor.SetMapper( wAvg_ptsMapper )
# # wAvg_ptsActor.GetProperty().SetPointSize( 10 )
# # wAvg_ptsActor.GetProperty().SetColor( 0, 0, 1 )
# # wAvg_ptsActor.GetProperty().SetOpacity( 1.0 )
# # wAvg_ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 


# Transform Data to a set of tangent vectors 
nData = len( pt_list_all )

# BPF - Geodesic
tVec_list = [] 
tVec_mat_list = []
p_tVec_train_mat = np.zeros( [ nData, 3 ] )

# # BPF - Frechet Mean 
# tVec_mu_list = []
# tVec_mu_mat_list = []
# p_tVec_mu_train_mat = np.zeros( [ nData, 3 ] )

for i in range( len( pt_list_all ) ):	
	# Base point on a geodesic 
	t_i = t_list_all[ i ]

	bp_i = base_all_lReg.ExponentialMap( tangent_all_lReg.ScalarMultiply( t_i ) )

	tVec_i = bp_i.LogMap( pt_list_all[ i ] )
	tVec_list.append( tVec_i )
	tVec_mat_list.append( tVec_i.tVector )
	p_tVec_train_mat[ i, 0 ] = tVec_i.tVector[ 0 ]
	p_tVec_train_mat[ i, 1 ] = tVec_i.tVector[ 1 ]
	p_tVec_train_mat[ i, 2 ] = tVec_i.tVector[ 2 ]

# 	# Base point on a Frechet mean 
# 	tVec_mu_i = mean_mu.LogMap( pt_list_all[ i ] )
# 	tVec_mu_list.append( tVec_mu_i ) 
# 	tVec_mu_mat_list.append( tVec_mu_i.tVector )

# 	p_tVec_mu_train_mat[ i, 0 ] = tVec_mu_i.tVector[ 0 ]
# 	p_tVec_mu_train_mat[ i, 1 ] = tVec_mu_i.tVector[ 1 ]
# 	p_tVec_mu_train_mat[ i, 2 ] = tVec_mu_i.tVector[ 2 ]

# K
K_tr_tr = np.zeros( [ nData, nData ] )

for i in range( nData ):
	for j in range( nData ):
		K_tr_tr[ i, j ] = kernel_k( t_list_all[ i ], t_list_all[ j ], sigma_f=sigma_f, l=l )

# Add Noisy Data Variation
for i in range( nData ):
	K_tr_tr[ i, i ] = K_tr_tr[ i, i ] + sigma


# Test Data 
nTestData = nPt_line
t_test_list = np.linspace( np.min( t_list_all ), np.max( t_list_all ), num=nTestData )

# K_**
K_test_test = np.zeros( [ nTestData, nTestData ] )

# K_*
K_test_tr = np.zeros( [ nTestData, nData ] )

for i in range( nTestData ):
	for j in range( nTestData ):
		K_test_test[ i, j ] = kernel_k( t_test_list[ i ], t_test_list[ j ], sigma_f=sigma_f, l=l )


for i in range( nTestData ):
	for j in range( nData ):
		K_test_tr[ i, j ] = kernel_k( t_test_list[ i ], t_list_all[ j ], sigma_f=sigma_f, l=l )

# K_* x K^{-1} 
K_star_K_inv = np.matmul( K_test_tr, np.linalg.inv( K_tr_tr ) )

# mu_test
p_test_tVec_mat = np.matmul( K_star_K_inv, p_tVec_train_mat )

# # mu_test_frechet_mean 
# p_test_tVec_mu_mat = np.matmul( K_star_K_inv, p_tVec_mu_train_mat )

# Sigma_test
Sigma_test = np.subtract( K_test_test, np.matmul( K_test_tr, np.matmul( np.linalg.inv( K_tr_tr ), K_test_tr.T ) ) )
pt_test_list = []
sigma_test_list = []
sigma_test_list_ind = []

pt_test_mu_list = []

for i in range( nTestData ):
	# From geodesic BPF
	tVec_arr_i = p_test_tVec_mat[ i, : ]
	v_test_i = manifolds.sphere_tVec( nDimManifold )
	v_test_i.SetTangentVector( [ tVec_arr_i[ 0 ], tVec_arr_i[ 1 ], tVec_arr_i[ 2 ] ] )

	t_test_i = t_test_list[ i ] 
	bp_i = base_all_lReg.ExponentialMap( tangent_all_lReg.ScalarMultiply( t_test_i ) )

	p_test_i = bp_i.ExponentialMap( v_test_i )
	pt_test_list.append( p_test_i )
	sigma_test_list.append( Sigma_test[ i, i ] )

# 	# From Frechet Mean BPF
# 	tVec_arr_mu_i = p_test_tVec_mu_mat[ i, : ]
# 	v_test_mu_i = manifolds.sphere_tVec( nDimManifold )
# 	v_test_mu_i.SetTangentVector( [ tVec_arr_mu_i[ 0 ], tVec_arr_mu_i[ 1 ], tVec_arr_mu_i[ 2 ] ] ) 

# 	p_test_mu_i = mean_mu.ExponentialMap( v_test_mu_i )
# 	pt_test_mu_list.append( p_test_mu_i ) 

## Test Points Visualization - BPF - Geodesic
test_points = vtk.vtkPoints()

for i in range( len( pt_test_list ) ):
	test_points.InsertNextPoint( pt_test_list[ i ].pt[0], pt_test_list[ i ].pt[1], pt_test_list[ i ].pt[2] )

test_ptsPolyData = vtk.vtkPolyData()
test_ptsPolyData.SetPoints( test_points )

test_vertFilter = vtk.vtkVertexGlyphFilter()
test_vertFilter.SetInputData( test_ptsPolyData )
test_vertFilter.Update()

test_ptsMapper = vtk.vtkPolyDataMapper()
test_ptsMapper.SetInputData( test_vertFilter.GetOutput() )

# Magenta - Gaussian Process
test_ptsActor = vtk.vtkActor()
test_ptsActor.SetMapper( test_ptsMapper )
test_ptsActor.GetProperty().SetPointSize( 10 )
test_ptsActor.GetProperty().SetColor( 1, 0, 1 )
test_ptsActor.GetProperty().SetOpacity( 1.0 )
test_ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

#################################################
#	 Uncertainty and BPF - Frechet Test Codes 	#
#################################################
# ## Test Points Visualization - BPF - Frechet Mean
# test_mu_points = vtk.vtkPoints()

# for i in range( len( pt_test_mu_list ) ):
# 	test_mu_points.InsertNextPoint( pt_test_mu_list[ i ].pt[0], pt_test_mu_list[ i ].pt[1], pt_test_mu_list[ i ].pt[2] )

# test_mu_ptsPolyData = vtk.vtkPolyData()
# test_mu_ptsPolyData.SetPoints( test_mu_points )

# test_mu_vertFilter = vtk.vtkVertexGlyphFilter()
# test_mu_vertFilter.SetInputData( test_mu_ptsPolyData )
# test_mu_vertFilter.Update()

# test_mu_ptsMapper = vtk.vtkPolyDataMapper()
# test_mu_ptsMapper.SetInputData( test_mu_vertFilter.GetOutput() )

# # Red - Gaussian Process
# test_mu_ptsActor = vtk.vtkActor()
# test_mu_ptsActor.SetMapper( test_mu_ptsMapper )
# test_mu_ptsActor.GetProperty().SetPointSize( 10 )
# test_mu_ptsActor.GetProperty().SetColor( 1, 0, 0 )
# test_mu_ptsActor.GetProperty().SetOpacity( 1.0 )
# test_mu_ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# #############################################
# # There is something wrong with probability
# # Maybe change it to the confidence interval with 2 sigma
# #############################################

# # Calculate Uncertainty of the GP estimation
# # Sample points on sphere 
# sphere_uncertainty = vtk.vtkSphereSource()
# sphere_uncertainty.SetThetaResolution( 60 )
# sphere_uncertainty.SetPhiResolution( 30 )
# sphere_uncertainty.SetRadius( 1.0 )
# sphere_uncertainty.SetCenter( 0.0, 0.0, 0.0 )
# sphere_uncertainty.SetLatLongTessellation( False )
# sphere_uncertainty.Update()

# sphere_uncertainty_polyData = sphere_uncertainty.GetOutput()

# # The number of points
# nSpherePt = sphere_uncertainty_polyData.GetNumberOfPoints()
# prob_sphere_GP = np.zeros( [ nSpherePt, len( pt_test_list ) ] )

# max_prob_sphere_arr = np.zeros( nSpherePt )

# for j in range( len( pt_test_list ) ):
# 	test_j = pt_test_list[ j ]

# 	prob_surface_j = np.zeros( nSpherePt )

# 	for i in range( nSpherePt ):
# 		sphere_pt_i_arr = sphere_uncertainty_polyData.GetPoint( i )
# 		sphere_pt_i = manifolds.sphere( 3 )
# 		sphere_pt_i.SetPoint( sphere_pt_i_arr )

# 		log_test_j_to_sphere_pt_i = test_j.LogMap( sphere_pt_i )

# 		distSq_sphere_pt_i_to_test_j = log_test_j_to_sphere_pt_i.normSquared()
# 		dist_sphere_pt_i_to_test_j = log_test_j_to_sphere_pt_i.norm()

# 		if dist_sphere_pt_i_to_test_j < 2 * np.sqrt( sigma_test_list[ j ] ):
# 			max_prob_sphere_arr[ i ] = 1 

# colorPointArr = vtk.vtkUnsignedCharArray() 
# colorPointArr.SetNumberOfComponents( 3 )
# colorPointArr.SetName( "Colors" ) 

# for i in range( len( max_prob_sphere_arr) ):
# 	if max_prob_sphere_arr[ i ] > 0.5:
# 		colorPointArr.InsertNextTuple( [ 0, 0, 255 ] )
# 	else:
# 		colorPointArr.InsertNextTuple( [ 255, 255, 255 ] )

# sphere_uncertainty_polyData.GetPointData().SetScalars( colorPointArr )
# sphere_uncertainty_polyData.Modified()


# 3) Semi-Parametric Wrapped Gaussian Process
#########################################################################
#	  Semi-Parametric Wrapped Gaussian Process for Longitudinal Data	#
#########################################################################
est_subj_geo_interp_list = []
est_subj_geo_slope_list = []

for i in range( nSubj ):
	base_lReg_subj_i, tangent_lReg_subj_i = sm.LinearizedGeodesicRegression( t_list[ i ], pt_list[ i ], max_iter, step_size, step_tol, False, False )

	est_subj_geo_interp_list.append( base_lReg_subj_i )
	est_subj_geo_slope_list.append( tangent_lReg_subj_i )


# BPF - Subject-specific Geodesics
tVec_list_Lon = [] 
tVec_mat_list_Lon = []
p_tVec_train_mat_Lon = np.zeros( [ nData, 3 ] )

# BPF - Subject-specific Geodesics : Parallel Transported to Population Level Geodesic
tVec_list_Lon_PT = [] 
tVec_mat_list_Lon_PT = []
p_tVec_train_mat_Lon_PT = np.zeros( [ nData, 3 ] )

cnt_ij = 0

for i in range( len( pt_list ) ):
	for j in range( len( pt_list[ i ] ) ):
		# Base point on a Hierarchical Geodesic Model (No PT)
		t_ij = t_list[ i ][ j ]

		bp_ij = est_subj_geo_interp_list[ i ].ExponentialMap( est_subj_geo_slope_list[ i ].ScalarMultiply( t_ij ) )
		tVec_ij = bp_ij.LogMap( pt_list[ i ][ j ] )

		tVec_list_Lon.append( tVec_ij )
		tVec_mat_list_Lon.append( tVec_ij.tVector )
		p_tVec_train_mat_Lon[ cnt_ij, 0 ] = tVec_ij.tVector[ 0 ]
		p_tVec_train_mat_Lon[ cnt_ij, 1 ] = tVec_ij.tVector[ 1 ]
		p_tVec_train_mat_Lon[ cnt_ij, 2 ] = tVec_ij.tVector[ 2 ]

		# Base point on a Hierarchical Geodesic Model (Yes PT)
		pop_bp_ij = est_beta0.ExponentialMap( est_gamma0_null.ScalarMultiply( t_ij ) )

		tVec_ij_Population = bp_ij.ParallelTranslateAtoB( bp_ij, pop_bp_ij, tVec_ij )
				
		tVec_list_Lon_PT.append( tVec_ij_Population )		
		tVec_mat_list_Lon_PT.append( tVec_ij_Population.tVector )		
		p_tVec_train_mat_Lon_PT[ cnt_ij, 0 ] = tVec_ij_Population.tVector[ 0 ]
		p_tVec_train_mat_Lon_PT[ cnt_ij, 1 ] = tVec_ij_Population.tVector[ 1 ]
		p_tVec_train_mat_Lon_PT[ cnt_ij, 2 ] = tVec_ij_Population.tVector[ 2 ]

		cnt_ij += 1

# mu_test
p_test_tVec_mat_Lon = np.matmul( K_star_K_inv, p_tVec_train_mat_Lon )		

pt_test_list_Lon = []

for i in range( nTestData ):
	# From Hiearchicla Geodesic BPF
	tVec_arr_i = p_test_tVec_mat_Lon[ i, : ]
	v_test_i = manifolds.sphere_tVec( nDimManifold )
	v_test_i.SetTangentVector( [ tVec_arr_i[ 0 ], tVec_arr_i[ 1 ], tVec_arr_i[ 2 ] ] )

	t_test_i = t_test_list[ i ] 
	bp_i = est_beta0.ExponentialMap( est_gamma0_null.ScalarMultiply( t_test_i ) )

	p_test_i = bp_i.ExponentialMap( v_test_i )
	pt_test_list_Lon.append( p_test_i )
	
## Test Points Visualization - BPF - Geodesic
test_points_Lon = vtk.vtkPoints()

for i in range( len( pt_test_list_Lon ) ):
	test_points_Lon.InsertNextPoint( pt_test_list_Lon[ i ].pt[0], pt_test_list_Lon[ i ].pt[1], pt_test_list_Lon[ i ].pt[2] )

test_ptsPolyData_Lon = vtk.vtkPolyData()
test_ptsPolyData_Lon.SetPoints( test_points_Lon )

test_vertFilter_Lon = vtk.vtkVertexGlyphFilter()
test_vertFilter_Lon.SetInputData( test_ptsPolyData_Lon )
test_vertFilter_Lon.Update()

test_ptsMapper_Lon = vtk.vtkPolyDataMapper()
test_ptsMapper_Lon.SetInputData( test_vertFilter_Lon.GetOutput() )

# Magenta - Gaussian Process
test_ptsActor_Lon = vtk.vtkActor()
test_ptsActor_Lon.SetMapper( test_ptsMapper_Lon )
test_ptsActor_Lon.GetProperty().SetPointSize( 15 )
test_ptsActor_Lon.GetProperty().SetColor( 1, 0, 0 )
test_ptsActor_Lon.GetProperty().SetOpacity( 1.0 )
test_ptsActor_Lon.GetProperty().SetRenderPointsAsSpheres( 1 ) 

#####################################################
#			 Parallel Transported W-GP 				#
#####################################################
# mu_test
p_test_tVec_mat_Lon_PT = np.matmul( K_star_K_inv, p_tVec_train_mat_Lon_PT )		

pt_test_list_Lon_PT = []

for i in range( nTestData ):
	# From Hiearchicla Geodesic BPF
	tVec_arr_i = p_test_tVec_mat_Lon_PT[ i, : ]
	v_test_i = manifolds.sphere_tVec( nDimManifold )
	v_test_i.SetTangentVector( [ tVec_arr_i[ 0 ], tVec_arr_i[ 1 ], tVec_arr_i[ 2 ] ] )

	t_test_i = t_test_list[ i ] 
	bp_i = est_beta0.ExponentialMap( est_gamma0_null.ScalarMultiply( t_test_i ) )

	p_test_i = bp_i.ExponentialMap( v_test_i )
	pt_test_list_Lon_PT.append( p_test_i )
	
## Test Points Visualization - BPF - Geodesic
test_points_Lon_PT = vtk.vtkPoints()

for i in range( len( pt_test_list_Lon_PT ) ):
	test_points_Lon_PT.InsertNextPoint( pt_test_list_Lon_PT[ i ].pt[0], pt_test_list_Lon_PT[ i ].pt[1], pt_test_list_Lon_PT[ i ].pt[2] )

test_ptsPolyData_Lon_PT = vtk.vtkPolyData()
test_ptsPolyData_Lon_PT.SetPoints( test_points_Lon_PT )

test_vertFilter_Lon_PT = vtk.vtkVertexGlyphFilter()
test_vertFilter_Lon_PT.SetInputData( test_ptsPolyData_Lon_PT )
test_vertFilter_Lon_PT.Update()

test_ptsMapper_Lon_PT = vtk.vtkPolyDataMapper()
test_ptsMapper_Lon_PT.SetInputData( test_vertFilter_Lon_PT.GetOutput() )

# Magenta - Gaussian Process
test_ptsActor_Lon_PT = vtk.vtkActor()
test_ptsActor_Lon_PT.SetMapper( test_ptsMapper_Lon_PT )
test_ptsActor_Lon_PT.GetProperty().SetPointSize( 15 )
test_ptsActor_Lon_PT.GetProperty().SetColor( 0, 1, 1 )
test_ptsActor_Lon_PT.GetProperty().SetOpacity( 1.0 )
test_ptsActor_Lon_PT.GetProperty().SetRenderPointsAsSpheres( 1 ) 


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

conMapper_sph = vtk.vtkPolyDataMapper()
conMapper_sph.SetInputData( sphere.GetOutput() )
conMapper_sph.ScalarVisibilityOff()
conMapper_sph.Update()

conActor_sph = vtk.vtkActor()
conActor_sph.SetMapper( conMapper_sph )
conActor_sph.GetProperty().SetOpacity( 1.0 )
conActor_sph.GetProperty().SetColor( 0.9, 0.9, 0.9 )
conActor_sph.GetProperty().SetEdgeColor( 0.4, 0.4, 0.7 )
conActor_sph.GetProperty().EdgeVisibilityOn()
conActor_sph.GetProperty().SetAmbient(0.3)
conActor_sph.GetProperty().SetDiffuse(0.375)
conActor_sph.GetProperty().SetSpecular(0.0)

# Folded Visualization EVERYTHING

# # # Confidence Interval Sphere
# # conMapper = vtk.vtkPolyDataMapper()
# # conMapper.SetInputData( sphere_uncertainty_polyData )
# # # conMapper.ScalarVisibilityOn()
# # # conMapper.SetScalarRange( 0, 1 )
# # conMapper.Update()

# # conActor = vtk.vtkActor()
# # conActor.SetMapper( conMapper )
# # conActor.GetProperty().SetOpacity( 0.5 )
# # # conActor.GetProperty().SetColor( 0.9, 0.9, 0.9 )
# # # conActor.GetProperty().SetEdgeColor( 0.4, 0.4, 0.7 )
# # conActor.GetProperty().EdgeVisibilityOff()
# # conActor.GetProperty().SetAmbient(0.3)
# # conActor.GetProperty().SetDiffuse(0.375)
# # conActor.GetProperty().SetSpecular(0.0)

# # Visualize projected sin function - Blue : pt_list
# points_proj_sin = vtk.vtkPoints()

# for i in range( len( pt_list_proj_sin ) ):
# 	points_proj_sin.InsertNextPoint( pt_list_proj_sin[ i ].pt[0], pt_list_proj_sin[ i ].pt[1], pt_list_proj_sin[ i ].pt[2] )

# ptsPolyData_proj_sin = vtk.vtkPolyData()
# ptsPolyData_proj_sin.SetPoints( points_proj_sin )

# vertFilter_proj_sin = vtk.vtkVertexGlyphFilter()
# vertFilter_proj_sin.SetInputData( ptsPolyData_proj_sin )
# vertFilter_proj_sin.Update()

# ptsMapper_proj_sin = vtk.vtkPolyDataMapper()
# ptsMapper_proj_sin.SetInputData( vertFilter_proj_sin.GetOutput() )

# ptsActor_proj_sin = vtk.vtkActor()
# ptsActor_proj_sin.SetMapper( ptsMapper_proj_sin )
# ptsActor_proj_sin.GetProperty().SetPointSize( 25 )
# ptsActor_proj_sin.GetProperty().SetColor( 0, 0, 1 )
# ptsActor_proj_sin.GetProperty().SetOpacity( 1.0 )
# ptsActor_proj_sin.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# Visualize Subject-Specific Data - Geodesics
red = Color( "red" )
colors_2 = list( red.range_to( Color( "yellow" ), nSubj ) ) 

actorArr = []

for i in range( nSubj ):	
	points_i = vtk.vtkPoints()

	pt_list_i = pt_list[ i ]
	t_list_i = t_list[ i ] 

	for j in range( nData_Subj ):
		points_i.InsertNextPoint( pt_list_i[ j ].pt[ 0 ], pt_list_i[ j ].pt[ 1 ], pt_list_i[ j ].pt[ 2 ] )

	polyData_i = vtk.vtkPolyData()
	polyData_i.SetPoints( points_i )

	vtxFilter_i = vtk.vtkVertexGlyphFilter()
	vtxFilter_i.SetInputData( polyData_i )
	vtxFilter_i.Update()

	ptsMapper_i = vtk.vtkPolyDataMapper()
	ptsMapper_i.SetInputData( vtxFilter_i.GetOutput() )

	ptsActor_i = vtk.vtkActor()
	ptsActor_i.SetMapper( ptsMapper_i )
	ptsActor_i.GetProperty().SetPointSize( 15 )
	ptsActor_i.GetProperty().SetColor( colors_2[ i ].rgb[ 0 ], colors_2[ i ].rgb[ 1 ], colors_2[ i ].rgb[ 2 ] )
	ptsActor_i.GetProperty().SetOpacity( 1.0 )
	ptsActor_i.GetProperty().SetRenderPointsAsSpheres( 1 ) 

	actorArr.append( ptsActor_i )

# Visualize the longitudinal data set cross-sectionally 
nBWColor = 255
t_bw_step = ( ( bias_param_t1 + subj_t1 ) - ( bias_param_t0 + subj_t0 ) ) / float( nBWColor - 1 )

allPts_vtk = vtk.vtkPoints()

allPtsData_vtk = vtk.vtkPolyData() 

BWColorPointArr = vtk.vtkUnsignedCharArray() 
BWColorPointArr.SetNumberOfComponents( 3 )
BWColorPointArr.SetName( "Colors" ) 

for i in range( len( pt_list_all ) ):
	allPts_vtk.InsertNextPoint( pt_list_all[ i ].pt[ 0 ], pt_list_all[ i ].pt[ 1 ], pt_list_all[ i ].pt[ 2 ] )

	# BWColorPointArr.InsertNextTuple( [ 0, 255, 0 ] )

	BWColorPointArr.InsertNextTuple( [ int( ( ( t_list_all[ i ] - ( bias_param_t0 + subj_t0) ) / t_bw_step ) ), int( ( ( t_list_all[ i ] - ( bias_param_t0 + subj_t0) ) / t_bw_step ) ), int( ( ( t_list_all[ i ] - ( bias_param_t0 + subj_t0) ) / t_bw_step ) ) ] )

BWColorPointArr.Modified()

allPtsData_vtk.SetPoints( allPts_vtk )
allPtsData_vtk.Modified()

allPtsDataVertex = vtk.vtkVertexGlyphFilter()
allPtsDataVertex.SetInputData( allPtsData_vtk )
allPtsDataVertex.Update()

allPtsData_vis = allPtsDataVertex.GetOutput()
allPtsData_vis.GetPointData().SetScalars( BWColorPointArr )

allPtsMapper = vtk.vtkPolyDataMapper()
allPtsMapper.SetInputData( allPtsData_vis )

allPtsActor= vtk.vtkActor()
allPtsActor.SetMapper( allPtsMapper )
allPtsActor.GetProperty().SetPointSize( 15 )
allPtsActor.GetProperty().SetOpacity( 1.0 )
allPtsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 


# Renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

light = vtk.vtkLight() 
light.SetFocalPoint(1.875,0.6125,0)
light.SetPosition(0.875,1.6125,1)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

# Background Sphere
ren.AddActor( conActor_sph )

# # Ground Truth Population Trend
# ren.AddActor( ptsActor_proj_sin )

# Subject-wise Trends
for i in range( nSubj ):
	ren.AddActor( actorArr[ i ] )

# # Estimated Geodesic Regression
# ren.AddActor( egRegAll_lineActor )

# # Estimated Kernel Regression
# ren.AddActor( wAvg_ptsActor )

# # Estimated WGP - BPF Geodesic
# ren.AddActor( test_ptsActor )

# # Confidence Interval
# ren.AddActor( conActor )


# # Estimated WGP - BPF Frechet Mean
# ren.AddActor( test_mu_ptsActor )

renWin.Render()
renIn.Start()

# Remove Subject-wise actors
for i in range( nSubj ):
	ren.RemoveActor( actorArr[ i ] )

ren.AddActor( allPtsActor )

renWin.Render()
renIn.Start()

for i in range( nSubj ):
	actorArr[i ].GetProperty().SetOpacity( 0.2 )
	ren.AddActor( actorArr[ i ] )

ren.AddActor( HGeoRegActor )
ren.AddActor( geoRegActor )
ren.AddActor( test_ptsActor )
ren.AddActor( test_ptsActor_Lon )
ren.AddActor( test_ptsActor_Lon_PT )


renWin.Render()
renIn.Start()
