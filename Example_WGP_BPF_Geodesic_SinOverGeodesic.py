# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds 
import numpy as np
import StatsModel as sm

# Visualization
import vtk

# Time
import time

# Generating sphere manifolds. distributed over time perturbed by Gaussian random
# Time
t0 = 0	
t1 = 2

# Generate a random point on the manifold
# Parameters
nDimManifold = 3
nData = 20
dim = nDimManifold
sigma = 0.15

# Ground Truth
p_interp = manifolds.sphere( nDimManifold )
v_slope = manifolds.sphere_tVec( nDimManifold )

p_interp.SetPoint( [ 0.0, 0.0, 1.0 ] )
v_slope.SetTangentVector( [ 0, np.pi * 0.25, 0 ] )

# Point List
# Base Function ( Geodesic )
pt_list = []

# Projected sin on the base point function at time t
pt_list_proj_sin = []

# Time List
t_list = []

# Points Generation over a sin function
for n in range( nData ):
	time_pt = ( t1 - t0 ) * n / nData + t0

	# Generate a random Gaussians with polar Box-Muller Method
	rand_pt = np.zeros( nDimManifold ).tolist()

	v_t = manifolds.sphere_tVec( nDimManifold ) 

	for i in range( nDimManifold ):
		v_t.tVector[ i ] = v_slope.tVector[ i ] * time_pt

	mean = p_interp.ExponentialMap( v_t )
	pt_list.append( mean )

	# Project sin
	v_t_sin = manifolds.sphere_tVec( nDimManifold )
	v_t_sin.tVector[ 0 ] = np.sin( time_pt * np.pi * 3 ) / 2.0 
	v_t_sin.tVector[ 1 ] = 0
	v_t_sin.tVector[ 2 ] = 0

	v_t_proj_sin = mean.ProjectTangent( mean, v_t_sin )
	pt_proj_sin = mean.ExponentialMap( v_t_proj_sin )

	pt_list_proj_sin.append( pt_proj_sin )

	t_list.append( time_pt )

##################################
# Linearized Geodesic Regression #
##################################
step_size = 0.01
max_iter = 500
step_tol = 1e-8

start_egreg = time.time() 
base_lReg, tangent_lReg = sm.LinearizedGeodesicRegression( t_list, pt_list_proj_sin, max_iter, step_size, step_tol, False, False )
end_egreg = time.time()


nPt_line = 300
l = 0.1
sigma_f = 0.15


sigma = 0.01

def kernel_k( x, y, sigma_f=1, l=0.5 ):
	return sigma_f * np.exp( -( ( x - y )**2 ) / ( 2 * (l**2) ) )


##################################
# 	    Kernel Regression	     #
##################################

# Draw weighted means 
min_t = np.min( t_list )
max_t = np.max( t_list )
t_step = float( max_t - min_t ) / float( nPt_line - 1 )

# Weighted Average over time
wAvg_list = [] 
t_list_out = []

for i in range( nPt_line ):
	t_i = min_t + ( float( i ) * t_step )
	t_list_out.append( t_i ) 

	w_list_i = [] 

	for j in range( len( pt_list_proj_sin ) ):
		w_list_i.append( kernel_k( t_i, t_list[ j ], sigma_f=sigma_f, l=l ) )

	print( np.sum( w_list_i ) )
	w_list_i = np.divide( w_list_i, np.sum( w_list_i ) )
	print( np.sum( w_list_i ) )

	w_avg_i = sm.WeightedFrechetMean( pt_list_proj_sin, w_list_i )
	wAvg_list.append( w_avg_i )

##################################
#    Wrapped Gaussian Process    #
##################################
# BPF : Geodesic function / Frechet Mean 
mean_mu = sm.FrechetMean( pt_list_proj_sin )

# Transform Data to a set of tangent vectors 
nData = len( pt_list_proj_sin )


# BPF - Geodesic
tVec_list = [] 
tVec_mat_list = []
p_tVec_train_mat = np.zeros( [ nData, 3 ] )

# BPF - Frechet Mean 
tVec_mu_list = []
tVec_mu_mat_list = []
p_tVec_mu_train_mat = np.zeros( [ nData, 3 ] )


for i in range( len( pt_list_proj_sin ) ):	
	# Base point on a geodesic 
	t_i = t_list[ i ]
	bp_i = base_lReg.ExponentialMap( tangent_lReg.ScalarMultiply( t_i ) )
	tVec_i = bp_i.LogMap( pt_list_proj_sin[ i ] )
	tVec_list.append( tVec_i )
	tVec_mat_list.append( tVec_i.tVector )
	p_tVec_train_mat[ i, 0 ] = tVec_i.tVector[ 0 ]
	p_tVec_train_mat[ i, 1 ] = tVec_i.tVector[ 1 ]
	p_tVec_train_mat[ i, 2 ] = tVec_i.tVector[ 2 ]

	# Base point on a Frechet mean 
	tVec_mu_i = mean_mu.LogMap( pt_list_proj_sin[ i ] )
	tVec_mu_list.append( tVec_mu_i ) 
	tVec_mu_mat_list.append( tVec_mu_i.tVector )

	p_tVec_mu_train_mat[ i, 0 ] = tVec_mu_i.tVector[ 0 ]
	p_tVec_mu_train_mat[ i, 1 ] = tVec_mu_i.tVector[ 1 ]
	p_tVec_mu_train_mat[ i, 2 ] = tVec_mu_i.tVector[ 2 ]


# K
K_tr_tr = np.zeros( [ nData, nData ] )

for i in range( nData ):
	for j in range( nData ):
		K_tr_tr[ i, j ] = kernel_k( t_list[ i ], t_list[ j ], sigma_f=sigma_f, l=l )

# Add Noisy Data Variation
for i in range( nData ):
	K_tr_tr[ i, i ] = K_tr_tr[ i, i ] + sigma


# Test Data 
nTestData = nPt_line

t_test_list = np.linspace( np.min( t_list ), np.max( t_list ), num=nTestData )

## Ground Truth Sin over Geodesic
# Projected sin on the base point function at time t
gt_out_list_proj_sin = []

# Points Generation over a sin function
for n in range( len( t_test_list ) ):
	time_pt = t_test_list[ n ]

	v_t = manifolds.sphere_tVec( nDimManifold ) 

	for i in range( nDimManifold ):
		v_t.tVector[ i ] = v_slope.tVector[ i ] * time_pt

	mean = p_interp.ExponentialMap( v_t )

	# Project sin
	v_t_sin = manifolds.sphere_tVec( nDimManifold )
	v_t_sin.tVector[ 0 ] = np.sin( time_pt * np.pi * 3 ) / 2.0 
	v_t_sin.tVector[ 1 ] = 0
	v_t_sin.tVector[ 2 ] = 0

	v_t_proj_sin = mean.ProjectTangent( mean, v_t_sin )
	pt_proj_sin = mean.ExponentialMap( v_t_proj_sin )

	gt_out_list_proj_sin.append( pt_proj_sin )


# K_**
K_test_test = np.zeros( [ nTestData, nTestData ] )

# K_*
K_test_tr = np.zeros( [ nTestData, nData ] )

for i in range( nTestData ):
	for j in range( nTestData ):
		K_test_test[ i, j ] = kernel_k( t_test_list[ i ], t_test_list[ j ], sigma_f=sigma_f, l=l )


for i in range( nTestData ):
	for j in range( nData ):
		K_test_tr[ i, j ] = kernel_k( t_test_list[ i ], t_list[ j ], sigma_f=sigma_f, l=l )

# K_* x K^{-1} 
K_star_K_inv = np.matmul( K_test_tr, np.linalg.inv( K_tr_tr ) )

# mu_test
p_test_tVec_mat = np.matmul( K_star_K_inv, p_tVec_train_mat )

# mu_test_frechet_mean 
p_test_tVec_mu_mat = np.matmul( K_star_K_inv, p_tVec_mu_train_mat )


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
	bp_i = base_lReg.ExponentialMap( tangent_lReg.ScalarMultiply( t_test_i ) ) 

	p_test_i = bp_i.ExponentialMap( v_test_i )
	pt_test_list.append( p_test_i )
	sigma_test_list.append( Sigma_test[ i, i ] )

	# From Frechet Mean BPF
	tVec_arr_mu_i = p_test_tVec_mu_mat[ i, : ]
	v_test_mu_i = manifolds.sphere_tVec( nDimManifold )
	v_test_mu_i.SetTangentVector( [ tVec_arr_mu_i[ 0 ], tVec_arr_mu_i[ 1 ], tVec_arr_mu_i[ 2 ] ] ) 

	p_test_mu_i = mean_mu.ExponentialMap( v_test_mu_i )
	pt_test_mu_list.append( p_test_mu_i ) 

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


## Test Points Visualization - BPF - Frechet Mean
test_mu_points = vtk.vtkPoints()

for i in range( len( pt_test_mu_list ) ):
	test_mu_points.InsertNextPoint( pt_test_mu_list[ i ].pt[0], pt_test_mu_list[ i ].pt[1], pt_test_mu_list[ i ].pt[2] )

test_mu_ptsPolyData = vtk.vtkPolyData()
test_mu_ptsPolyData.SetPoints( test_mu_points )

test_mu_vertFilter = vtk.vtkVertexGlyphFilter()
test_mu_vertFilter.SetInputData( test_mu_ptsPolyData )
test_mu_vertFilter.Update()

test_mu_ptsMapper = vtk.vtkPolyDataMapper()
test_mu_ptsMapper.SetInputData( test_mu_vertFilter.GetOutput() )

# Red - Gaussian Process
test_mu_ptsActor = vtk.vtkActor()
test_mu_ptsActor.SetMapper( test_mu_ptsMapper )
test_mu_ptsActor.GetProperty().SetPointSize( 10 )
test_mu_ptsActor.GetProperty().SetColor( 1, 0, 0 )
test_mu_ptsActor.GetProperty().SetOpacity( 1.0 )
test_mu_ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

## Ground Truth Point Visualization - Sin over Geodesic
GT_sinPoints = vtk.vtkPoints()

for i in range( len( gt_out_list_proj_sin ) ):
	GT_sinPoints.InsertNextPoint( gt_out_list_proj_sin[ i ].pt[0], gt_out_list_proj_sin[ i ].pt[1], gt_out_list_proj_sin[ i ].pt[2] )

GT_sinPoints_ptsPolyData = vtk.vtkPolyData()
GT_sinPoints_ptsPolyData.SetPoints( GT_sinPoints )

GT_sinPoints_vertFilter = vtk.vtkVertexGlyphFilter()
GT_sinPoints_vertFilter.SetInputData( GT_sinPoints_ptsPolyData )
GT_sinPoints_vertFilter.Update()

GT_sinPoints_ptsMapper = vtk.vtkPolyDataMapper()
GT_sinPoints_ptsMapper.SetInputData( GT_sinPoints_vertFilter.GetOutput() )

# Red - Gaussian Process
GT_sinPoints_ptsActor = vtk.vtkActor()
GT_sinPoints_ptsActor.SetMapper( GT_sinPoints_ptsMapper )
GT_sinPoints_ptsActor.GetProperty().SetPointSize( 10 )
GT_sinPoints_ptsActor.GetProperty().SetColor( 0, 0, 1 )
GT_sinPoints_ptsActor.GetProperty().SetOpacity( 1.0 )
GT_sinPoints_ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 


#############################################
# There is something wrong with probability
# Maybe change it to the confidence interval with 2 sigma
#############################################

# Calculate Uncertainty of the GP estimation
# Sample points on sphere 
sphere_uncertainty = vtk.vtkSphereSource()
sphere_uncertainty.SetThetaResolution( 720 )
sphere_uncertainty.SetPhiResolution( 360 )
sphere_uncertainty.SetRadius( 1.0 )
sphere_uncertainty.SetCenter( 0.0, 0.0, 0.0 )
sphere_uncertainty.SetLatLongTessellation( False )
sphere_uncertainty.Update()

sphere_uncertainty_polyData = sphere_uncertainty.GetOutput()

# The number of points
nSpherePt = sphere_uncertainty_polyData.GetNumberOfPoints()
prob_sphere_GP = np.zeros( [ nSpherePt, len( pt_test_list ) ] )

max_prob_sphere_arr = np.zeros( nSpherePt )


for j in range( len( pt_test_list ) ):
	test_j = pt_test_list[ j ]

	prob_surface_j = np.zeros( nSpherePt )

	for i in range( nSpherePt ):
		sphere_pt_i_arr = sphere_uncertainty_polyData.GetPoint( i )
		sphere_pt_i = manifolds.sphere( 3 )
		sphere_pt_i.SetPoint( sphere_pt_i_arr )

		log_test_j_to_sphere_pt_i = test_j.LogMap( sphere_pt_i )

		distSq_sphere_pt_i_to_test_j = log_test_j_to_sphere_pt_i.normSquared()
		dist_sphere_pt_i_to_test_j = log_test_j_to_sphere_pt_i.norm()
		# print( "Sigma_j_j" ) 
		# print( sigma_test_list[ j ] )
		# print( "Distance_i_j" )
		# print( distSq_sphere_pt_i_to_test_j )

		if dist_sphere_pt_i_to_test_j < 2 * np.sqrt( sigma_test_list[ j ] ):
			max_prob_sphere_arr[ i ] = 1 

print( "Maximum Sigma" )
print( np.max( sigma_test_list ) )

# Color Schemes 
lut = vtk.vtkLookupTable()
lut.SetNanColor( 1, 0, 0, 1 )
lut.SetTableRange( 0.0, 1.0 )
lut.SetNumberOfTableValues( 2 )
lut.Build()

colorPointArr = vtk.vtkUnsignedCharArray() 
colorPointArr.SetNumberOfComponents( 3 )
colorPointArr.SetName( "Colors" ) 

for i in range( len( max_prob_sphere_arr) ):
	if max_prob_sphere_arr[ i ] > 0.5:
		color_i = lut.GetTableValue( int( max_prob_sphere_arr[ i ] ) )
		colorPointArr.InsertNextTuple( [ 0, 0, 255 ] )
		# color_i = lut.GetTableValue( int( max_prob_sphere_arr[ i ] ) )
		# colorPointArr.InsertNextTuple( [ int( color_i[ 0 ] * 255 ), int( color_i[ 1 ] * 255 ), int( color_i[ 2 ] * 255 ), 0.5 ] )
	else:
		color_i = lut.GetTableValue( int( max_prob_sphere_arr[ i ] ) )
		colorPointArr.InsertNextTuple( [ 255, 255, 255 ] )

sphere_uncertainty_polyData.GetPointData().SetScalars( colorPointArr )
sphere_uncertainty_polyData.Modified()



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


conMapper = vtk.vtkPolyDataMapper()
conMapper.SetInputData( sphere_uncertainty_polyData )
# conMapper.ScalarVisibilityOn()
# conMapper.SetScalarRange( 0, 1 )
conMapper.Update()

conActor = vtk.vtkActor()
conActor.SetMapper( conMapper )
conActor.GetProperty().SetOpacity( 0.5 )
# conActor.GetProperty().SetColor( 0.9, 0.9, 0.9 )
# conActor.GetProperty().SetEdgeColor( 0.4, 0.4, 0.7 )
conActor.GetProperty().EdgeVisibilityOff()
conActor.GetProperty().SetAmbient(0.3)
conActor.GetProperty().SetDiffuse(0.375)
conActor.GetProperty().SetSpecular(0.0)

# Visualize base points - Blue : pt_list
points = vtk.vtkPoints()

for i in range( len( pt_list ) ):
	points.InsertNextPoint( pt_list[ i ].pt[0], pt_list[ i ].pt[1], pt_list[ i ].pt[2] )

ptsPolyData = vtk.vtkPolyData()
ptsPolyData.SetPoints( points )

vertFilter = vtk.vtkVertexGlyphFilter()
vertFilter.SetInputData( ptsPolyData )
vertFilter.Update()

ptsMapper = vtk.vtkPolyDataMapper()
ptsMapper.SetInputData( vertFilter.GetOutput() )

ptsActor = vtk.vtkActor()
ptsActor.SetMapper( ptsMapper )
ptsActor.GetProperty().SetPointSize( 15 )
ptsActor.GetProperty().SetColor( 0, 0, 1 )
ptsActor.GetProperty().SetOpacity( 1.0 )
ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# # Visualize additive sin function - Red : pt_list_add_sin
# points_add_sin = vtk.vtkPoints()

# for i in range( len( pt_list_add_sin ) ):
# 	points_add_sin.InsertNextPoint( pt_list_add_sin[ i ].pt[0], pt_list_add_sin[ i ].pt[1], pt_list_add_sin[ i ].pt[2] )

# ptsPolyData_add_sin = vtk.vtkPolyData()
# ptsPolyData_add_sin.SetPoints( points_add_sin )

# vertFilter_add_sin = vtk.vtkVertexGlyphFilter()
# vertFilter_add_sin.SetInputData( ptsPolyData_add_sin )
# vertFilter_add_sin.Update()

# ptsMapper_add_sin = vtk.vtkPolyDataMapper()
# ptsMapper_add_sin.SetInputData( vertFilter_add_sin.GetOutput() )

# ptsActor_add_sin = vtk.vtkActor()
# ptsActor_add_sin.SetMapper( ptsMapper_add_sin )
# ptsActor_add_sin.GetProperty().SetPointSize( 15 )
# ptsActor_add_sin.GetProperty().SetColor( 1, 0, 0 )
# ptsActor_add_sin.GetProperty().SetOpacity( 1.0 )
# ptsActor_add_sin.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# Visualize projected sin function - Green : pt_list
points_proj_sin = vtk.vtkPoints()

for i in range( len( pt_list_proj_sin ) ):
	points_proj_sin.InsertNextPoint( pt_list_proj_sin[ i ].pt[0], pt_list_proj_sin[ i ].pt[1], pt_list_proj_sin[ i ].pt[2] )

ptsPolyData_proj_sin = vtk.vtkPolyData()
ptsPolyData_proj_sin.SetPoints( points_proj_sin )

vertFilter_proj_sin = vtk.vtkVertexGlyphFilter()
vertFilter_proj_sin.SetInputData( ptsPolyData_proj_sin )
vertFilter_proj_sin.Update()

ptsMapper_proj_sin = vtk.vtkPolyDataMapper()
ptsMapper_proj_sin.SetInputData( vertFilter_proj_sin.GetOutput() )

ptsActor_proj_sin = vtk.vtkActor()
ptsActor_proj_sin.SetMapper( ptsMapper_proj_sin )
ptsActor_proj_sin.GetProperty().SetPointSize( 15 )
ptsActor_proj_sin.GetProperty().SetColor( 0, 1, 0 )
ptsActor_proj_sin.GetProperty().SetOpacity( 1.0 )
ptsActor_proj_sin.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# Visualize a ground truth trend line - yellow 
trend_pt_list = []
trend_t_list = []
nTimePt = 100

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ v_slope.tVector[0] * time_pt, v_slope.tVector[1] * time_pt, v_slope.tVector[2] * time_pt ]  )
	mean = p_interp.ExponentialMap( v_t )

	trend_pt_list.append( mean )
	trend_t_list.append( time_pt )

linePolyData = vtk.vtkPolyData()
linePts = vtk.vtkPoints()

for i in range( len( trend_pt_list ) ):
	linePts.InsertNextPoint( trend_pt_list[ i ].pt[0], trend_pt_list[ i ].pt[1], trend_pt_list[ i ].pt[2] )

linePolyData.SetPoints( linePts )

lines = vtk.vtkCellArray()

for i in range( len( trend_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	lines.InsertNextCell( line_i )

linePolyData.SetLines( lines )

lineMapper = vtk.vtkPolyDataMapper()
lineMapper.SetInputData( linePolyData )

lineActor = vtk.vtkActor()
lineActor.SetMapper( lineMapper )
lineActor.GetProperty().SetColor( 1, 1, 0 )
lineActor.GetProperty().SetOpacity( 1.0 )
lineActor.GetProperty().SetLineWidth( 10 )
lineActor.GetProperty().SetRenderLinesAsTubes( 1 ) 

# Visualize an estimated trend of Linearized Geodesic Regression Model - NYU Violet
est_trend_LReg_pt_list = []
est_trend_LReg_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ tangent_lReg.tVector[0] * time_pt, tangent_lReg.tVector[1] * time_pt, tangent_lReg.tVector[2] * time_pt ]  )
	mean = base_lReg.ExponentialMap( v_t )

	est_trend_LReg_pt_list.append( mean )
	est_trend_LReg_t_list.append( time_pt )

estLinePolyData_LReg = vtk.vtkPolyData()
estLinePts_LReg = vtk.vtkPoints()

for i in range( len( est_trend_LReg_pt_list ) ):
	estLinePts_LReg.InsertNextPoint( est_trend_LReg_pt_list[ i ].pt[0], est_trend_LReg_pt_list[ i ].pt[1], est_trend_LReg_pt_list[ i ].pt[2] )

estLinePolyData_LReg.SetPoints( estLinePts_LReg )
estLines_LReg = vtk.vtkCellArray()

for i in range( len( est_trend_LReg_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	estLines_LReg.InsertNextCell( line_i )

estLinePolyData_LReg.SetLines( estLines_LReg )

estLineMapper_LReg = vtk.vtkPolyDataMapper()
estLineMapper_LReg.SetInputData( estLinePolyData_LReg )

estLineActor_LReg = vtk.vtkActor()
estLineActor_LReg.SetMapper( estLineMapper_LReg )
estLineActor_LReg.GetProperty().SetColor( 0.980, 0.447,0.408 )
estLineActor_LReg.GetProperty().SetOpacity( 0.7 )
estLineActor_LReg.GetProperty().SetLineWidth( 30 )
estLineActor_LReg.GetProperty().SetRenderLinesAsTubes( 1 ) 


# Weighted Averages Over Time 
wAvg_PolyData = vtk.vtkPolyData()
wAvg_Pts = vtk.vtkPoints()
wAvg_Lines = vtk.vtkCellArray()

for i in range( len( wAvg_list ) ):
	wAvg_Pts.InsertNextPoint( wAvg_list[ i ].pt[ 0 ], wAvg_list[ i ].pt[ 1 ], wAvg_list[ i ].pt[ 2 ] )

for i in range( len( wAvg_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 0, i + 1 )
	wAvg_Lines.InsertNextCell( line_i )

wAvg_PolyData.SetPoints( wAvg_Pts )
# wAvg_PolyData.SetLines( wAvg_Lines ) 
wAvg_PolyData.Modified()


wAvg_vertFilter = vtk.vtkVertexGlyphFilter()
wAvg_vertFilter.SetInputData( wAvg_PolyData )
wAvg_vertFilter.Update()

wAvgMapper = vtk.vtkPolyDataMapper()
wAvgMapper.SetInputData( wAvg_vertFilter.GetOutput() )

wAvgActor = vtk.vtkActor()
wAvgActor.SetMapper( wAvgMapper )
wAvgActor.GetProperty().SetColor( 0.0, 1.0, 1.0 )
wAvgActor.GetProperty().SetOpacity( 0.5 )
wAvgActor.GetProperty().SetLineWidth( 5 )
wAvgActor.GetProperty().SetRenderLinesAsTubes( 0 )
wAvgActor.GetProperty().SetRenderPointsAsSpheres( 1 )
wAvgActor.GetProperty().SetPointSize( 10 )


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

ren.AddActor( conActor_sph )
ren.AddActor( conActor )
ren.AddActor( ptsActor )
ren.AddActor( ptsActor_proj_sin )
ren.AddActor( lineActor )
ren.AddActor( estLineActor_LReg )
ren.AddActor( wAvgActor )
ren.AddActor( test_ptsActor )
ren.AddActor( test_mu_ptsActor )
ren.AddActor( GT_sinPoints_ptsActor )



renWin.Render()
renIn.Start()
