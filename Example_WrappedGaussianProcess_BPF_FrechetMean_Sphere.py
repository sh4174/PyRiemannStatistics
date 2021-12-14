# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds 
import numpy as np
import StatsModel as sm

# Visualization
import vtk

# Time
import time

# Parameters
nDimManifold = 3

# Ground Truth
p_interp = manifolds.sphere( nDimManifold )
v_slope = manifolds.sphere_tVec( nDimManifold )

p_interp.SetPoint( [ 0.0, 0.0, 1.0 ] )
v_slope.SetTangentVector( [ 0, np.pi * 0.25, 0 ] )

# Generating sphere manifolds. distributed over time perturbed by Gaussian random
# Time
t0 = 0	
t1 = 2

# Generate a random point on the manifold
nData = 15
dim = nDimManifold
sigma = 0.15

pt_list = []
t_list = []

for n in range( nData ):
	# time_pt = np.random.uniform( t0, t1 )
	time_pt = ( t1 - t0 ) * n / nData + t0

	# Generate a random Gaussians with polar Box-Muller Method
	rand_pt = np.zeros( nDimManifold ).tolist()

	for i in range( dim ):
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

	v_t = manifolds.sphere_tVec( nDimManifold ) 

	for i in range( nDimManifold ):
		v_t.tVector[ i ] = v_slope.tVector[ i ] * time_pt

	mean = p_interp.ExponentialMap( v_t )

	# print( "Mean At Time : " + str( time_pt ) )	
	# print( mean.pt )

	# Projected Tangent to Mean Point
	rand_tVec_projected = mean.ProjectTangent( mean, rand_tVec )

	# print( "Random Tangent" )
	# print( rand_tVec.tVector )

	# print( "Projected Random Tangent" )
	# print( rand_tVec_projected.tVector )

	# Perturbed point at time_pt 
	pt_perturbed = mean.ExponentialMap( rand_tVec_projected )

	# print( "Perturbed pt At Time : " + str( time_pt ) )	
	# print( pt_perturbed.pt )

	pt_list.append( pt_perturbed )
	t_list.append( time_pt )

##################################
# Linearized Geodesic Regression #
##################################
step_size = 0.01
max_iter = 500
step_tol = 1e-8

start_egreg = time.time() 
base_lReg, tangent_lReg = sm.LinearizedGeodesicRegression( t_list, pt_list, max_iter, step_size, step_tol, False, False )
end_egreg = time.time()

##################################
#    Wrapped Gaussian Process    #
##################################
# BPF : Mean
mean_mu = sm.FrechetMean( pt_list )

nPt_line = 100
l = 0.5
sigma_f = 1

def kernel_k( x, y, sigma_f=1, l=0.5 ):
	return sigma_f * sigma_f * np.exp( -( ( x - y )**2 ) / ( 2 * (l**2) ) )

# Draw weighted means 
min_t = np.min( t_list )
max_t = np.max( t_list )
t_step = float( max_t - min_t ) / float( nPt_line - 1 )

# Weighted Average over time
wAvg_list = [] 

for i in range( nPt_line ):
	t_i = min_t + ( float( i ) * t_step )
	w_list_i = [] 

	for j in range( len( pt_list ) ):
		w_list_i.append( kernel_k( t_i, t_list[ j ], sigma_f=sigma_f, l=l ) )

	print( np.sum( w_list_i ) )
	w_list_i = np.divide( w_list_i, np.sum( w_list_i ) )
	print( np.sum( w_list_i ) )

	w_avg_i = sm.WeightedFrechetMean( pt_list, w_list_i )
	wAvg_list.append( w_avg_i )

# Transform Data to a set of tangent vectors 
nData = len( pt_list )

tVec_list = [] 
tVec_mat_list = []

p_tVec_train_mat = np.zeros( [ nData, 3 ] )

for i in range( len( pt_list ) ):
	tVec_i = mean_mu.LogMap( pt_list[ i ] )

	tVec_list.append( tVec_i )
	tVec_mat_list.append( tVec_i.tVector )
	p_tVec_train_mat[ i, 0 ] = tVec_i.tVector[ 0 ]
	p_tVec_train_mat[ i, 1 ] = tVec_i.tVector[ 1 ]
	p_tVec_train_mat[ i, 2 ] = tVec_i.tVector[ 2 ]

# K
K_tr_tr = np.zeros( [ nData, nData ] )

for i in range( nData ):
	for j in range( nData ):
		K_tr_tr[ i, j ] = kernel_k( t_list[ i ], t_list[ j ], sigma_f=sigma_f, l=l )

# Add Noisy Data Variation
for i in range( nData ):
	K_tr_tr[ i, i ] = K_tr_tr[ i, i ] + sigma


# Test Data 
nTestData = 50

t_test_list = np.linspace( np.min( t_list ), np.max( t_list ), num=nTestData )

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

# Sigma_test
Sigma_test = np.subtract( K_test_test, np.matmul( K_test_tr, np.matmul( np.linalg.inv( K_tr_tr ), K_test_tr.T ) ) )
pt_test_list = []
pt_test_list_p2s = []
pt_test_list_n2s = []

for i in range( nTestData ):
	tVec_arr_i = p_test_tVec_mat[ i, : ]
	v_test_i = manifolds.sphere_tVec( nDimManifold )
	v_test_i.SetTangentVector( [ tVec_arr_i[ 0 ], tVec_arr_i[ 1 ], tVec_arr_i[ 2 ] ] )
	p_test_i = mean_mu.ExponentialMap( v_test_i )

	pt_test_list.append( p_test_i )

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


# Cyan - Weighted Average-like Gaussian Process 
pt_test_list_wAvg_GProcess = [] 

for i in range( nTestData ):
	# weight_K_i = np.divide( K_star_K_inv[ i, : ], np.sum( K_star_K_inv[ i, : ] ) )
	weight_K_i = K_star_K_inv[ i, : ]
	w_avg_i = sm.WeightedFrechetMean( pt_list, weight_K_i )
	pt_test_list_wAvg_GProcess.append( w_avg_i )

	print( np.sum( weight_K_i ) )

wAvgTest_points = vtk.vtkPoints()

for i in range( len( pt_test_list_wAvg_GProcess ) ):
	wAvgTest_points.InsertNextPoint( pt_test_list_wAvg_GProcess[ i ].pt[0], pt_test_list_wAvg_GProcess[ i ].pt[1], pt_test_list_wAvg_GProcess[ i ].pt[2] )

wAvgTest_ptsPolyData = vtk.vtkPolyData()
wAvgTest_ptsPolyData.SetPoints( wAvgTest_points )

wAvgTest_vertFilter = vtk.vtkVertexGlyphFilter()
wAvgTest_vertFilter.SetInputData( wAvgTest_ptsPolyData )
wAvgTest_vertFilter.Update()

wAvgTest_ptsMapper = vtk.vtkPolyDataMapper()
wAvgTest_ptsMapper.SetInputData( wAvgTest_vertFilter.GetOutput() )

wAvgTest_ptsActor = vtk.vtkActor()
wAvgTest_ptsActor.SetMapper( wAvgTest_ptsMapper )
wAvgTest_ptsActor.GetProperty().SetPointSize( 10 )
wAvgTest_ptsActor.GetProperty().SetColor( 0, 1, 1 )
wAvgTest_ptsActor.GetProperty().SetOpacity( 1.0 )
wAvgTest_ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# # Confidence Interval - Confidence Points
# sphere_confidence = vtk.vtkSphereSource()
# sphere_confidence.SetThetaResolution( 120 )
# sphere_confidence.SetPhiResolution( 120 )
# sphere_confidence.SetRadius( 1.0 )
# sphere_confidence.SetCenter( 0.0, 0.0, 0.0 )
# sphere_confidence.Update()

# sphere_conf_data = sphere_confidence.GetOutput()

# for i in range( sphere_conf_data.GetNumberOfPoints() ):
# 	sphere_pt_i = sphere_conf_data.GetPoint( i )

# 	for j in range( nTestData ):
# 		mu_test_j = pt_test_list[ j ]

# 		sigma_test_j = 



# # Assume mu_2 = 0 --> This will make the GP go back to BPF (Frechet Mean)

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

# Visualize spherical points - Red
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
ptsActor.GetProperty().SetPointSize( 30 )
ptsActor.GetProperty().SetColor( 0, 0, 1 )
ptsActor.GetProperty().SetOpacity( 1.0 )
ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# Visualize an initial anchor point - Magenta
points_init = vtk.vtkPoints()

points_init.InsertNextPoint( pt_list[ 0 ].pt[0], pt_list[ 0 ].pt[1], pt_list[ 0 ].pt[2] )

ptInitPolyData = vtk.vtkPolyData()
ptInitPolyData.SetPoints( points_init )

init_vertFilter = vtk.vtkVertexGlyphFilter()
init_vertFilter.SetInputData( ptInitPolyData )
init_vertFilter.Update()

ptInitMapper = vtk.vtkPolyDataMapper()
ptInitMapper.SetInputData( init_vertFilter.GetOutput() )

ptInitActor = vtk.vtkActor()
ptInitActor.SetMapper( ptInitMapper )
ptInitActor.GetProperty().SetPointSize( 30 )
ptInitActor.GetProperty().SetColor( 1, 0, 0 )
ptInitActor.GetProperty().SetOpacity( 1.0 )
ptInitActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

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
wAvgActor.GetProperty().SetColor( 0.0, 1.0, 0.0 )
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

ren.AddActor( conActor )
ren.AddActor( ptsActor )
ren.AddActor( lineActor )
ren.AddActor( estLineActor_LReg )
ren.AddActor( wAvgActor )
ren.AddActor( test_ptsActor )
ren.AddActor( wAvgTest_ptsActor )

renWin.Render()
renIn.Start()

print( "EGReg time" )
print( end_egreg - start_egreg )


