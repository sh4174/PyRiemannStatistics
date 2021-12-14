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
nData = 1000
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

#######################
# Geodesic Regression #
#######################

# Gradient Descent Parameters
step_size = 0.1
max_iter = 200
step_tol = 1e-4


start_greg = time.time()
base_GReg, tangent_GReg = sm.GeodesicRegression( t_list, pt_list, max_iter, step_size, step_tol, False )
end_greg = time.time()

#######################
# Linearized Geodesic Regression #
#######################

step_size = 0.01
max_iter = 500
step_tol = 1e-8

base_lReg1, tangent_lReg1 = sm.LinearizedGeodesicRegression( t_list, pt_list, 1, step_size, step_tol, False, True )

base_lReg2, tangent_lReg2 = sm.LinearizedGeodesicRegression( t_list, pt_list, 2, step_size, step_tol, False, True )

base_lReg3, tangent_lReg3 = sm.LinearizedGeodesicRegression( t_list, pt_list, 3, step_size, step_tol, False, True )

base_lReg4, tangent_lReg4 = sm.LinearizedGeodesicRegression( t_list, pt_list, 4, step_size, step_tol, False, True )

start_egreg = time.time() 
base_lReg, tangent_lReg = sm.LinearizedGeodesicRegression( t_list, pt_list, max_iter, step_size, step_tol, False, False )
end_egreg = time.time()


print( "True P" )
print( p_interp.pt )
print( "True V" )
print( v_slope.tVector )

# Linearized Regression Results
print( "=================================================" )
print( "    Resulsts : Linearized Geodesic Regression    " )
print( "=================================================" )

print( "Estimated P" )
print( base_lReg.pt )
print( "Estimated V" ) 
print( tangent_lReg.tVector )

R2 = sm.R2Statistics( t_list, pt_list, base_lReg, tangent_lReg )

print( "R2 Statistics" )
print( R2 )

RMSE = sm.RootMeanSquaredError( t_list, pt_list, base_lReg, tangent_lReg )

print( "RMSE" )
print( RMSE )


# Linearized Regression Results
print( "=================================================" )
print( "    Resulsts : Geodesic Regression    " )
print( "=================================================" )

print( "Estimated P" )
print( base_GReg.pt )
print( "Estimated V" ) 
print( tangent_GReg.tVector )

R2_GReg = sm.R2Statistics( t_list, pt_list, base_GReg, tangent_GReg )

print( "R2 Statistics" )
print( R2_GReg )

RMSE_GReg = sm.RootMeanSquaredError( t_list, pt_list, base_GReg, tangent_GReg )

print( "RMSE" )
print( RMSE_GReg )


# # Permutation Test
# nTrial = 10000

# print( "======================================" )
# print( "Random Permutation Testing........    " )
# print( ( "# of Trials : %d", nTrial )  )
# print( "======================================" )

# P_val = sm.NullHypothesisTestingPermutationTest( t_list, pt_list, base, tangent, nTrial, max_iter, step_size, step_tol )

# print( "P-Value" )
# print( P_val )


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
ptsActor.GetProperty().SetPointSize( 8 )
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

# Visualize an estimated trend - Geodesic Regression : Carolina Blue
est_trend_pt_list = []
est_trend_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ tangent_GReg.tVector[0] * time_pt, tangent_GReg.tVector[1] * time_pt, tangent_GReg.tVector[2] * time_pt ]  )
	mean = base_GReg.ExponentialMap( v_t )

	est_trend_pt_list.append( mean )
	est_trend_t_list.append( time_pt )

estLinePolyData = vtk.vtkPolyData()
estLinePts = vtk.vtkPoints()

for i in range( len( est_trend_pt_list ) ):
	estLinePts.InsertNextPoint( est_trend_pt_list[ i ].pt[0], est_trend_pt_list[ i ].pt[1], est_trend_pt_list[ i ].pt[2] )

estLinePolyData.SetPoints( estLinePts )

estLines = vtk.vtkCellArray()

for i in range( len( est_trend_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	estLines.InsertNextCell( line_i )

estLinePolyData.SetLines( estLines )

estLineMapper = vtk.vtkPolyDataMapper()
estLineMapper.SetInputData( estLinePolyData )

estLineActor = vtk.vtkActor()
estLineActor.SetMapper( estLineMapper )
estLineActor.GetProperty().SetColor( 0.6, 0.729, 0.867 )
estLineActor.GetProperty().SetOpacity( 0.5 )
estLineActor.GetProperty().SetLineWidth( 20 )
estLineActor.GetProperty().SetRenderLinesAsTubes( 0 ) 

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


points_final = vtk.vtkPoints()

points_final.InsertNextPoint( base_lReg3.pt[0], base_lReg3.pt[1], base_lReg3.pt[2] )

ptFinalPolyData = vtk.vtkPolyData()
ptFinalPolyData.SetPoints( points_final )

final_vertFilter = vtk.vtkVertexGlyphFilter()
final_vertFilter.SetInputData( ptFinalPolyData )
final_vertFilter.Update()

ptFinalMapper = vtk.vtkPolyDataMapper()
ptFinalMapper.SetInputData( final_vertFilter.GetOutput() )

ptFinalActor = vtk.vtkActor()
ptFinalActor.SetMapper( ptFinalMapper )
ptFinalActor.GetProperty().SetPointSize( 30 )
ptFinalActor.GetProperty().SetColor( 1, 0, 0 )
ptFinalActor.GetProperty().SetOpacity( 1.0 )
ptFinalActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 

# Visualize an estimated trend of Linearized Geodesic Regression Model - Iteration 0
est_trend_LReg_pt_list1 = []
est_trend_LReg_t_list1 = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ tangent_lReg1.tVector[0] * time_pt, tangent_lReg1.tVector[1] * time_pt, tangent_lReg1.tVector[2] * time_pt ]  )
	mean = base_lReg1.ExponentialMap( v_t )

	est_trend_LReg_pt_list1.append( mean )
	est_trend_LReg_t_list1.append( time_pt )

estLinePolyData_LReg1 = vtk.vtkPolyData()
estLinePts_LReg1 = vtk.vtkPoints()

for i in range( len( est_trend_LReg_pt_list1 ) ):
	estLinePts_LReg1.InsertNextPoint( est_trend_LReg_pt_list1[ i ].pt[0], est_trend_LReg_pt_list1[ i ].pt[1], est_trend_LReg_pt_list1[ i ].pt[2] )

estLinePolyData_LReg1.SetPoints( estLinePts_LReg1 )
estLines_LReg1 = vtk.vtkCellArray()

for i in range( len( est_trend_LReg_pt_list1 ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	estLines_LReg1.InsertNextCell( line_i )

estLinePolyData_LReg1.SetLines( estLines_LReg1 )

estLineMapper_LReg1 = vtk.vtkPolyDataMapper()
estLineMapper_LReg1.SetInputData( estLinePolyData_LReg1 )

estLineActor_LReg1 = vtk.vtkActor()
estLineActor_LReg1.SetMapper( estLineMapper_LReg1 )
estLineActor_LReg1.GetProperty().SetColor( 0.980, 0.447,0.408 )
estLineActor_LReg1.GetProperty().SetOpacity( 0.7 )
estLineActor_LReg1.GetProperty().SetLineWidth( 30 )
estLineActor_LReg1.GetProperty().SetRenderLinesAsTubes( 1 ) 


# Visualize an estimated trend of Linearized Geodesic Regression Model - Iteration 1
est_trend_LReg_pt_list2 = []
est_trend_LReg_t_list2 = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ tangent_lReg2.tVector[0] * time_pt, tangent_lReg2.tVector[1] * time_pt, tangent_lReg2.tVector[2] * time_pt ]  )
	mean = base_lReg2.ExponentialMap( v_t )

	est_trend_LReg_pt_list2.append( mean )
	est_trend_LReg_t_list2.append( time_pt )

estLinePolyData_LReg2 = vtk.vtkPolyData()
estLinePts_LReg2 = vtk.vtkPoints()

for i in range( len( est_trend_LReg_pt_list2 ) ):
	estLinePts_LReg2.InsertNextPoint( est_trend_LReg_pt_list2[ i ].pt[0], est_trend_LReg_pt_list2[ i ].pt[1], est_trend_LReg_pt_list2[ i ].pt[2] )

estLinePolyData_LReg2.SetPoints( estLinePts_LReg2 )
estLines_LReg2 = vtk.vtkCellArray()

for i in range( len( est_trend_LReg_pt_list2 ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	estLines_LReg2.InsertNextCell( line_i )

estLinePolyData_LReg2.SetLines( estLines_LReg2 )

estLineMapper_LReg2 = vtk.vtkPolyDataMapper()
estLineMapper_LReg2.SetInputData( estLinePolyData_LReg2 )

estLineActor_LReg2 = vtk.vtkActor()
estLineActor_LReg2.SetMapper( estLineMapper_LReg2 )
estLineActor_LReg2.GetProperty().SetColor( 0.980, 0.447,0.408 )
estLineActor_LReg2.GetProperty().SetOpacity( 0.7 )
estLineActor_LReg2.GetProperty().SetLineWidth( 30 )
estLineActor_LReg2.GetProperty().SetRenderLinesAsTubes( 1 ) 


points_2 = vtk.vtkPoints()

points_2.InsertNextPoint( base_lReg1.pt[0], base_lReg1.pt[1], base_lReg1.pt[2] )

ptPolyData2 = vtk.vtkPolyData()
ptPolyData2.SetPoints( points_2 )

vertFilter2 = vtk.vtkVertexGlyphFilter()
vertFilter2.SetInputData( ptPolyData2 )
vertFilter2.Update()

ptMapper2 = vtk.vtkPolyDataMapper()
ptMapper2.SetInputData( vertFilter2.GetOutput() )

ptActor2 = vtk.vtkActor()
ptActor2.SetMapper( ptMapper2 )
ptActor2.GetProperty().SetPointSize( 30 )
ptActor2.GetProperty().SetColor( 1, 0, 0 )
ptActor2.GetProperty().SetOpacity( 1.0 )
ptActor2.GetProperty().SetRenderPointsAsSpheres( 1 ) 


# Visualize an estimated trend of Linearized Geodesic Regression Model - Iteration 3
est_trend_LReg_pt_list3 = []
est_trend_LReg_t_list3 = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ tangent_lReg3.tVector[0] * time_pt, tangent_lReg3.tVector[1] * time_pt, tangent_lReg3.tVector[2] * time_pt ]  )
	mean = base_lReg3.ExponentialMap( v_t )

	est_trend_LReg_pt_list3.append( mean )
	est_trend_LReg_t_list3.append( time_pt )

estLinePolyData_LReg3 = vtk.vtkPolyData()
estLinePts_LReg3 = vtk.vtkPoints()

for i in range( len( est_trend_LReg_pt_list3 ) ):
	estLinePts_LReg3.InsertNextPoint( est_trend_LReg_pt_list3[ i ].pt[0], est_trend_LReg_pt_list3[ i ].pt[1], est_trend_LReg_pt_list3[ i ].pt[2] )

estLinePolyData_LReg3.SetPoints( estLinePts_LReg3 )
estLines_LReg3 = vtk.vtkCellArray()

for i in range( len( est_trend_LReg_pt_list3 ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	estLines_LReg3.InsertNextCell( line_i )

estLinePolyData_LReg3.SetLines( estLines_LReg3 )

estLineMapper_LReg3 = vtk.vtkPolyDataMapper()
estLineMapper_LReg3.SetInputData( estLinePolyData_LReg3 )

estLineActor_LReg3 = vtk.vtkActor()
estLineActor_LReg3.SetMapper( estLineMapper_LReg3 )
estLineActor_LReg3.GetProperty().SetColor( 0.980, 0.447,0.408 )
estLineActor_LReg3.GetProperty().SetOpacity( 0.7 )
estLineActor_LReg3.GetProperty().SetLineWidth( 30 )
estLineActor_LReg3.GetProperty().SetRenderLinesAsTubes( 1 ) 


# Visualize an estimated trend of Linearized Geodesic Regression Model - Iteration 3
est_trend_LReg_pt_list4 = []
est_trend_LReg_t_list4 = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ tangent_lReg4.tVector[0] * time_pt, tangent_lReg4.tVector[1] * time_pt, tangent_lReg4.tVector[2] * time_pt ]  )
	mean = base_lReg4.ExponentialMap( v_t )

	est_trend_LReg_pt_list4.append( mean )
	est_trend_LReg_t_list4.append( time_pt )

estLinePolyData_LReg4 = vtk.vtkPolyData()
estLinePts_LReg4 = vtk.vtkPoints()

for i in range( len( est_trend_LReg_pt_list4 ) ):
	estLinePts_LReg4.InsertNextPoint( est_trend_LReg_pt_list4[ i ].pt[0], est_trend_LReg_pt_list4[ i ].pt[1], est_trend_LReg_pt_list4[ i ].pt[2] )

estLinePolyData_LReg4.SetPoints( estLinePts_LReg4 )
estLines_LReg4 = vtk.vtkCellArray()

for i in range( len( est_trend_LReg_pt_list4 ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	estLines_LReg4.InsertNextCell( line_i )

estLinePolyData_LReg4.SetLines( estLines_LReg4 )

estLineMapper_LReg4 = vtk.vtkPolyDataMapper()
estLineMapper_LReg4.SetInputData( estLinePolyData_LReg4 )

estLineActor_LReg4 = vtk.vtkActor()
estLineActor_LReg4.SetMapper( estLineMapper_LReg4 )
estLineActor_LReg4.GetProperty().SetColor( 0.980, 0.447,0.408 )
estLineActor_LReg4.GetProperty().SetOpacity( 0.7 )
estLineActor_LReg4.GetProperty().SetLineWidth( 30 )
estLineActor_LReg4.GetProperty().SetRenderLinesAsTubes( 1 ) 


# Renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

# Iteration 0 
ren.AddActor( conActor )
ren.AddActor( ptsActor )
ren.AddActor( lineActor )
# ren.AddActor( estLineActor )
ren.AddActor( estLineActor_LReg1 )
ren.AddActor( ptInitActor )

light = vtk.vtkLight() 
light.SetFocalPoint(1.875,0.6125,0)
light.SetPosition(0.875,1.6125,1)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

renWin.Render()
renIn.Start()

# Iteration 2
ren.RemoveActor( ptInitActor )
ren.RemoveActor( estLineActor_LReg1 )
ren.AddActor( estLineActor_LReg2 )
ren.AddActor( ptActor2 )

renWin.Render()
renIn.Start()

# # Iteration 3
# ren.RemoveActor( estLineActor_LReg2 )
# ren.AddActor( estLineActor_LReg3 )

# renWin.Render()
# renIn.Start()

# # Iteration 4
# ren.RemoveActor( estLineActor_LReg3 )
# ren.AddActor( estLineActor_LReg4 )

# renWin.Render()
# renIn.Start()


# Final Estimation
ren.RemoveActor( estLineActor_LReg2 )
ren.RemoveActor( ptActor2 ) 
ren.AddActor( estLineActor_LReg )
# ren.AddActor( estLineActor )
ren.AddActor( ptFinalActor )

renWin.Render()
renIn.Start()


print( "GReg time ")
print( end_greg - start_greg )

print( "EGReg time" )
print( end_egreg - start_egreg )


