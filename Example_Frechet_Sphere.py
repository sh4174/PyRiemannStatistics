# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds 
import numpy as np
import StatsModel as sm

# Visualization
import vtk

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
t1 = 2.0

# Generate a random point on the manifold
nData = 500
dim = nDimManifold
sigma = 0.1

pt_list = []
t_list = []

for n in range( nData ):
	time_pt = np.random.uniform( t0, t1 )
	# time_pt = ( t1 - t0 ) * n / nData + t0

	# if time_pt > 0.1 and time_pt < 1.5:
	# 	n = n - 1 
	# 	continue

	# if time_pt <= 0.1:
	# 	sigma = 0.1
	# elif time_pt >= 1.5:
	# 	sigma = 0.2 	

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
# Frechet Mean #
#######################

# Gradient Descent Parameters
step_size = 0.01
max_iter = 500
step_tol = 1e-8

mu = sm.FrechetMean( pt_list, max_iter, step_tol )

#######################
# AGReg #
#######################

# Gradient Descent Parameters
step_size = 0.01
max_iter = 600
step_tol = 1e-8

base_lReg, tangent_lReg = sm.LinearizedGeodesicRegression( t_list, pt_list, max_iter, step_size, step_tol, False, True )

#######################
# GReg #
#######################

step_size = 0.01
max_iter = 500
step_tol = 1e-8

base_GReg, tangent_GReg = sm.GeodesicRegression( t_list, pt_list, max_iter, step_size, step_tol, False )


# Linearized Regression Results
print( "=================================================" )
print( "    Resulsts : Linearized Geodesic Regression    " )
print( "=================================================" )

print( "True P" )
print( p_interp.pt )
print( "True V" )
print( v_slope.tVector )

print( "Estimated P" )
print( base_lReg.pt )
print( "Estimated V" ) 
print( tangent_lReg.tVector )

# Geodesic Regression Results
print( "=================================================" )
print( "    Resulsts : Geodesic Regression    " )
print( "=================================================" )

print( "Estimated P" )
print( base_GReg.pt )
print( "Estimated V" ) 
print( tangent_GReg.tVector )

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

# Visualize spherical points
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
ptsActor.GetProperty().SetColor( 1, 0, 1 )
ptsActor.GetProperty().SetOpacity( 1.0 )
ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 ) 


# Visualize Frechet Mean
points_mu = vtk.vtkPoints()

points_mu.InsertNextPoint( mu.pt[0], mu.pt[1], mu.pt[2] )

ptsPolyData_mu = vtk.vtkPolyData()
ptsPolyData_mu.SetPoints( points_mu )

vertFilter_mu = vtk.vtkVertexGlyphFilter()
vertFilter_mu.SetInputData( ptsPolyData_mu )
vertFilter_mu.Update()

ptsMapper_mu = vtk.vtkPolyDataMapper()
ptsMapper_mu.SetInputData( vertFilter_mu.GetOutput() )

ptsActor_mu = vtk.vtkActor()
ptsActor_mu.SetMapper( ptsMapper_mu )
ptsActor_mu.GetProperty().SetPointSize( 15 )
ptsActor_mu.GetProperty().SetColor( 0, 1, 1 )
ptsActor_mu.GetProperty().SetOpacity( 1.0 )
ptsActor_mu.GetProperty().SetRenderPointsAsSpheres( 1 ) 

nTimePt = 100
# Visualize an estimated trend of Linearized Geodesic Regression Model - Red
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
estLineActor_LReg.GetProperty().SetColor( 1, 0, 0 )
estLineActor_LReg.GetProperty().SetOpacity( 0.5 )
estLineActor_LReg.GetProperty().SetLineWidth( 20 )
estLineActor_LReg.GetProperty().SetRenderLinesAsTubes( 1 ) 


# Visualize an estimated trend - Geodesic Regression : Green
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
estLineActor.GetProperty().SetColor( 0, 1, 0 )
estLineActor.GetProperty().SetOpacity( 0.5 )
estLineActor.GetProperty().SetLineWidth( 12 )
estLineActor.GetProperty().SetRenderLinesAsTubes( 0 ) 

# Renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )
ren.AddActor( ptsActor )
ren.AddActor( ptsActor_mu )
# ren.AddActor( estLineActor_LReg )
ren.AddActor( estLineActor )


light = vtk.vtkLight() 
light.SetFocalPoint(1.875,0.6125,0)
light.SetPosition(0.875,1.6125,1)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

renWin.Render()
renIn.Start()
