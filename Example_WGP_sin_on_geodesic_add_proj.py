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
nData = 500
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

# Added sin on the first axis
pt_list_add_sin = []

# Projected sin on the base point function at time t
pt_list_proj_sin = []

## Point List - Sin Function: Parallel Transported


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

	# Add sin
	v_t_add_sin = manifolds.sphere_tVec( nDimManifold )

	v_t_add_sin.tVector[ 0 ] = np.sin( time_pt * np.pi * 3 ) / 2.0
	v_t_add_sin.tVector[ 1 ] = v_t.tVector[ 1 ]
	v_t_add_sin.tVector[ 2 ] = v_t.tVector[ 2 ]

	pt_add_sin = p_interp.ExponentialMap( v_t_add_sin )
	pt_list_add_sin.append( pt_add_sin )

	# Project sin
	v_t_sin = manifolds.sphere_tVec( nDimManifold )
	v_t_sin.tVector[ 0 ] = np.sin( time_pt * np.pi * 3 ) / 2.0 
	v_t_sin.tVector[ 1 ] = 0
	v_t_sin.tVector[ 2 ] = 0

	v_t_proj_sin = mean.ProjectTangent( mean, v_t_sin )
	pt_proj_sin = mean.ExponentialMap( v_t_proj_sin )

	pt_list_proj_sin.append( pt_proj_sin )

	t_list.append( time_pt )


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


# Visualize additive sin function - Red : pt_list_add_sin
points_add_sin = vtk.vtkPoints()

for i in range( len( pt_list_add_sin ) ):
	points_add_sin.InsertNextPoint( pt_list_add_sin[ i ].pt[0], pt_list_add_sin[ i ].pt[1], pt_list_add_sin[ i ].pt[2] )

ptsPolyData_add_sin = vtk.vtkPolyData()
ptsPolyData_add_sin.SetPoints( points_add_sin )

vertFilter_add_sin = vtk.vtkVertexGlyphFilter()
vertFilter_add_sin.SetInputData( ptsPolyData_add_sin )
vertFilter_add_sin.Update()

ptsMapper_add_sin = vtk.vtkPolyDataMapper()
ptsMapper_add_sin.SetInputData( vertFilter_add_sin.GetOutput() )

ptsActor_add_sin = vtk.vtkActor()
ptsActor_add_sin.SetMapper( ptsMapper_add_sin )
ptsActor_add_sin.GetProperty().SetPointSize( 15 )
ptsActor_add_sin.GetProperty().SetColor( 1, 0, 0 )
ptsActor_add_sin.GetProperty().SetOpacity( 1.0 )
ptsActor_add_sin.GetProperty().SetRenderPointsAsSpheres( 1 ) 

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
ren.AddActor( ptsActor_add_sin )
ren.AddActor( ptsActor_proj_sin )

renWin.Render()
renIn.Start()
