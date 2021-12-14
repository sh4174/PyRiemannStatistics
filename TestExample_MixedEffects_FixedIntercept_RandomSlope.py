# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds 
import numpy as np
import StatsModel as sm

# Visualization
import vtk

# Parameters
nDimManifold = 3

# Mean Interpolation Point
p_interp = manifolds.sphere( nDimManifold )
p_interp.SetPoint( [ 0.0, 0.0, 1.0 ] )

# Mean Slope Tangent Vector
v_slope = manifolds.sphere_tVec( nDimManifold )
v_slope.SetTangentVector( [ 0, np.pi * 0.25, 0 ] )

# Mean Interpolation Point / Mean Slope Tangent Vector - VTK PolyData Line
nTimePt = 100
t1 = 1.0
t0 = 0

meanLinePointsArr = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ v_slope.tVector[0] * time_pt, v_slope.tVector[1] * time_pt, v_slope.tVector[2] * time_pt ]  )
	mean = p_interp.ExponentialMap( v_t )

	meanLinePointsArr.append( mean )

meanLinePolyData = vtk.vtkPolyData()
meanLinePts = vtk.vtkPoints()

for i in range( len( meanLinePointsArr ) ):
	meanLinePts.InsertNextPoint( meanLinePointsArr[ i ].pt[0], meanLinePointsArr[ i ].pt[1], meanLinePointsArr[ i ].pt[2] )

meanLinePolyData.SetPoints( meanLinePts )
meanLineCellLines = vtk.vtkCellArray()

for i in range( len( meanLinePointsArr ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	meanLineCellLines.InsertNextCell( line_i )

meanLinePolyData.SetLines( meanLineCellLines )

# Perturb slopes
nSet = 100
sigma = 0.2

pert_pt_list = [] 
pert_v_slope_list = []

for n in range( nSet ):
	# Generate a random Gaussians with polar Box-Muller Method
	rand_pt = np.zeros( nDimManifold ).tolist()

	for i in range( nDimManifold ):
		r2 = 0
		x = 0
		y = 0

		while( r2 > 1.0 or r2 == 0 ):
			x = ( 2.0 * np.random.rand() - 1.0 )
			y = ( 2.0 * np.random.rand() - 1.0 )

			r2 = x * x + y * y

		gen_rand_no = sigma * y * np.sqrt( -2.0 * np.log( r2 ) / r2 )

		rand_pt[ i ] = gen_rand_no

	# Set Random Vector to Tangent Vector - ListToTangent
	v_slope_pert = manifolds.sphere_tVec( nDimManifold )
	v_slope_pert.SetTangentVector( [ v_slope.tVector[ 0 ] + rand_pt[ 0 ], v_slope.tVector[ 1 ] + rand_pt[ 1 ], v_slope.tVector[ 2 ] + rand_pt[ 2 ] ] )

	pert_v_slope_list.append( v_slope_pert )

	# Append Interpolation Point
	pert_pt_list.append( p_interp )


# Perturbed Interpolation Point / Parallel Slope Tangent Vector - VTK PolyData Line
pertLinePolyData_arr = []

for s in range( nSet ):
	pertLinePointsArr = []

	for n in range( nTimePt ):
		time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

		# Generate a random Gaussians with polar Box-Muller Method
		v_t = manifolds.sphere_tVec( nDimManifold ) 
		v_t.SetTangentVector( [ pert_v_slope_list[ s ].tVector[0] * time_pt, pert_v_slope_list[ s ].tVector[1] * time_pt, pert_v_slope_list[ s ].tVector[2] * time_pt ]  )
		mean = pert_pt_list[ s ].ExponentialMap( v_t )

		pertLinePointsArr.append( mean )

	pertLinePolyData = vtk.vtkPolyData()
	pertLinePts = vtk.vtkPoints()

	for i in range( len( pertLinePointsArr ) ):
		pertLinePts.InsertNextPoint( pertLinePointsArr[ i ].pt[0], pertLinePointsArr[ i ].pt[1], pertLinePointsArr[ i ].pt[2] )

	pertLinePolyData.SetPoints( pertLinePts )
	pertLineCellLines = vtk.vtkCellArray()

	for i in range( len( pertLinePointsArr ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, i )
		line_i.GetPointIds().SetId( 1, i + 1 )
		pertLineCellLines.InsertNextCell( line_i )

	pertLinePolyData.SetLines( meanLineCellLines )

	pertLinePolyData_arr.append( pertLinePolyData )	


# Frechet Mean 
pert_mean_pt = sm.FrechetMean( pert_pt_list )
mean_v_slope_pert_ParTr_Arr = [ 0, 0, 0 ]

for i in range( nSet ):
	pert_pt_i = pert_pt_list[ i ]

	v_slope_pert_i = pert_v_slope_list[ i ]

	v_slope_pert_i_ParTr = pert_pt_i.ParallelTranslateAtoB( pert_pt_i, pert_mean_pt, v_slope_pert_i )

	mean_v_slope_pert_ParTr_Arr[ 0 ] += ( float( v_slope_pert_i_ParTr.tVector[ 0 ] ) / float( nSet ) )
	mean_v_slope_pert_ParTr_Arr[ 1 ] += ( float( v_slope_pert_i_ParTr.tVector[ 1 ] ) / float( nSet ) )
	mean_v_slope_pert_ParTr_Arr[ 2 ] += ( float( v_slope_pert_i_ParTr.tVector[ 2 ] ) / float( nSet ) )

mean_v_slope_pert = manifolds.sphere_tVec( nDimManifold )
mean_v_slope_pert.SetTangentVector( mean_v_slope_pert_ParTr_Arr )


# Mean Interpolation Point / Mean Slope Tangent Vector - VTK PolyData Line
pert_meanLinePointsArr = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ mean_v_slope_pert.tVector[0] * time_pt, mean_v_slope_pert.tVector[1] * time_pt, mean_v_slope_pert.tVector[2] * time_pt ]  )
	mean = pert_mean_pt.ExponentialMap( v_t )

	pert_meanLinePointsArr.append( mean )

pert_meanLinePolyData = vtk.vtkPolyData()
pert_meanLinePts = vtk.vtkPoints()

for i in range( len( pert_meanLinePointsArr ) ):
	pert_meanLinePts.InsertNextPoint( pert_meanLinePointsArr[ i ].pt[0], pert_meanLinePointsArr[ i ].pt[1], pert_meanLinePointsArr[ i ].pt[2] )

pert_meanLinePolyData.SetPoints( pert_meanLinePts )
pert_meanLineCellLines = vtk.vtkCellArray()

for i in range( len( pert_meanLinePointsArr ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	pert_meanLineCellLines.InsertNextCell( line_i )

pert_meanLinePolyData.SetLines( pert_meanLineCellLines )

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

# Visualize a mean interpolation point and a slope
lineMapper_Mean = vtk.vtkPolyDataMapper()
lineMapper_Mean.SetInputData( meanLinePolyData )

lineActor_Mean = vtk.vtkActor()
lineActor_Mean.SetMapper( lineMapper_Mean )
lineActor_Mean.GetProperty().SetColor( 1, 0, 0 )
lineActor_Mean.GetProperty().SetOpacity( 0.5 )
lineActor_Mean.GetProperty().SetLineWidth( 20 )
lineActor_Mean.GetProperty().SetRenderLinesAsTubes( 1 ) 

# Visualize a pert_mean interpolation point and a slope
lineMapper_PertMean = vtk.vtkPolyDataMapper()
lineMapper_PertMean.SetInputData( pert_meanLinePolyData )

lineActor_PertMean = vtk.vtkActor()
lineActor_PertMean.SetMapper( lineMapper_PertMean )
lineActor_PertMean.GetProperty().SetColor( 0, 1, 0 )
lineActor_PertMean.GetProperty().SetOpacity( 0.8 )
lineActor_PertMean.GetProperty().SetLineWidth( 15 )
lineActor_PertMean.GetProperty().SetRenderLinesAsTubes( 1 ) 

# Renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )

for s in range( nSet ):

	# Visualize a mean interpolation point and a slope
	pert_lineMapper_Mean = vtk.vtkPolyDataMapper()
	pert_lineMapper_Mean.SetInputData( pertLinePolyData_arr[ s ] )

	pert_lineActor_Mean = vtk.vtkActor()
	pert_lineActor_Mean.SetMapper( pert_lineMapper_Mean )
	pert_lineActor_Mean.GetProperty().SetColor( 0, 0, 1 )
	pert_lineActor_Mean.GetProperty().SetOpacity( 0.5 )
	pert_lineActor_Mean.GetProperty().SetLineWidth( 15 )
	pert_lineActor_Mean.GetProperty().SetRenderLinesAsTubes( 1 ) 

	ren.AddActor( pert_lineActor_Mean )

ren.AddActor( lineActor_Mean )
ren.AddActor( lineActor_PertMean )

light = vtk.vtkLight() 
light.SetFocalPoint(1.875,0.6125,0)
light.SetPosition(0.875,1.6125,1)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

renWin.Render()
renIn.Start()
