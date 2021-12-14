# Geodesic Regression on Sphere Manifold
# Manifolds 
import manifolds 
import numpy as np
import StatsModel as sm

import matplotlib.pyplot as plt
# Visualization
import vtk

# Parameters
nDimManifold = 3


# Group Level Parameters
## Ground Truth
p0 = manifolds.sphere( nDimManifold )

v0 = manifolds.sphere_tVec( nDimManifold )
v1 = manifolds.sphere_tVec( nDimManifold )

## Random interpolation - mean
# p0.SetPoint( [ -1.0, 0.0, 0.0 ] )

# p0 = sm.GaussianNoisePerturbation( p0, 2.0 )

# Mean Interpolation Point - Fixed for Experiment
p0.SetPoint( [ np.sqrt( 0.28 ), 0.6, 0.6  ] )

print( p0.pt )

## Time
t0 = 0.0
t1 = 60.0

# CAG repeat length range 
c0 = 15.0
c1 = 40.0

# Fixed Slope for ct 
v0.SetTangentVector( [ 0, 0.001, 0 ] )

# Fixed Slope for t 
v1.SetTangentVector( [ 0, 0.0, 0.01 ] )

# # Visualize group level grount truth
# Curve Visualization Parameter 
nLineTimePt = 100
paramCurve_t0 = 0.0
paramCurve_t1 = 1.0

nC_GroupGeodesic = 30

group_geodesic_vtk_list = []

for k in range( nC_GroupGeodesic ):
	c_k = ( c1 - c0 ) * k / nC_GroupGeodesic + c0
	print( c_k )

	group_geodesic_pt_list = []

	print( [ v0.tVector[ 0 ] * c_k  + v1.tVector[ 0 ], v0.tVector[ 1 ] * c_k + v1.tVector[ 1 ], v0.tVector[ 2 ] * c_k + v1.tVector[ 2 ] ] )

	for t in range( nLineTimePt ):
		ct_pt = c_k * ( ( t1 - t0 ) * t / nLineTimePt + t0 )

		t_pt = ( t1 - t0 ) * t / nLineTimePt + t0

		v_t = manifolds.sphere_tVec( nDimManifold )

		v_t.SetTangentVector( [ v0.tVector[ 0 ] * ct_pt + v1.tVector[ 0 ] * t_pt, v0.tVector[ 1 ] * ct_pt + v1.tVector[ 1 ] * t_pt, v0.tVector[ 2 ] * ct_pt + v1.tVector[ 2 ] * t_pt ] )
		p_t = p0.ExponentialMap( v_t )
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

	group_geodesic_vtk_list.append( group_geodesic_vtk )


# Subject Level Parameters 
## No. of subjects
nSubject = 100

## No. of observations per subject
nObj = 2 

## Subject-wise time interval - random variable over a normal distribution ~N( 3, 0.2^2 )
subj_time_interval_mu = 4
subj_time_interval_sigma = 0.5

## Subject-wise initial observation time - random variable over a uniform distribution ~U( t0, t1 )
## Subject-wise interpolation point - random variable over a normal distribution ~N( p_interp, 0.15^2 )
pt_sigma = 0.1

# Subject interpolation point list 
subj_pt0_list = []
subj_pt1_list = []

# Subject-wise all points list
subj_all_pts_list = []
subj_all_time_pts_list = []

# Subject-wise CAG repeat length c 
subj_c_list = []

# subject-wise geodesic vtkPolyData list
subj_geodesic_list = []

for i in range( nSubject ):
	# Subject-wise time interval
	subj_time_interval = np.random.normal( subj_time_interval_mu, subj_time_interval_sigma )

	# Subject-wise initial time point 
	subj_init_time = np.random.uniform( t0, t1 )

	# Subject-wise CAG repeat length 
	c_i = np.random.uniform( c0, c1 )

	print( c_i )

	subj_c_list.append( c_i )

	# # p0_i at subj_init_time without perturbation
	# v_t0_i = manifolds.sphere_tVec( nDimManifold )
	# v_t0_i.SetTangentVector( [ v0.tVector[ 0 ] * ct0_pt + v1.tVector[ 0 ] * subj_init_time, v0.tVector[ 1 ] * ct0_pt + v1.tVector[ 1 ] * subj_init_time, v0.tVector[ 2 ] * ct0_pt + v1.tVector[ 2 ] * subj_init_time ] )
	# p0_i_mean = p0.ExponentialMap( v_t0_i )

	# p0_i - Subject-wise intercept point perturbation at groupwise t0 = 0
	p0_i = sm.GaussianNoisePerturbation( p0, pt_sigma )

	# Parallel transport v0 and v1 from p0 to p0_i
	v0_i = p0.ParallelTranslateAtoB( p0, p0_i, v0 )
	v1_i = p0.ParallelTranslateAtoB( p0, p0_i, v1 )

	ct0_i = c_i * subj_init_time
	t0_i = subj_init_time

	v_i_t0_i = manifolds.sphere_tVec( nDimManifold )
	v_i_t0_i.SetTangentVector( [ v0_i.tVector[ 0 ] * ct0_i + v1_i.tVector[ 0 ] * t0_i, v0_i.tVector[ 1 ] * ct0_i + v1_i.tVector[ 1 ] * t0_i, v0_i.tVector[ 2 ] * ct0_i + v1_i.tVector[ 2 ] * t0_i ] )

	p0_i_t0 = p0_i.ExponentialMap( v_i_t0_i )

	subj_pt0_list.append( p0_i_t0 )

	# p1_i
	ct1_i = c_i * ( subj_time_interval + subj_init_time )
	t1_i = subj_time_interval + subj_init_time 
	v0_t1_i = v0_i.ScalarMultiply( ct1_i  )
	v1_t1_i = v1_i.ScalarMultiply( t1_i )

	# v0_t1_i = p0_i_mean.ParallelTranslateAtoB( p0_i_mean, p0_i, p0.ParallelTranslateAtoB( p0, p0_i_mean, v0 ) ).ScalarMultiply( c_i * subj_time_interval )
	# v1_t1_i = p0_i_mean.ParallelTranslateAtoB( p0_i_mean, p0_i, p0.ParallelTranslateAtoB( p0, p0_i_mean, v1 ) ).ScalarMultiply( subj_time_interval )

	# print( "Direct Parallel Transport from p0 to p0_i(t0_i)")
	# print( v0_t1_i.tVector )
	# print( v1_t1_i.tVector )

	# print( "Parallel Transport from p0 to p0_mean(t0_i) and then to p0_i(t0_i)")
	# print( v0_t1_i_check.tVector )
	# print( v1_t1_i_check.tVector )

	v_t1_i = manifolds.sphere_tVec( nDimManifold )
	v_t1_i.SetTangentVector( [ v0_t1_i.tVector[ 0 ] + v1_t1_i.tVector[ 0 ], v0_t1_i.tVector[ 1 ] + v1_t1_i.tVector[ 1 ], v0_t1_i.tVector[ 2 ] + v1_t1_i.tVector[ 2 ] ] )

	p1_i = p0_i.ExponentialMap( v_t1_i )
	subj_pt1_list.append( p1_i ) 

	# Draw a geodesic
	subj_geodesic_pt_list = []

	for t in range( nLineTimePt ):
		t_i_t = ( ( subj_time_interval ) * t ) / nLineTimePt + subj_init_time
		ct_i_t = t_i_t * c_i

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ v0_i.tVector[ 0 ] * ct_i_t + v1_i.tVector[ 0 ] * t_i_t, v0_i.tVector[ 1 ] * ct_i_t + v1_i.tVector[ 1 ] * t_i_t, v0_i.tVector[ 2 ] * ct_i_t + v1_i.tVector[ 2 ] * t_i_t ] )

		p_t = p0_i.ExponentialMap( v_t )
		subj_geodesic_pt_list.append( p_t )

	subj_geodesic_vtk = vtk.vtkPolyData()
	subj_geodesic_pts = vtk.vtkPoints()

	for t in range( len( subj_geodesic_pt_list ) ):
		subj_geodesic_pts.InsertNextPoint( subj_geodesic_pt_list[ t ].pt[ 0 ], subj_geodesic_pt_list[ t ].pt[ 1 ], subj_geodesic_pt_list[ t ].pt[ 2 ] )

	subj_geodesic_line = vtk.vtkCellArray()
	for t in range( len( subj_geodesic_pt_list ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		subj_geodesic_line.InsertNextCell( line_i )

	subj_geodesic_vtk.SetPoints( subj_geodesic_pts )
	subj_geodesic_vtk.SetLines( subj_geodesic_line )

	subj_geodesic_list.append( subj_geodesic_vtk )

	# Subject-wise all points list
	subj_pts_list_i = []

	subj_pts_list_i.append( p0_i )
	subj_pts_list_i.append( p1_i )

	subj_all_pts_list.append( subj_pts_list_i ) 

	subj_time_pts_list_i = []
	subj_time_pts_list_i.append( subj_init_time )
	subj_time_pts_list_i.append( subj_init_time + subj_time_interval )

	subj_all_time_pts_list.append( subj_time_pts_list_i )


###################################################
#######			 Regression Models 			#######
###################################################
# Gradient Descent Parameters
step_size = 0.01
max_iter = 500
step_tol = 1e-8

est_subj_geodesic_list = []
est_subj_p0_list = []
est_subj_v0_list = []

# Subject-wise geodesic regression model 
for i in range( nSubject ):
	c_i = subj_c_list[ i ] 	
	subj_pts_list_i = subj_all_pts_list[ i ] 
	subj_time_pts_list_i = subj_all_time_pts_list[ i ]

	subj_c_time_pts_list_i = []

	# Multiply c_i and time, c_i * t
	for j in range( len( subj_time_pts_list_i ) ):
		subj_c_time_pts_list_i.append( subj_time_pts_list_i[ j ] * c_i )

	p0_i, v0_i = sm.LinearizedGeodesicRegression( subj_c_time_pts_list_i, subj_pts_list_i, max_iter, step_size, step_tol, False )
	# p0_i, v0_i = sm.GeodesicRegression( subj_time_pts_list_i, subj_pts_list_i, max_iter, step_size, step_tol, False )

	# Subjet-wise estimation result to vtk for visualization - Time range : subj_t0 ~ subj_t1
	subj_t1 = subj_c_time_pts_list_i[ 1 ]
	subj_t0 = subj_c_time_pts_list_i[ 0 ]

	# Draw a geodesics
	est_subj_geodesic_pt_list = []

	for t in range( nLineTimePt ):
		time_pt = ( ( subj_t1 - subj_t0 ) * t ) / nLineTimePt + subj_t0

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ v0_i.tVector[ 0 ] * time_pt, v0_i.tVector[ 1 ] * time_pt, v0_i.tVector[ 2 ] * time_pt ] )

		p_t = p0_i.ExponentialMap( v_t )
		est_subj_geodesic_pt_list.append( p_t )

	est_subj_geodesic_vtk = vtk.vtkPolyData()
	est_subj_geodesic_pts = vtk.vtkPoints()

	for t in range( len( est_subj_geodesic_pt_list ) ):
		est_subj_geodesic_pts.InsertNextPoint( est_subj_geodesic_pt_list[ t ].pt[ 0 ], est_subj_geodesic_pt_list[ t ].pt[ 1 ], est_subj_geodesic_pt_list[ t ].pt[ 2 ] )

	est_subj_geodesic_line = vtk.vtkCellArray()
	for t in range( len( est_subj_geodesic_pt_list ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		est_subj_geodesic_line.InsertNextCell( line_i )

	est_subj_geodesic_vtk.SetPoints( est_subj_geodesic_pts )
	est_subj_geodesic_vtk.SetLines( est_subj_geodesic_line )

	est_subj_geodesic_list.append( est_subj_geodesic_vtk )

	est_subj_p0_list.append( p0_i )
	est_subj_v0_list.append( v0_i )


# Group level geodesic regression model - Calculation : Frechet Mean and Parallel Transport
# Calculate Frechet Mean
p0_mean = sm.FrechetMean( est_subj_p0_list )

# Parallel transport and (Euclidean) average of subject-wise tangent vectors
v0_mean = manifolds.sphere_tVec( nDimManifold )
v0_mean_tVec_arr = [ 0, 0, 0 ]	

for j in range( nSubject ):
	p0_j = est_subj_p0_list[ j ]
	v0_j = est_subj_v0_list[ j ]

	v0_j_pt = p0_mean.ParallelTranslateAtoB( p0_j, p0_mean, v0_j ) 

	v0_mean_tVec_arr[ 0 ] += ( v0_j_pt.tVector[ 0 ] / float( nSubject ) )
	v0_mean_tVec_arr[ 1 ] += ( v0_j_pt.tVector[ 1 ] / float( nSubject ) )
	v0_mean_tVec_arr[ 2 ] += ( v0_j_pt.tVector[ 2 ] / float( nSubject ) )

v0_mean.SetTangentVector( v0_mean_tVec_arr )

# Group level geodesic results to vtk 
est_group_geodesic_pt_list = []

for t in range( nLineTimePt ):
	time_pt = ( c1 * t1 - c0 * t0 ) * t / nLineTimePt + c0 * t0

	v_t = manifolds.sphere_tVec( nDimManifold )
	v_t.SetTangentVector( [ v0_mean.tVector[ 0 ] * time_pt, v0_mean.tVector[ 1 ] * time_pt, v0_mean.tVector[ 2 ] * time_pt ] )

	p_t = p0_mean.ExponentialMap( v_t )
	est_group_geodesic_pt_list.append( p_t )

est_group_geodesic_vtk = vtk.vtkPolyData()
est_group_geodesic_pts = vtk.vtkPoints()

for t in range( len( est_group_geodesic_pt_list ) ):
	est_group_geodesic_pts.InsertNextPoint( est_group_geodesic_pt_list[ t ].pt[ 0 ], est_group_geodesic_pt_list[ t ].pt[ 1 ], est_group_geodesic_pt_list[ t ].pt[ 2 ] )

est_group_geodesic_line = vtk.vtkCellArray()
for t in range( len( est_group_geodesic_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, t )
	line_i.GetPointIds().SetId( 1, t + 1 )
	est_group_geodesic_line.InsertNextCell( line_i )

est_group_geodesic_vtk.SetPoints( est_group_geodesic_pts )
est_group_geodesic_vtk.SetLines( est_group_geodesic_line )

print( "Estimated p0" ) 
print( p0_mean.pt )

print( "Estimated v0" )
print( v0_mean.tVector )

est_group_to_subj_geodesic_list = []

# Project Group level geodesic to a subject-wise dataset (random interpolation) 
for i in range( nSubject ):
	c_i = subj_c_list[ i ] 	

	subj_pts_list_i = subj_all_pts_list[ i ] 
	subj_time_pts_list_i = subj_all_time_pts_list[ i ]

	subj_c_time_pts_list_i = []

	# Multiply c_i and time, c_i * t
	for j in range( len( subj_time_pts_list_i ) ):
		subj_c_time_pts_list_i.append( subj_time_pts_list_i[ j ] * c_i )


	# Subjet-wise estimation result to vtk for visualization - Time range : subj_t0 ~ subj_t1
	subj_t1 = subj_c_time_pts_list_i[ 1 ]
	subj_t0 = subj_c_time_pts_list_i[ 0 ]

	subj_time_interval_i = subj_t1 - subj_t0

	# Estimated interpolation point
	p0_i = est_subj_p0_list[ i ]
	v0_i_ind_est = est_subj_v0_list[ i ] 

	p0_t0 = p0_i.ExponentialMap( v0_i_ind_est.ScalarMultiply( subj_t0 ) )

	# Parallel Transport v0_mean to the subject-wise interpolation point
	v0_i = p0_mean.ParallelTranslateAtoB( p0_mean, p0_t0, v0_mean ).ScalarMultiply( subj_time_interval_i )

	# Draw a geodesics
	est_group_to_subj_geodesic_pt_list = []

	for t in range( nLineTimePt ):
		time_pt = ( ( paramCurve_t1 - paramCurve_t0 ) * t ) / nLineTimePt

		v_t = manifolds.sphere_tVec( nDimManifold )
		v_t.SetTangentVector( [ v0_i.tVector[ 0 ] * time_pt, v0_i.tVector[ 1 ] * time_pt, v0_i.tVector[ 2 ] * time_pt ] )

		p_t = p0_t0.ExponentialMap( v_t )
		est_group_to_subj_geodesic_pt_list.append( p_t )

	est_group_to_subj_geodesic_vtk = vtk.vtkPolyData()
	est_group_to_subj_geodesic_pts = vtk.vtkPoints()

	for t in range( len( est_group_to_subj_geodesic_pt_list ) ):
		est_group_to_subj_geodesic_pts.InsertNextPoint( est_group_to_subj_geodesic_pt_list[ t ].pt[ 0 ], est_group_to_subj_geodesic_pt_list[ t ].pt[ 1 ], est_group_to_subj_geodesic_pt_list[ t ].pt[ 2 ] )

	est_group_to_subj_geodesic_line = vtk.vtkCellArray()
	for t in range( len( est_group_to_subj_geodesic_pt_list ) - 1 ):
		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, t )
		line_i.GetPointIds().SetId( 1, t + 1 )
		est_group_to_subj_geodesic_line.InsertNextCell( line_i )

	est_group_to_subj_geodesic_vtk.SetPoints( est_group_to_subj_geodesic_pts )
	est_group_to_subj_geodesic_vtk.SetLines( est_group_to_subj_geodesic_line )

	est_group_to_subj_geodesic_list.append( est_group_to_subj_geodesic_vtk )



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

# Visualize Data points
points = vtk.vtkPoints()

for i in range( len( subj_pt0_list ) ):
	points.InsertNextPoint( subj_pt0_list[ i ].pt[0], subj_pt0_list[ i ].pt[1], subj_pt0_list[ i ].pt[2] )

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

# Visualize Data points
points1 = vtk.vtkPoints()

for i in range( len( subj_pt1_list ) ):
	points1.InsertNextPoint( subj_pt1_list[ i ].pt[0], subj_pt1_list[ i ].pt[1], subj_pt1_list[ i ].pt[2] )

ptsPolyData1 = vtk.vtkPolyData()
ptsPolyData1.SetPoints( points1 )

vertFilter1 = vtk.vtkVertexGlyphFilter()
vertFilter1.SetInputData( ptsPolyData1 )
vertFilter1.Update()

ptsMapper1 = vtk.vtkPolyDataMapper()
ptsMapper1.SetInputData( vertFilter1.GetOutput() )

ptsActor1 = vtk.vtkActor()
ptsActor1.SetMapper( ptsMapper1 )
ptsActor1.GetProperty().SetPointSize( 8 )
ptsActor1.GetProperty().SetColor( 1, 0, 0 )
ptsActor1.GetProperty().SetOpacity( 1.0 )
ptsActor1.GetProperty().SetRenderPointsAsSpheres( 1 )

# # Visualize Intrinsic Mean Point
# meanPolyData1 = vtk.vtkPolyData()
# meanPt1 = vtk.vtkPoints()

# meanPt1.InsertNextPoint( mu.pt[0], mu.pt[1], mu.pt[2] )
# meanPt1.Modified()

# meanPolyData1.SetPoints( meanPt1 )
# meanPolyData1.Modified()

# meanVertFilter1 = vtk.vtkVertexGlyphFilter()
# meanVertFilter1.SetInputData( meanPolyData1 )
# meanVertFilter1.Update()


# meanMapper1 = vtk.vtkPolyDataMapper()
# meanMapper1.SetInputData( meanVertFilter1.GetOutput() )

# meanActor1 = vtk.vtkActor()
# meanActor1.SetMapper( meanMapper1 )
# meanActor1.GetProperty().SetColor( 0, 0.5, 1 )
# meanActor1.GetProperty().SetOpacity( 0.6 )
# meanActor1.GetProperty().SetPointSize( 15 )
# meanActor1.GetProperty().SetRenderPointsAsSpheres(1 )

# Renderer1
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )

# Visualize Group level geodesic

for k in range( nC_GroupGeodesic ):
	group_geodesic_mapper = vtk.vtkPolyDataMapper()
	group_geodesic_mapper.SetInputData( group_geodesic_vtk_list[ k ] )

	group_geodesic_actor = vtk.vtkActor()
	group_geodesic_actor.SetMapper( group_geodesic_mapper )
	group_geodesic_actor.GetProperty().SetColor( 0, 0, 1 )
	group_geodesic_actor.GetProperty().SetOpacity( 0.3 )
	group_geodesic_actor.GetProperty().SetLineWidth( 8 )
	group_geodesic_actor.GetProperty().SetRenderLinesAsTubes( 1 )

	ren.AddActor( group_geodesic_actor )

# Visualize subject-wise geodesics 
for i in range( len( subj_geodesic_list ) ):
	subj_geodesic_mapper_i = vtk.vtkPolyDataMapper()
	subj_geodesic_mapper_i.SetInputData( subj_geodesic_list[ i ] )

	subj_geodesic_actor_i = vtk.vtkActor()
	subj_geodesic_actor_i.SetMapper( subj_geodesic_mapper_i )
	subj_geodesic_actor_i.GetProperty().SetColor( 1.0, 0, 1.0 )
	subj_geodesic_actor_i.GetProperty().SetOpacity( 0.4 )
	subj_geodesic_actor_i.GetProperty().SetLineWidth( 15 )
	subj_geodesic_actor_i.GetProperty().SetRenderLinesAsTubes( 1 )

	ren.AddActor( subj_geodesic_actor_i )

	est_subj_geodesic_mapper_i = vtk.vtkPolyDataMapper()
	est_subj_geodesic_mapper_i.SetInputData( est_subj_geodesic_list[ i ] )

	est_subj_geodesic_actor_i = vtk.vtkActor()
	est_subj_geodesic_actor_i.SetMapper( est_subj_geodesic_mapper_i )
	est_subj_geodesic_actor_i.GetProperty().SetColor( 0.0, 1.0, 0.0 )
	est_subj_geodesic_actor_i.GetProperty().SetOpacity( 0.4 )
	est_subj_geodesic_actor_i.GetProperty().SetLineWidth( 10 )
	est_subj_geodesic_actor_i.GetProperty().SetRenderLinesAsTubes( 1 )

	ren.AddActor( est_subj_geodesic_actor_i )

# Visualize estimated group-wise tangent vector to a subject-wise trajectory
for i in range( len( est_group_to_subj_geodesic_list ) ):
	subj_group_geodesic_mapper_i = vtk.vtkPolyDataMapper()
	subj_group_geodesic_mapper_i.SetInputData( est_group_to_subj_geodesic_list[ i ] )

	subj_group_geodesic_actor_i = vtk.vtkActor()
	subj_group_geodesic_actor_i.SetMapper( subj_group_geodesic_mapper_i )
	subj_group_geodesic_actor_i.GetProperty().SetColor( 1.0, 1.0, 1.0 )
	subj_group_geodesic_actor_i.GetProperty().SetOpacity( 1.0 )
	subj_group_geodesic_actor_i.GetProperty().SetLineWidth( 5 )
	subj_group_geodesic_actor_i.GetProperty().SetRenderLinesAsTubes( 1 )

	ren.AddActor( subj_group_geodesic_actor_i )


# Visualized Group level estimated geodesic
estGroupGeodesicMapper = vtk.vtkPolyDataMapper()
estGroupGeodesicMapper.SetInputData( est_group_geodesic_vtk )

estGroupGeodesicActor = vtk.vtkActor()
estGroupGeodesicActor.SetMapper( estGroupGeodesicMapper )
estGroupGeodesicActor.GetProperty().SetLineWidth( 8 )
estGroupGeodesicActor.GetProperty().SetColor( 1, 1, 0 )
estGroupGeodesicActor.GetProperty().SetOpacity( 1.0 )
estGroupGeodesicActor.GetProperty().SetRenderLinesAsTubes( 1 )

# ren.AddActor( estGroupGeodesicActor )

ren.AddActor( ptsActor )
ren.AddActor( ptsActor1 )

# ren.AddActor( meanActor1 )

light = vtk.vtkLight() 
light.SetFocalPoint(0,0.6125,1.875)
light.SetPosition(1,0.875,1.6125)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

renWin.Render()
renIn.Start()
