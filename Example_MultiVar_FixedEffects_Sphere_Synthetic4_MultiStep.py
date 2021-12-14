################################################################################################
################################################################################################
# Synthetic Example - Multivariate Fixed Effects Model                                         #
# Manifold - Sphere Manifold 																   #
# Ind. Variables - age, sex 																   #	
# Model - y = beta_0 + beta_1 s + beta_2 t + beta_3 st 													   #	
# Model explanation - sex affects an intercept point only									   #
# 					  a shape changes can be differeny by sex,   #
#                     and initial baseline shapes of male and female are different			   #
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

nData_1 = 300
nData_2 = 300
nData = nData_1 + nData_2

# Data Noise Parameter
pt_sigma = 0.05

# Ground Truth

# Base Intercept Point
beta0 = manifolds.sphere( nDimManifold )
beta0.SetPoint( [ 1, 0, 0 ] )

## Time
t0 = 0.0
t1 = 60.0

# A tangent vector for different intercepts for different sex
beta1 = manifolds.sphere_tVec( nDimManifold )
beta1.SetTangentVector( [ 0.0, 0.5 , 0.0 ] )

# A slope tangent vector for age
beta2 = manifolds.sphere_tVec( nDimManifold )
beta2.SetTangentVector( [ 0.0, 0.0, 0.01 ] )

# A slope tangent vector for different sex
beta3 = manifolds.sphere_tVec( nDimManifold )
beta3.SetTangentVector( [ 0.0, 0.0, 0 ] )


# # Visualize group level grount truth
# Curve Visualization Parameter 
nLineTimePt = 100
# paramCurve_t0 = 0.0
# paramCurve_t1 = 1.0

group_vtk_list = []
# Group 1 - s = 0
group_geodesic_pt_list = []

for t in range( nLineTimePt ):
	time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

	v_t = manifolds.sphere_tVec( nDimManifold )
	v_t.SetTangentVector( [ beta2.tVector[ 0 ] * time_pt, beta2.tVector[ 1 ] * time_pt, beta2.tVector[ 2 ] * time_pt ] )

	p_t = beta0.ExponentialMap( v_t )
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

group_vtk_list.append( group_geodesic_vtk )

# Group 2 - s = 1
group_geodesic_pt_list2 = []

p_1 = beta0.ExponentialMap( beta1 )

beta_2_tilde = beta0.ParallelTranslateAtoB( beta0, p_1, beta2 )
beta_3_tilde = beta0.ParallelTranslateAtoB( beta0, p_1, beta3 )

for t in range( nLineTimePt ):
	time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

	v_t = manifolds.sphere_tVec( nDimManifold )
	v_t.SetTangentVector( [ ( beta_2_tilde.tVector[ 0 ] + beta_3_tilde.tVector[ 0 ] ) * time_pt, ( beta_2_tilde.tVector[ 1 ] + beta_3_tilde.tVector[ 1 ] ) * time_pt, ( beta_2_tilde.tVector[ 2 ] + beta_3_tilde.tVector[ 2 ] ) * time_pt ] )

	p_t = p_1.ExponentialMap( v_t )
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

group_vtk_list.append( group_geodesic_vtk2 )


# Group 1 - Synthetic Data Point Generation with Generalized Gaussian Noise
group1_pt_list = []
group1_t_list = []
group1_cov_int_list = [ 0 ]
group1_cov_slope_list = [ 0 ]

for i in range( nData_1 ):
	t_i = np.random.uniform( t0, t1 )

	v_t_i = beta2.ScalarMultiply( t_i )

	p_i_mean = beta0.ExponentialMap( v_t_i )

	p_i_pert = sm.GaussianNoisePerturbation( p_i_mean, pt_sigma )

	# cov_i = []
	# cov_i.append( 0 )

	# cov_slope_i = []
	# cov_slope_i.append( cov_i[ 0 ] )

	group1_pt_list.append( p_i_pert )
	group1_t_list.append( t_i )
	# group1_cov_int_list.append( cov_i )
	# group1_cov_slope_list.append( cov_slope_i )

# Group 2 - Synthetic Data Point Generation with Generalized Gaussian Noise
group2_pt_list = []
group2_t_list = []
group2_cov_int_list = [ 1 ]
group2_cov_slope_list = [ 1 ]

for i in range( nData_1 ):
	t_i = np.random.uniform( t0, t1 )

	v_t_i = manifolds.sphere_tVec( nDimManifold )
	v_t_i.SetTangentVector( [ ( beta_2_tilde.tVector[ 0 ] + beta_3_tilde.tVector[ 0 ] ) * t_i, ( beta_2_tilde.tVector[ 1 ] + beta_3_tilde.tVector[ 1 ] ) * t_i, ( beta_2_tilde.tVector[ 2 ] + beta_3_tilde.tVector[ 2 ] ) * t_i ] )

	p_i_mean = p_1.ExponentialMap( v_t_i )

	p_i_pert = sm.GaussianNoisePerturbation( p_i_mean, pt_sigma )

	# cov_i = []
	# cov_i.append( 1 )

	# cov_slope_i = []
	# cov_slope_i.append( cov_i[ 0 ] )

	group2_pt_list.append( p_i_pert )
	group2_t_list.append( t_i )
	# group2_cov_int_list.append( cov_i )
	# group2_cov_slope_list.append( cov_slope_i )	

# group_pt_list = group1_pt_list + group2_pt_list
# group_t_list = group1_t_list + group2_t_list
# group_s_list = group1_s_list + group2_s_list

# group_ind_var_list = [] 

# for i in range( len( group_pt_list ) ):
# 	group_ind_var_list.append( [ group_s_list[ i ], group_t_list[ i ] ] )

t_list = [ group1_t_list, group2_t_list ]
pt_list = [ group1_pt_list, group2_pt_list ]
cov_int_list = [ group1_cov_int_list, group2_cov_int_list ]
cov_slope_list = [ group1_cov_slope_list, group2_cov_slope_list ]

print( "cov_int_list" )
print( cov_int_list )
print( "cov_slope_list" )
print( cov_slope_list ) 

beta0, tangent_intercept_arr, tangent_slope_arr = sm.MultivariateLinearizedGeodesicRegression_Sphere_BottomUp( t_list, pt_list, cov_int_list, cov_slope_list, max_iter=5, verbose=True )

print( beta0.pt )
print( tangent_intercept_arr[0].tVector )
print( tangent_slope_arr[ 0 ].tVector )
print( tangent_slope_arr[ 1 ].tVector )

base = beta0

print( base.pt )

est_beta_1 = manifolds.sphere_tVec( nDimManifold )
est_beta_1.SetTangentVector( tangent_intercept_arr[ 0 ].tVector )

est_beta_2 = manifolds.sphere_tVec( nDimManifold )
est_beta_2.SetTangentVector( tangent_slope_arr[ 1 ].tVector )

est_beta_3 = manifolds.sphere_tVec( nDimManifold )
est_beta_3.SetTangentVector( tangent_slope_arr[ 0 ].tVector )

# Estimated Results
est_group_vtk_list = []
# Group 1 - s = 0
est_group_geodesic_pt_list = []

for t in range( nLineTimePt ):
	time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

	v_t = manifolds.sphere_tVec( nDimManifold )
	v_t.SetTangentVector( [ est_beta_2.tVector[ 0 ] * time_pt, est_beta_2.tVector[ 1 ] * time_pt, est_beta_2.tVector[ 2 ] * time_pt ] )

	p_t = base.ExponentialMap( v_t )
	est_group_geodesic_pt_list.append( p_t )

est_group_geodesic_vtk = vtk.vtkPolyData()
est_group_geodesic_pts = vtk.vtkPoints()

for t in range( len( group_geodesic_pt_list ) ):
	est_group_geodesic_pts.InsertNextPoint( est_group_geodesic_pt_list[ t ].pt[ 0 ], est_group_geodesic_pt_list[ t ].pt[ 1 ], est_group_geodesic_pt_list[ t ].pt[ 2 ] )

est_group_geodesic_line = vtk.vtkCellArray()

for t in range( len( est_group_geodesic_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, t )
	line_i.GetPointIds().SetId( 1, t + 1 )
	est_group_geodesic_line.InsertNextCell( line_i )

est_group_geodesic_vtk.SetPoints( est_group_geodesic_pts )
est_group_geodesic_vtk.SetLines( est_group_geodesic_line )
est_group_vtk_list.append( est_group_geodesic_vtk )

# Group 2 - s = 1
est_group_geodesic_pt_list2 = []

est_p_1 = base.ExponentialMap( est_beta_1 )
est_beta_2_tilde = base.ParallelTranslateAtoB( base, est_p_1, est_beta_2  )
est_beta_3_tilde = base.ParallelTranslateAtoB( base, est_p_1, est_beta_3  )

for t in range( nLineTimePt ):
	time_pt = ( t1 - t0 ) * t / nLineTimePt + t0

	v_t2 = manifolds.sphere_tVec( nDimManifold )
	v_t2.SetTangentVector( [ ( est_beta_2_tilde.tVector[ 0 ] + est_beta_3_tilde.tVector[ 0 ] ) * time_pt, ( est_beta_2_tilde.tVector[ 1 ] + est_beta_3_tilde.tVector[ 1 ] ) * time_pt, ( est_beta_2_tilde.tVector[ 2 ] + est_beta_3_tilde.tVector[ 2 ] ) * time_pt ] )

	p_t2 = est_p_1.ExponentialMap( v_t2 )
	est_group_geodesic_pt_list2.append( p_t2 )

est_group_geodesic_vtk2 = vtk.vtkPolyData()
est_group_geodesic_pts2 = vtk.vtkPoints()

for t in range( len( est_group_geodesic_pt_list2 ) ):
	est_group_geodesic_pts2.InsertNextPoint( est_group_geodesic_pt_list2[ t ].pt[ 0 ], est_group_geodesic_pt_list2[ t ].pt[ 1 ], est_group_geodesic_pt_list2[ t ].pt[ 2 ] )

est_group_geodesic_line2 = vtk.vtkCellArray()
for t in range( len( est_group_geodesic_pt_list2 ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, t )
	line_i.GetPointIds().SetId( 1, t + 1 )
	est_group_geodesic_line2.InsertNextCell( line_i )

est_group_geodesic_vtk2.SetPoints( est_group_geodesic_pts2 )
est_group_geodesic_vtk2.SetLines( est_group_geodesic_line2 )

est_group_vtk_list.append( est_group_geodesic_vtk2 )


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

points = vtk.vtkPoints()

for i in range( len( group1_pt_list ) ):
	points.InsertNextPoint( group1_pt_list[ i ].pt[0], group1_pt_list[ i ].pt[1], group1_pt_list[ i ].pt[2] )

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
ptsActor.GetProperty().SetColor( group_gt_color[ 0 ] )
ptsActor.GetProperty().SetOpacity( 1.0 )
ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 )

# Visualize Data points - Group 2
points1 = vtk.vtkPoints()

for i in range( len( group2_pt_list ) ):
	points1.InsertNextPoint( group2_pt_list[ i ].pt[0], group2_pt_list[ i ].pt[1], group2_pt_list[ i ].pt[2] )

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
ptsActor1.GetProperty().SetColor( group_gt_color[ 1 ] )
ptsActor1.GetProperty().SetOpacity( 1.0 )
ptsActor1.GetProperty().SetRenderPointsAsSpheres( 1 )

# Renderer1
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )

# Visualize ground trurh group level geodesics
for i in range( len( group_vtk_list ) ):
	group_geodesic_mapper = vtk.vtkPolyDataMapper()
	group_geodesic_mapper.SetInputData( group_vtk_list[ i ] )

	group_geodesic_actor = vtk.vtkActor()
	group_geodesic_actor.SetMapper( group_geodesic_mapper )
	group_geodesic_actor.GetProperty().SetColor( group_gt_color[ i ] )
	group_geodesic_actor.GetProperty().SetOpacity( 0.7 )
	group_geodesic_actor.GetProperty().SetLineWidth( 15 )
	group_geodesic_actor.GetProperty().SetRenderLinesAsTubes( 1 )

	ren.AddActor( group_geodesic_actor )

# Visualized estimated geodesic - Group 1 
estGroupGeodesicMapper = vtk.vtkPolyDataMapper()
estGroupGeodesicMapper.SetInputData( est_group_vtk_list[ 0 ] )

estGroupGeodesicActor = vtk.vtkActor()
estGroupGeodesicActor.SetMapper( estGroupGeodesicMapper )
estGroupGeodesicActor.GetProperty().SetLineWidth( 8 )
estGroupGeodesicActor.GetProperty().SetColor( group_est_color[ 0 ] )
estGroupGeodesicActor.GetProperty().SetOpacity( 1.0 )
estGroupGeodesicActor.GetProperty().SetRenderLinesAsTubes( 1 )

ren.AddActor( estGroupGeodesicActor )

# Visualized estimated geodesic - Group 2 
estGroupGeodesicMapper2 = vtk.vtkPolyDataMapper()
estGroupGeodesicMapper2.SetInputData( est_group_vtk_list[ 1 ] )

estGroupGeodesicActor2 = vtk.vtkActor()
estGroupGeodesicActor2.SetMapper( estGroupGeodesicMapper2 )
estGroupGeodesicActor2.GetProperty().SetLineWidth( 8 )
estGroupGeodesicActor2.GetProperty().SetColor( group_est_color[ 1 ] )
estGroupGeodesicActor2.GetProperty().SetOpacity( 1.0 )
estGroupGeodesicActor2.GetProperty().SetRenderLinesAsTubes( 1 )

ren.AddActor( estGroupGeodesicActor2 )


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
