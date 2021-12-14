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

def drawArrow( start_point, end_point ):
	math = vtk.vtkMath()
	# X-Axis is a vector from start to end
	normX = np.subtract( start_point, end_point )
	length = np.linalg.norm( normX )
	normX = np.divide( normX, length )

	# Z axis is an arbitrary vector cross X
	arbitrary = [ 0.0, 0.0, 1.0 ] 
	normZ = np.cross( normX, arbitrary )

	# Y-Axis is Z cross X 
	normY = np.cross( normZ, normX )

	tVec_at_b0_mat = vtk.vtkMatrix4x4()
	tVec_at_b0_mat.Identity()

	for i in range( 3 ):
		tVec_at_b0_mat.SetElement( i, 0, normX[ i ] )
		tVec_at_b0_mat.SetElement( i, 1, normY[ i ] )
		tVec_at_b0_mat.SetElement( i, 2, normZ[ i ] )

	arrowBase = vtk.vtkArrowSource()
	arrowBase.Update()

	print( arrowBase.GetTipLength() )
	print( arrowBase.GetTipRadius() )
	print( arrowBase.GetTipResolution() )

	print( arrowBase.GetShaftRadius() )
	print( arrowBase.GetShaftResolution() )

	arrow_transform = vtk.vtkTransform()
	arrow_transform.Translate( start_point )
	arrow_transform.Concatenate( tVec_at_b0_mat )
	arrow_transform.Scale( length, length, length )
	arrow_transform.Update()

	transformPD = vtk.vtkTransformPolyDataFilter()
	transformPD.SetInputData( arrowBase.GetOutput() )
	transformPD.SetTransform( arrow_transform )
	transformPD.Update()

	return transformPD.GetOutput()

# Base Intercept Point
beta0 = manifolds.sphere( nDimManifold )
beta0.SetPoint( [ 1, 0, 0 ] )

gamma1 = manifolds.sphere_tVec( nDimManifold )
gamma1.SetTangentVector( [ 0.0, -0.3 , 0.0 ] )

tVec_to_f_i = manifolds.sphere_tVec( nDimManifold )
tVec_to_f_i.SetTangentVector( [ 0.0, -1.0, 0.0 ] )

f_i = beta0.ExponentialMap( tVec_to_f_i )
# f_i.SetPoint( [ 0, 1, 0 ] )

gamma1_f_i = beta0.ParallelTranslateAtoB( beta0, f_i, gamma1 )

pt_sigma = 0.15

# Draw beta0 to f_i - Main Geodesic
nLineTimePt = 100
t1 = 1.0
t0 = 0.0 

beta0_f_i_geodesic_vtk = vtk.vtkPolyData()
beta0_f_i_geodesic_pts = vtk.vtkPoints()
beta0_f_i_geodesic_lines = vtk.vtkCellArray()

for i in range( nLineTimePt ):
	t_i = ( i + 1 - t0 ) / float( nLineTimePt )

	v_t = tVec_to_f_i.ScalarMultiply( t_i )

	p_t = beta0.ExponentialMap( v_t )

	beta0_f_i_geodesic_pts.InsertNextPoint( p_t.pt[ 0 ], p_t.pt[ 1 ], p_t.pt[ 2 ] )

	if i == ( nLineTimePt - 1 ):
		continue

	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	beta0_f_i_geodesic_lines.InsertNextCell( line_i )

beta0_f_i_geodesic_pts.Modified()
beta0_f_i_geodesic_lines.Modified()

beta0_f_i_geodesic_vtk.SetPoints( beta0_f_i_geodesic_pts )
beta0_f_i_geodesic_vtk.SetLines( beta0_f_i_geodesic_lines )
beta0_f_i_geodesic_vtk.Modified()

beta0_f_i_geodesic_mapper = vtk.vtkPolyDataMapper()
beta0_f_i_geodesic_mapper.SetInputData( beta0_f_i_geodesic_vtk )
beta0_f_i_geodesic_mapper.Update()

# Draw geodesics from a_i to f_i 

# Draw Arrows
# Base Tangent Vector
arrow_tVec_org = drawArrow( beta0.pt, np.add( beta0.pt, gamma1.tVector ) ) 

# Tangent Vector Arrows
stopPT_arrow_at_beta0_arr = []
directPT_arrow_at_beta0_arr = []

pts_arr = []
arrow_at_pts_arr = []
a_to_f_geodesics_arr = []
a_to_b0_geodesics_arr = []


for i in range( 10 ):
	a_i = sm.GaussianNoisePerturbation( f_i, pt_sigma )

	gamma1_a_i = beta0.ParallelTranslateAtoB( f_i, a_i, gamma1_f_i )
	gamma1_a_i_to_b0_direct = a_i.ParallelTranslateAtoB( a_i, beta0, gamma1_a_i )

	gamma1_a_i_to_f_i = a_i.ParallelTranslateAtoB( a_i, f_i, gamma1_a_i )
	gamma1_a_i_to_b0_stopover = f_i.ParallelTranslateAtoB( f_i, beta0, gamma1_a_i_to_f_i )

	arrow_tVec_at_a_i = drawArrow( a_i.pt, np.add( a_i.pt, gamma1_a_i.tVector ) )
	# arrow_tVec_at_f_i = drawArrow( f_i.pt, np.add( f_i.pt, gamma1_f_i.tVector ) )

	arrow_tVec_a_i_directP_at_beta0 = drawArrow( beta0.pt, np.add( beta0.pt, gamma1_a_i_to_b0_direct.tVector ) )
	arrow_tVec_a_i_StopOverP_at_beta0 = drawArrow( beta0.pt, np.add( beta0.pt, gamma1_a_i_to_b0_stopover.tVector ) ) 
	
	pts_arr.append( a_i )
	arrow_at_pts_arr.append( arrow_tVec_at_a_i )
	stopPT_arrow_at_beta0_arr.append( arrow_tVec_a_i_StopOverP_at_beta0 )
	directPT_arrow_at_beta0_arr.append( arrow_tVec_a_i_directP_at_beta0 )

	# Geodesics from a_i to f_i
	tVec_a_i_to_f_i = a_i.LogMap( f_i )

	a_i_f_i_geodesic_vtk = vtk.vtkPolyData()
	a_i_f_i_geodesic_pts = vtk.vtkPoints()
	a_i_f_i_geodesic_lines = vtk.vtkCellArray()

	for i in range( nLineTimePt ):
		t_i = ( i + 1 - t0 ) / float( nLineTimePt )

		v_t = tVec_a_i_to_f_i.ScalarMultiply( t_i )

		p_t = a_i.ExponentialMap( v_t )

		a_i_f_i_geodesic_pts.InsertNextPoint( p_t.pt[ 0 ], p_t.pt[ 1 ], p_t.pt[ 2 ] )

		if i == ( nLineTimePt - 1 ):
			continue

		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, i )
		line_i.GetPointIds().SetId( 1, i + 1 )
		a_i_f_i_geodesic_lines.InsertNextCell( line_i )

	a_i_f_i_geodesic_pts.Modified()
	a_i_f_i_geodesic_lines.Modified()

	a_i_f_i_geodesic_vtk.SetPoints( a_i_f_i_geodesic_pts )
	a_i_f_i_geodesic_vtk.SetLines( a_i_f_i_geodesic_lines )
	a_i_f_i_geodesic_vtk.Modified()

	a_to_f_geodesics_arr.append( a_i_f_i_geodesic_vtk )

	# Geodesics from a_i to beta0
	tVec_a_i_to_b0 = a_i.LogMap( beta0 )

	a_i_b0_geodesic_vtk = vtk.vtkPolyData()
	a_i_b0_geodesic_pts = vtk.vtkPoints()
	a_i_b0_geodesic_lines = vtk.vtkCellArray()

	for i in range( nLineTimePt ):
		t_i = ( i + 1 - t0 ) / float( nLineTimePt )

		v_t = tVec_a_i_to_b0.ScalarMultiply( t_i )

		p_t = a_i.ExponentialMap( v_t )

		a_i_b0_geodesic_pts.InsertNextPoint( p_t.pt[ 0 ], p_t.pt[ 1 ], p_t.pt[ 2 ] )

		if i == ( nLineTimePt - 1 ):
			continue

		line_i = vtk.vtkLine()
		line_i.GetPointIds().SetId( 0, i )
		line_i.GetPointIds().SetId( 1, i + 1 )
		a_i_b0_geodesic_lines.InsertNextCell( line_i )

	a_i_b0_geodesic_pts.Modified()
	a_i_b0_geodesic_lines.Modified()

	a_i_b0_geodesic_vtk.SetPoints( a_i_b0_geodesic_pts )
	a_i_b0_geodesic_vtk.SetLines( a_i_b0_geodesic_lines )
	a_i_b0_geodesic_vtk.Modified()

	a_to_b0_geodesics_arr.append( a_i_b0_geodesic_vtk )

# Geodesics Append
a_to_f_geodesics_appender = vtk.vtkAppendPolyData()
a_to_b0_geodesics_appender = vtk.vtkAppendPolyData()

for i in range( len( a_to_b0_geodesics_arr ) ):
	a_to_f_geodesics_appender.AddInputData( a_to_f_geodesics_arr[ i ] )
	a_to_b0_geodesics_appender.AddInputData( a_to_b0_geodesics_arr[ i ] )

a_to_f_geodesics_appender.Update()
a_to_b0_geodesics_appender.Update()

a_to_f_geodesics_cleaner = vtk.vtkCleanPolyData()
a_to_f_geodesics_cleaner.SetInputData( a_to_f_geodesics_appender.GetOutput() )
a_to_f_geodesics_cleaner.Update()

a_to_b0_geodesics_cleaner = vtk.vtkCleanPolyData()
a_to_b0_geodesics_cleaner.SetInputData( a_to_b0_geodesics_appender.GetOutput() )
a_to_b0_geodesics_cleaner.Update()




# a_i to f_i actor
a_to_f_geodesicsMapper = vtk.vtkPolyDataMapper()
a_to_f_geodesicsMapper.SetInputData( a_to_f_geodesics_cleaner.GetOutput() )
a_to_f_geodesicsMapper.Update()

# a_i to b0 actor
a_to_b0_geodesicsMapper = vtk.vtkPolyDataMapper()
a_to_b0_geodesicsMapper.SetInputData( a_to_b0_geodesics_cleaner.GetOutput() )
a_to_b0_geodesicsMapper.Update()


# Direct Transported Tangent Vectors
points_Arrows_Appender = vtk.vtkAppendPolyData()
directPT_Arrows_Appender = vtk.vtkAppendPolyData()
stopOverPT_Arrows_Appender = vtk.vtkAppendPolyData()

for i in range( len( directPT_arrow_at_beta0_arr ) ):
	directPT_Arrows_Appender.AddInputData( directPT_arrow_at_beta0_arr[ i ] )
	stopOverPT_Arrows_Appender.AddInputData( stopPT_arrow_at_beta0_arr[ i ] )
	points_Arrows_Appender.AddInputData( arrow_at_pts_arr[ i ] )

directPT_Arrows_Appender.Update()
stopOverPT_Arrows_Appender.Update()
points_Arrows_Appender.Update()

directPT_Cleaner = vtk.vtkCleanPolyData()
directPT_Cleaner.SetInputData( directPT_Arrows_Appender.GetOutput() )
directPT_Cleaner.Update()

stopOverPT_Cleaner = vtk.vtkCleanPolyData()
stopOverPT_Cleaner.SetInputData( stopOverPT_Arrows_Appender.GetOutput() )
stopOverPT_Cleaner.Update()

ArrowsAtPTs_Cleaner = vtk.vtkCleanPolyData()
ArrowsAtPTs_Cleaner.SetInputData( points_Arrows_Appender.GetOutput() )
ArrowsAtPTs_Cleaner.Update()

# f and beta0
points_bases = vtk.vtkPoints()

points_bases.InsertNextPoint( f_i.pt[ 0 ], f_i.pt[ 1 ], f_i.pt[ 2 ]  )
points_bases.InsertNextPoint( beta0.pt[ 0 ], beta0.pt[ 1 ], beta0.pt[ 2 ]  )

pts_bases_polyData = vtk.vtkPolyData()
pts_bases_polyData.SetPoints( points_bases )

vertFilter_bases = vtk.vtkVertexGlyphFilter()
vertFilter_bases.SetInputData( pts_bases_polyData )
vertFilter_bases.Update()

ptsPolyData_bases = vertFilter_bases.GetOutput()


ptsMapper_bases = vtk.vtkPolyDataMapper()
ptsMapper_bases.SetInputData( ptsPolyData_bases )

# Given Points 
points = vtk.vtkPoints()

for i in range( len( pts_arr ) ):
	points.InsertNextPoint( pts_arr[ i ].pt[ 0 ], pts_arr[ i ].pt[ 1 ], pts_arr[ i ].pt[ 2 ]  )

pts_polyData = vtk.vtkPolyData()
pts_polyData.SetPoints( points )

vertFilter = vtk.vtkVertexGlyphFilter()
vertFilter.SetInputData( pts_polyData )
vertFilter.Update()

ptsPolyData = vertFilter.GetOutput()

ptsMapper = vtk.vtkPolyDataMapper()
ptsMapper.SetInputData( ptsPolyData )



# Actors
# f / beta0
ptsActor_bases = vtk.vtkActor()
ptsActor_bases.SetMapper( ptsMapper_bases )
ptsActor_bases.GetProperty().SetPointSize( 25 )
ptsActor_bases.GetProperty().SetColor( 205 / 255.0, 133 / 255.0, 63 / 255.0 )
ptsActor_bases.GetProperty().SetRenderPointsAsSpheres( 1 )

# Tangent Vector at beta0
arrowMapper_org = vtk.vtkPolyDataMapper()
arrowMapper_org.SetInputData( arrow_tVec_org )
arrowMapper_org.Update()
arrowActor_org = vtk.vtkActor()
arrowActor_org.SetMapper( arrowMapper_org )
arrowActor_org.GetProperty().SetColor( 205 / 255.0, 133 / 255.0, 63 / 255.0 )

# beta0 to f
beta0_f_i_geodesic_actor = vtk.vtkActor()
beta0_f_i_geodesic_actor.SetMapper( beta0_f_i_geodesic_mapper )
beta0_f_i_geodesic_actor.GetProperty().SetOpacity( 1 )
beta0_f_i_geodesic_actor.GetProperty().SetColor( [ 1.0, 0.0, 0.0 ] )
beta0_f_i_geodesic_actor.GetProperty().SetLineWidth( 15 )
beta0_f_i_geodesic_actor.GetProperty().SetRenderLinesAsTubes( 1 )

# a_i
ptsActor = vtk.vtkActor()
ptsActor.SetMapper( ptsMapper )
ptsActor.GetProperty().SetPointSize( 15 )
ptsActor.GetProperty().SetColor( 0.0, 1.0, 1.0 )
ptsActor.GetProperty().SetRenderPointsAsSpheres( 1 )

# Tangent Vectors at a_i 
arrowMapper_a_i = vtk.vtkPolyDataMapper()
arrowMapper_a_i.SetInputData( ArrowsAtPTs_Cleaner.GetOutput() )
arrowMapper_a_i.Update()
arrowActor_a_i = vtk.vtkActor()
arrowActor_a_i.SetMapper( arrowMapper_a_i )
arrowActor_a_i.GetProperty().SetColor( 0.0, 1.0, 1.0 )
arrowActor_a_i.GetProperty().SetOpacity( 0.7 )


# a_i to beta0
a_to_b0_geodesicsActor = vtk.vtkActor()
a_to_b0_geodesicsActor.SetMapper( a_to_b0_geodesicsMapper )
a_to_b0_geodesicsActor.GetProperty().SetColor( 1.0, 0.0, 1.0 )
a_to_b0_geodesicsActor.GetProperty().SetOpacity( 0.5 )
a_to_b0_geodesicsActor.GetProperty().SetLineWidth( 8 )
a_to_b0_geodesicsActor.GetProperty().SetRenderLinesAsTubes( 1 )

# Direct Transported Tangent Vectors
arrowMapper_directP = vtk.vtkPolyDataMapper()
arrowMapper_directP.SetInputData( directPT_Cleaner.GetOutput() )
arrowMapper_directP.Update()
arrowActor_directP = vtk.vtkActor()
arrowActor_directP.SetMapper( arrowMapper_directP )
arrowActor_directP.GetProperty().SetColor( 1.0, 0.0, 1.0 )
arrowActor_directP.GetProperty().SetOpacity( 0.5 )

# a_i to f
a_to_f_geodesicsActor = vtk.vtkActor()
a_to_f_geodesicsActor.SetMapper( a_to_f_geodesicsMapper )
a_to_f_geodesicsActor.GetProperty().SetColor( 0.0, 0.0, 1.0 )
a_to_f_geodesicsActor.GetProperty().SetLineWidth( 8 )
a_to_f_geodesicsActor.GetProperty().SetRenderLinesAsTubes( 1 )

# beta0 to f
beta0_f_i_geodesic_actor2 = vtk.vtkActor()
beta0_f_i_geodesic_actor2.SetMapper( beta0_f_i_geodesic_mapper )
beta0_f_i_geodesic_actor2.GetProperty().SetOpacity( 0.8 )
beta0_f_i_geodesic_actor2.GetProperty().SetColor( [ 0.0, 0.0, 1.0 ] )
beta0_f_i_geodesic_actor2.GetProperty().SetLineWidth( 25 )
beta0_f_i_geodesic_actor2.GetProperty().SetRenderLinesAsTubes( 0 )

# Stop-Over Transported Tangent Vectors
arrowMapper_StopOverP = vtk.vtkPolyDataMapper()
arrowMapper_StopOverP.SetInputData( stopOverPT_Cleaner.GetOutput() )
arrowMapper_StopOverP.Update()
arrowActor_StopOverP = vtk.vtkActor()
arrowActor_StopOverP.SetMapper( arrowMapper_StopOverP )
arrowActor_StopOverP.GetProperty().SetColor( 0.0, 0.0, 1.0 )


'''
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
'''

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
conActor.GetProperty().SetOpacity( 1.0 )
conActor.GetProperty().SetColor( 0.9, 0.9, 0.9 )
conActor.GetProperty().SetEdgeColor( 0.4, 0.4, 0.7 )
conActor.GetProperty().EdgeVisibilityOn()
conActor.GetProperty().SetAmbient(0.3)
conActor.GetProperty().SetDiffuse(0.375)
conActor.GetProperty().SetSpecular(0.0)

# Renderer1
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

light = vtk.vtkLight() 
light.SetFocalPoint(0,0.6125,1.875)
light.SetPosition(1,0.875,1.6125)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

# Data 
ren.AddActor( conActor )
ren.AddActor( arrowActor_org )
ren.AddActor( arrowActor_a_i )
ren.AddActor( ptsActor )
ren.AddActor( beta0_f_i_geodesic_actor )
ren.AddActor( ptsActor_bases ) 

renWin.Render()
renIn.Start()


# Direct Transport 
ren.AddActor( arrowActor_directP )
ren.AddActor( a_to_b0_geodesicsActor )
renWin.Render()
renIn.Start()

# StopOver Transport
ren.RemoveActor( arrowActor_directP )
ren.RemoveActor( a_to_b0_geodesicsActor )

ren.AddActor( arrowActor_StopOverP )
ren.AddActor( a_to_f_geodesicsActor )
ren.AddActor( beta0_f_i_geodesic_actor2 )

renWin.Render()
renIn.Start()
