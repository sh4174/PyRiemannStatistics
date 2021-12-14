# Geodesic Regression on Sphere Manifold
# Manifolds 
import MReps 
import atom 
import numpy as np

# Visualization
import vtk

# Stats Model
import statsmodels.api as sm

# Ground Truth
p_interp = atom.sphere_atom()
v_slope = atom.sphere_tVec()


random_interp = np.random.rand(3)
random_interp_n = np.divide( random_interp, np.linalg.norm( random_interp ) )

random_tangent_vector = np.random.rand(3)
random_scale = np.random.rand(1) * 2
random_tangent_vector = np.multiply( random_tangent_vector, random_scale )


p_interp.SetSpherePt( random_interp_n )
v_slope.SetTangentVector( random_tangent_vector )

# Generating sphere atoms distributed over time perturbed by Gaussian random
# Time
t0 = 0
t1 = 1

# Generate a random point on the manifold
nData = 1000
dim = 3
sigma = 0.05

pt_list = []
t_list = []

# print( "Grount Truth" )

for n in range( nData ):
	time_pt = np.random.uniform( t0, t1 )
	# time_pt = ( t1 - t0 ) * n / nData + t0

	# Generate a random Gaussians with polar Box-Muller Method
	rand_pt = [ 0, 0, 0 ] 
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
	rand_tVec = atom.sphere_tVec()
	rand_tVec.SetTangentVector( rand_pt )

	v_t = atom.sphere_tVec() 
	v_t.SetTangentVector( [ v_slope.tVector[0] * time_pt, v_slope.tVector[1] * time_pt, v_slope.tVector[2] * time_pt ]  )
	mean = p_interp.ExponentialMap( v_t )

	# print( "Mean At Time : " + str( time_pt ) )	
	# print( mean.sphere_pt )

	# Projected Tangent to Mean Point
	rand_tVec_projected = mean.ProjectTangent( mean, rand_tVec )

	# print( "Random Tangent" )
	# print( rand_tVec.tVector )

	# print( "Projected Random Tangent" )
	# print( rand_tVec_projected.tVector )

	# Perturbed point at time_pt 
	pt_perturbed = mean.ExponentialMap( rand_tVec_projected )

	# print( "Perturbed pt At Time : " + str( time_pt ) )	
	# print( pt_perturbed.sphere_pt )

	pt_list.append( pt_perturbed )
	t_list.append( time_pt )

# print( pt_list[ 0 ].sphere_pt )
# print( t_list[ 0 ] )

#######################
# Geodesic Regression #
#######################



##########################
#  Anchored Point Linear Regression  #
##########################
print( "====================================" )
print( "Linear Regression on PGs" )
print( "====================================" ) 

# Calculate Intrinsic Mean
print( "====================================" )
print( "Calculate Intrinsic Mean" )
print( "====================================" ) 
# max_iter = 100
tol = 0.1

# Initialize
mu = atom.sphere_atom()
max_iter = 500

for i in range( max_iter ):
	print( "=================================" ) 
	print( str( i ) + "th Iteration" )
	print( "=================================" )

	dMu_k = atom.sphere_tVec()

	for j in range( nData ):
		Log_mu_M_j_k = mu.LogMap( pt_list[ j ] )

		dMu_k.tVector[ 0 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ] )
		dMu_k.tVector[ 1 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 1 ] )
		dMu_k.tVector[ 2 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 2 ] )
			
		Mu_k = mu.ExponentialMap( dMu_k )
		mu = Mu_k


mu = pt_list[0]

print( mu.sphere_pt )


print( "====================================" )
print( " Project data to Intrinsic Mean" )
print( "====================================" ) 

tVec_list = []
w_1 = [] 
w_2 = []
w_3 = []

for j in range( nData ):
	tVec_j = mu.LogMap( pt_list[ j ] )

	u_j_arr = []
	u_j_arr = [ tVec_j.tVector[0], tVec_j.tVector[1], tVec_j.tVector[2] ] 

	w_1_j = tVec_j.tVector[0]
	w_2_j = tVec_j.tVector[1]
	w_3_j = tVec_j.tVector[2]

	w_1.append( w_1_j )
	w_2.append( w_2_j )
	w_3.append( w_3_j )

w_1 = np.asarray( w_1 ) 
w_2 = np.asarray( w_2 )
w_3 = np.asarray( w_3 )


print( "======================================" )
print( " Linear Regression on Tangent Vectors " )
print( "======================================" ) 

t_list_sm = sm.add_constant( t_list )

LS_model1 = sm.OLS( w_1, t_list_sm )
est1 = LS_model1.fit()
print( est1.summary() )

LS_model2 = sm.OLS( w_2, t_list_sm )
est2 = LS_model2.fit()
print( est2.summary() )

LS_model3 = sm.OLS( w_3, t_list_sm )
est3 = LS_model3.fit()
print( est3.summary() )


nTimePt = 100
est_trend_pt_list = []
est_trend_pt_list_par_trans = [] 
est_trend_pt_list_add = []

est_trend_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0
	
	est1_fitted = np.add( np.multiply( time_pt, est1.params[1] ), est1.params[0] )

	est2_fitted = np.add( np.multiply( time_pt, est2.params[1] ), est2.params[0] )

	est3_fitted = np.add( np.multiply( time_pt, est3.params[1] ), est3.params[0] )

	tVec_t = atom.sphere_tVec()
	tVec_t.tVector[ 0 ] = est1_fitted
	tVec_t.tVector[ 1 ] = est2_fitted
	tVec_t.tVector[ 2 ] = est3_fitted

	pt_t = mu.ExponentialMap( tVec_t )
	est_trend_pt_list_add.append( pt_t )

	est_trend_t_list.append( time_pt )

AReg_tVec_v_w = atom.sphere_tVec()
AReg_tVec_v_w.tVector[0] = est1.params[1]
AReg_tVec_v_w.tVector[1] = est2.params[1]
AReg_tVec_v_w.tVector[2] = est3.params[1]

AReg_tVec_v_wp = atom.sphere_tVec()
AReg_tVec_v_wp.tVector[0] = est1.params[0]
AReg_tVec_v_wp.tVector[1] = est2.params[0]
AReg_tVec_v_wp.tVector[2] = est3.params[0]

AReg_p_interp = mu.ExponentialMap( AReg_tVec_v_wp )
AReg_tVec_v_p = mu.ParallelTranslateAtoB( mu, AReg_p_interp, AReg_tVec_v_w ) 

print( "=====================================" )
print( "            Ground Truth ")
print( "=====================================" )

print( "True P" )
print( p_interp.sphere_pt )
print( "True V" )
print( v_slope.tVector )

print( "=====================================" )
print( "   Geodesic Regression Results ")
print( "=====================================" )

print( "Estimated P" )
print( base.sphere_pt )
print( "Estimated V" ) 
print( tangent.tVector )

print( "============================================" )
print( "   Anchor Point Linear Regression Results ")
print( "============================================" )

print( "Estimated P" )
print( AReg_p_interp.sphere_pt )
print( "Estimated V" ) 
print( AReg_tVec_v_p.tVector )



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
conActor.GetProperty().SetOpacity( 0.3 )
conActor.GetProperty().SetColor( 0.6, 0.9, 0.7 )
conActor.GetProperty().SetEdgeColor( 0.4, 0.4, 0.7 )
conActor.GetProperty().EdgeVisibilityOn()

# Visualize spherical points
points = vtk.vtkPoints()

for i in range( len( pt_list ) ):
	points.InsertNextPoint( pt_list[ i ].sphere_pt[0], pt_list[ i ].sphere_pt[1], pt_list[ i ].sphere_pt[2] )

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


# Visualize a ground truth trend line
trend_pt_list = []
trend_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = atom.sphere_tVec() 
	v_t.SetTangentVector( [ v_slope.tVector[0] * time_pt, v_slope.tVector[1] * time_pt, v_slope.tVector[2] * time_pt ]  )
	mean = p_interp.ExponentialMap( v_t )

	trend_pt_list.append( mean )
	trend_t_list.append( time_pt )

linePolyData = vtk.vtkPolyData()
linePts = vtk.vtkPoints()

for i in range( len( trend_pt_list ) ):
	linePts.InsertNextPoint( trend_pt_list[ i ].sphere_pt[0], trend_pt_list[ i ].sphere_pt[1], trend_pt_list[ i ].sphere_pt[2] )

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
lineActor.GetProperty().SetColor( 0, 0, 1 )
lineActor.GetProperty().SetOpacity( 0.6 )
lineActor.GetProperty().SetLineWidth( 5 )

# # Visualize an estimated trend
# estLinePolyData = vtk.vtkPolyData()
# estLinePts = vtk.vtkPoints()

# for i in range( len( est_trend_pt_list ) ):
# 	estLinePts.InsertNextPoint( est_trend_pt_list[ i ].sphere_pt[0], est_trend_pt_list[ i ].sphere_pt[1], est_trend_pt_list[ i ].sphere_pt[2] )

# estLinePolyData.SetPoints( estLinePts )

# estLines = vtk.vtkCellArray()

# for i in range( len( est_trend_pt_list ) - 1 ):
# 	line_i = vtk.vtkLine()
# 	line_i.GetPointIds().SetId( 0, i )
# 	line_i.GetPointIds().SetId( 1, i + 1 )
# 	estLines.InsertNextCell( line_i )

# estLinePolyData.SetLines( estLines )

# estLineMapper = vtk.vtkPolyDataMapper()
# estLineMapper.SetInputData( estLinePolyData )

# estLineActor = vtk.vtkActor()
# estLineActor.SetMapper( estLineMapper )
# estLineActor.GetProperty().SetColor( 1, 0, 0 )
# estLineActor.GetProperty().SetOpacity( 0.6 )
# estLineActor.GetProperty().SetLineWidth( 4 )


# # Visualize an estimated trend 2
# estLinePolyData2 = vtk.vtkPolyData()
# estLinePts2 = vtk.vtkPoints()

# for i in range( len( est_trend_pt_list_par_trans ) ):
# 	estLinePts2.InsertNextPoint( est_trend_pt_list_par_trans[ i ].sphere_pt[0], est_trend_pt_list_par_trans[ i ].sphere_pt[1], est_trend_pt_list_par_trans[ i ].sphere_pt[2] )

# estLinePolyData2.SetPoints( estLinePts2 )

# estLines2 = vtk.vtkCellArray()

# for i in range( len( est_trend_pt_list_par_trans ) - 1 ):
# 	line_i = vtk.vtkLine()
# 	line_i.GetPointIds().SetId( 0, i )
# 	line_i.GetPointIds().SetId( 1, i + 1 )
# 	estLines2.InsertNextCell( line_i )

# estLinePolyData2.SetLines( estLines2 )

# estLineMapper2 = vtk.vtkPolyDataMapper()
# estLineMapper2.SetInputData( estLinePolyData2 )

# estLineActor2 = vtk.vtkActor()
# estLineActor2.SetMapper( estLineMapper2 )
# estLineActor2.GetProperty().SetColor( 0, 1, 0.5 )
# estLineActor2.GetProperty().SetOpacity( 0.6 )
# estLineActor2.GetProperty().SetLineWidth( 5 )


# Visualize an estimated trend 3
estLinePolyData3 = vtk.vtkPolyData()
estLinePts3 = vtk.vtkPoints()

for i in range( len( est_trend_pt_list_add ) ):
	estLinePts3.InsertNextPoint( est_trend_pt_list_add[ i ].sphere_pt[0], est_trend_pt_list_add[ i ].sphere_pt[1], est_trend_pt_list_add[ i ].sphere_pt[2] )

estLinePolyData3.SetPoints( estLinePts3 )

estLines3 = vtk.vtkCellArray()

for i in range( len( est_trend_pt_list_add ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	estLines3.InsertNextCell( line_i )

estLinePolyData3.SetLines( estLines3 )

estLineMapper3 = vtk.vtkPolyDataMapper()
estLineMapper3.SetInputData( estLinePolyData3 )

estLineActor3 = vtk.vtkActor()
estLineActor3.SetMapper( estLineMapper3 )
estLineActor3.GetProperty().SetColor( 0, 1, 0 )
# estLineActor3.GetProperty().SetOpacity( 1. )
estLineActor3.GetProperty().SetLineWidth( 3 )


# Renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )
ren.AddActor( ptsActor )
ren.AddActor( lineActor )
ren.AddActor( estLineActor )
# ren.AddActor( estLineActor2 )
ren.AddActor( estLineActor3 )

ren.SetBackground( 1.0, 1.0, 1.0 )


renWin.Render()
renIn.Start()
