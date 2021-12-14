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
#    PGA      #
#######################

# Gradient Descent Parameters
step_size = 0.01
max_iter = 500
step_tol = 1e-8

w, v, mu = sm.TangentPGA( pt_list, max_iter, step_tol, step_size )

print( "=======================================" )
print( " PGA Results " )
print( "=======================================" )

print( "=======================================" )
print( " Frechet Mean " )
print( mu.pt )
print( "=======================================" )


sum_w = np.sum( w )
w_accum = 0

print( "=======================================" )
print( "First 2 Components Eigenvalues" )
print( "=======================================" )

for i in range( 2 ):
	w_accum += w[ i ]
	print( w_accum / sum_w )

print( "=============================" ) 
print( " PGA Done " ) 
print( "=============================" )

print( "====================================" )
print( " Project data to PGs" )
print( "====================================" ) 

tVec_list = []
w_1 = [] 
w_2 = []

for j in range( nData ):
	tVec_j = mu.LogMap( pt_list[ j ] )

	u_j_arr = []
	u_j_arr = [ tVec_j.tVector[0], tVec_j.tVector[1], tVec_j.tVector[2] ] 

	tVec_list.append( tVec_j )

	w_1_j = np.dot( u_j_arr, v[ :, 0 ] )
	# u_j_arr_res = np.subtract( u_j_arr_res, np.multiply( w_k, v[ :, k ] ) )
	w_2_j = np.dot( u_j_arr, v[ :, 1 ] )

	w_1.append( w_1_j )
	w_2.append( w_2_j )

w_1 = np.asarray( w_1 ) 
w_2 = np.asarray( w_2 )

print( w )

print( "====================================" )
print( " PG 1 on Sphere : 3 Sigma " )
print( "====================================" ) 
nLambda = 50

lambda_list = np.linspace( -3 * np.sqrt( w[0] ), 3 * np.sqrt( w[ 0 ] ), nLambda )

PG_pt_list = [] 

for n in range( nLambda ):
	lambda_n = lambda_list[ n ] 

	tVec_n = manifolds.sphere_tVec(3)
	tVec_n.tVector[ 0 ] = v[ 0, 0 ] * lambda_n
	tVec_n.tVector[ 1 ] = v[ 1, 0 ] * lambda_n
	tVec_n.tVector[ 2 ] = v[ 2, 0 ] * lambda_n

	pt_n = mu.ExponentialMap( tVec_n )
	PG_pt_list.append( pt_n )


print( "====================================" )
print( " PG 2 on Sphere : 3 Sigma " )
print( "====================================" ) 
lambda_list2 = np.linspace( -3 * np.sqrt( w[1] ), 3 * np.sqrt( w[ 1 ] ), nLambda )

PG_pt_list2 = [] 

for n in range( nLambda ):
	lambda_n = lambda_list2[ n ] 

	tVec_n = manifolds.sphere_tVec(3)
	tVec_n.tVector[ 0 ] = v[ 0, 1 ] * lambda_n
	tVec_n.tVector[ 1 ] = v[ 1, 1 ] * lambda_n
	tVec_n.tVector[ 2 ] = v[ 2, 1 ] * lambda_n

	pt_n = mu.ExponentialMap( tVec_n )
	PG_pt_list2.append( pt_n )


###############################
#    Geodesic Regression      #
###############################

# Gradient Descent Parameters
step_size = 0.01
max_iter = 500
step_tol = 1e-8

interp_base, slope_tangent = sm.GeodesicRegression( t_list, pt_list, max_iter, step_size, step_tol, False )

print( "=======================================" )
print( " Geodesic Regression Results " )
print( "=======================================" )

print( "True P" )
print( p_interp.pt )
print( "True V" )
print( v_slope.tVector )

print( "Estimated P" )
print( interp_base.pt )
print( "Estimated V" ) 
print( slope_tangent.tVector )

print( "=======================================" )
print( " Geodesic Regression Trend on PG Spaces" )
print( "=======================================" )
nTimePt = 100
# Visualize an estimated trend of Linearized Geodesic Regression Model - Red
est_trend_pt_list = []
est_trend_pt_PG1_list = []
est_trend_pt_PG2_list = []

est_trend_t_list = []

for n in range( nTimePt ):
	time_pt = ( t1 - t0 ) * n / ( nTimePt - 1 ) + t0

	# Generate a random Gaussians with polar Box-Muller Method
	v_t = manifolds.sphere_tVec( nDimManifold ) 
	v_t.SetTangentVector( [ slope_tangent.tVector[0] * time_pt, slope_tangent.tVector[1] * time_pt, slope_tangent.tVector[2] * time_pt ]  )
	mean = interp_base.ExponentialMap( v_t )

	est_trend_pt_list.append( mean )
	est_trend_t_list.append( time_pt )

	tVec_n = mu.LogMap( mean )

	u_n_arr = [ tVec_n.tVector[0], tVec_n.tVector[1], tVec_n.tVector[2] ] 

	w_1_n = np.dot( u_n_arr, np.asarray( v[ :, 0 ].flatten() ).flatten() )
	# u_j_arr_res = np.subtract( u_j_arr_res, np.multiply( w_k, v[ :, k ] ) )
	w_2_n = np.dot( u_n_arr, np.asarray( v[ :, 1 ].flatten() ).flatten() )

	est_trend_pt_PG1_list.append( w_1_n )
	est_trend_pt_PG2_list.append( w_2_n )


########################################
#####        Visualization        ######   
########################################
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)

colors = [ [ 0, 0, 1 ], [ 1.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0, 0.5, 1.0 ], [ 0, 0, 1.0 ] ] 
est_colors =[ [ 0, 1, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  
est_lReg_colors =[ [ 1, 0, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  

# Regression line on PG 1 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( t_list, w_1, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( est_trend_t_list, est_trend_pt_PG1_list, c=est_colors[ 0 ], label="GeoReg" ) 
plt.xlabel('Time') 
plt.ylabel('PG 1 Coeff' )
plt.title( "PG1" )
plt.legend()
plt.tight_layout()

# Regression line on PG 2 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( t_list, w_2, c=colors[ 1 ], alpha=0.5, label="Data" )
plt.plot( est_trend_t_list, est_trend_pt_PG2_list, c=est_colors[ 1 ], label="GeoReg" ) 
plt.xlabel('Time') 
plt.ylabel('PG 2 Coeff' )
plt.title( "PG2" )
plt.legend()
plt.tight_layout()

plt.show() 

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
ptsActor.GetProperty().SetRenderPointsAsSpheres(1 )

# Visualize an PG1 trend
PG1PolyData = vtk.vtkPolyData()
PG1Pts = vtk.vtkPoints()

for i in range( len( PG_pt_list ) ):
	PG1Pts.InsertNextPoint( PG_pt_list[ i ].pt[0], PG_pt_list[ i ].pt[1], PG_pt_list[ i ].pt[2] )

PG1PolyData.SetPoints( PG1Pts )

PG1s = vtk.vtkCellArray()

for i in range( len( PG_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	PG1s.InsertNextCell( line_i )

PG1PolyData.SetLines( PG1s )

PG1Mapper = vtk.vtkPolyDataMapper()
PG1Mapper.SetInputData( PG1PolyData )

PG1Actor = vtk.vtkActor()
PG1Actor.SetMapper( PG1Mapper )
PG1Actor.GetProperty().SetColor( 1, 0, 0 )
PG1Actor.GetProperty().SetOpacity( 0.4 )
PG1Actor.GetProperty().SetLineWidth( 12 )

# Visualize an PG2 trend
PG2PolyData = vtk.vtkPolyData()
PG2Pts = vtk.vtkPoints()

for i in range( len( PG_pt_list2 ) ):
	PG2Pts.InsertNextPoint( PG_pt_list2[ i ].pt[0], PG_pt_list2[ i ].pt[1], PG_pt_list2[ i ].pt[2] )

PG2PolyData.SetPoints( PG2Pts )

PG2s = vtk.vtkCellArray()

for i in range( len( PG_pt_list2 ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	PG2s.InsertNextCell( line_i )

PG2PolyData.SetLines( PG2s )

PG2Mapper = vtk.vtkPolyDataMapper()
PG2Mapper.SetInputData( PG2PolyData )

PG2Actor = vtk.vtkActor()
PG2Actor.SetMapper( PG2Mapper )
PG2Actor.GetProperty().SetColor( 0.5, 0, 0.5 )
PG2Actor.GetProperty().SetOpacity( 0.4 )
PG2Actor.GetProperty().SetLineWidth( 12 )

# Visualize an GReg trend
GRegPolyData = vtk.vtkPolyData()
GRegPts = vtk.vtkPoints()

for i in range( len( est_trend_pt_list ) ):
	GRegPts.InsertNextPoint( est_trend_pt_list[ i ].pt[0], est_trend_pt_list[ i ].pt[1], est_trend_pt_list[ i ].pt[2] )

GRegPolyData.SetPoints( GRegPts )

GRegs = vtk.vtkCellArray()

for i in range( len( est_trend_pt_list ) - 1 ):
	line_i = vtk.vtkLine()
	line_i.GetPointIds().SetId( 0, i )
	line_i.GetPointIds().SetId( 1, i + 1 )
	GRegs.InsertNextCell( line_i )

GRegPolyData.SetLines( GRegs )

GRegMapper = vtk.vtkPolyDataMapper()
GRegMapper.SetInputData( GRegPolyData )

GRegActor = vtk.vtkActor()
GRegActor.SetMapper( GRegMapper )
GRegActor.GetProperty().SetColor( 0.0, 0, 0.1 )
GRegActor.GetProperty().SetOpacity( 1.0 )
GRegActor.GetProperty().SetLineWidth( 8 )

# Visualize Intrinsic Mean Point
meanPolyData1 = vtk.vtkPolyData()
meanPt1 = vtk.vtkPoints()

meanPt1.InsertNextPoint( mu.pt[0], mu.pt[1], mu.pt[2] )
meanPt1.Modified()

meanPolyData1.SetPoints( meanPt1 )
meanPolyData1.Modified()


meanVertFilter1 = vtk.vtkVertexGlyphFilter()
meanVertFilter1.SetInputData( meanPolyData1 )
meanVertFilter1.Update()


meanMapper1 = vtk.vtkPolyDataMapper()
meanMapper1.SetInputData( meanVertFilter1.GetOutput() )

meanActor1 = vtk.vtkActor()
meanActor1.SetMapper( meanMapper1 )
meanActor1.GetProperty().SetColor( 0, 0.5, 1 )
meanActor1.GetProperty().SetOpacity( 0.6 )
meanActor1.GetProperty().SetPointSize( 15 )
meanActor1.GetProperty().SetRenderPointsAsSpheres(1 )

# Renderer1
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

renIn = vtk.vtkRenderWindowInteractor()
renIn.SetRenderWindow( renWin )

ren.AddActor( conActor )
ren.AddActor( ptsActor )
ren.AddActor( PG1Actor )
ren.AddActor( PG2Actor )
ren.AddActor( meanActor1 )
ren.AddActor( GRegActor )

light = vtk.vtkLight() 
light.SetFocalPoint(0,0.6125,1.875)
light.SetPosition(1,0.875,1.6125)

ren.AddLight( light )
ren.SetBackground( 1.0, 1.0, 1.0 )

renWin.Render()
renIn.Start()
