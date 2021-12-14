#####################################################################
#####################################################################
# Example - Geodesic Regression										#		
# Manifold - Kendall 2D Manifold									#
#####################################################################
#####################################################################

import manifolds 
import numpy as np
import StatsModel as sm

import matplotlib.pyplot as plt
# Visualization
import vtk

import pandas as pd

# Read Point
pts_data = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/AllBoundaries.csv", header=None )

nObs = pts_data.shape[ 0 ]
nPt = pts_data.shape[ 1 ]
nPt = nPt - 1 

print( nObs )
print( nPt )

pts_list = [ ] 

for i in range( nObs ):
	pts_list_i = []

	for j in range( nPt ):
		pts_str = pts_data[ j ][ i ] 

		pts_str_parsed = ''.join( ( ch if ch in '0123456789.-e' else ' ' ) for ch in pts_str )

		pt_j = [ float( pt_j_str ) for  pt_j_str in pts_str_parsed.split() ]

		pts_list_i.append( pt_j ) 

	pts_list.append( pts_list_i )

print( len( pts_list ) )
print( len( pts_list[ 0 ] ) ) 

print( pts_list[ 0 ][ 0 ] )
print( pts_list[ 0 ][ 99 ] )

# Procrustes Filter
procrustes_filter = vtk.vtkProcrustesAlignmentFilter()

group = vtk.vtkMultiBlockDataGroupFilter()

pts_polyData_list = []

# Check
# nObs = 3

for i in range( nObs ):
	pt_polyData = vtk.vtkPolyData()
	pt_Points = vtk.vtkPoints()
	pt_lines = vtk.vtkCellArray()

	# Set Points
	for j in range( nPt ):
		pt_Points.InsertNextPoint( pts_list[ i ][ j ][ 0 ], pts_list[ i ][ j ][ 1 ], 0 )

	for j in range( nPt ):
		line_j = vtk.vtkLine()
		line_j.GetPointIds().SetId( 0, j )
		line_j.GetPointIds().SetId( 1, j + 1 )

		if j == ( nPt - 1 ):
			line_j.GetPointIds().SetId( 0, j )
			line_j.GetPointIds().SetId( 1, 0 )

		pt_lines.InsertNextCell( line_j )

	pt_polyData.SetPoints( pt_Points )
	pt_polyData.SetLines( pt_lines )

	pts_polyData_list.append( pt_polyData )

	group.AddInputData( pt_polyData )

group.Update()

procrustes_filter.SetInputData( group.GetOutput() )
procrustes_filter.GetLandmarkTransform().SetModeToSimilarity()
procrustes_filter.Update()

outputPolyData0 = procrustes_filter.GetOutput().GetBlock( 0 )
print( outputPolyData0.GetPoint( 0 ) )

# Read Subject ID
subj_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/subjects.csv", header=None )

# Read Time
age_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/age.csv", header=None )


# Exception List 
except_list = np.asarray( [ 246, 303, 356, 376, 636, 680, 934, 961, 998, 1005, 1033, 1039 ] )
except_list = except_list - 1

# Kendall Shape Space 
shape_list = []
shape_subj_list = []
shape_age_list = []

for i in range( nObs ):

	# EException leist
	if i in except_list:
		continue

	output_i = procrustes_filter.GetOutput().GetBlock( i )

	size_i = 0
	center_of_mass_i = 0

	for j in range( nPt ):
		pt_ij = output_i.GetPoint( j )

		size_i += ( pt_ij[ 0 ] ** 2 + pt_ij[ 1 ] ** 2 + pt_ij[ 2 ] ** 2 )
		center_of_mass_i += ( pt_ij[ 0 ] + pt_ij[ 1 ] + pt_ij[ 2 ] )	

	center_of_mass_i = center_of_mass_i / float( nPt )

	print( size_i ) 	
	print( center_of_mass_i )

	pt_mat_i = np.zeros( [ 2, nPt ] )

	for j in range( nPt ):
		pt_ij = output_i.GetPoint( j )

		pt_ij_norm = [ ( pt_ij[ 0 ] - center_of_mass_i ) / size_i, ( pt_ij[ 1 ] - center_of_mass_i ) / size_i ]

		pt_mat_i[ 0, j ] = pt_ij_norm[ 0 ]
		pt_mat_i[ 1, j ] = pt_ij_norm[ 1 ] 

	shape_i = manifolds.kendall2D( nPt )
	shape_i.SetPoint( pt_mat_i )

	shape_list.append( shape_i )
	shape_subj_list.append( subj_list[ 0 ][ i ] )
	shape_age_list.append( float( age_list[ 0 ][ i ] ) ) 


# print( shape_age_list[ 0 ] + shape_age_list[ 1 ] )

min_age = np.min( shape_age_list )
max_age = np.max( shape_age_list )

shape_normalized_age_list = np.divide( np.subtract( shape_age_list, min_age ), max_age - min_age )
max_norm_age = np.max( shape_normalized_age_list )
min_norm_age = np.min( shape_normalized_age_list )

# mean_shape = sm.FrechetMean( shape_list )

# # Visualization
# mean_x_list = [] 
# mean_y_list = []

# for k in range( nPt ):
# 	mean_x_list.append( mean_shape.pt[ 0, k ] )
# 	mean_y_list.append( mean_shape.pt[ 1, k ] )

# mean_x_list.append( mean_shape.pt[ 0, 0 ] )
# mean_y_list.append( mean_shape.pt[ 1, 0 ] )

# plt.figure()
# plt.plot( mean_x_list, mean_y_list )
# plt.show()

intercept, slope = sm.GeodesicRegression( shape_normalized_age_list, shape_list, max_iter=1000, stepSize=0.0005, step_tol=1e-12 )

print( intercept.pt )

# Visualization
intercept_x_list = [] 
intercept_y_list = []

for k in range( nPt ):
	intercept_x_list.append( intercept.pt[ 0, k ] )
	intercept_y_list.append( intercept.pt[ 1, k ] )

intercept_x_list.append( intercept.pt[ 0, 0 ] )
intercept_y_list.append( intercept.pt[ 1, 0 ] )

plt.figure()
plt.plot( intercept_x_list, intercept_y_list )

for t in range( 100 ):
	t_vis = ( t + 1 ) * 0.01

	slope_t = slope.ScalarMultiply( t_vis )

	shape_t = intercept.ExponentialMap( slope_t )

	shape_t_x_list = []
	shape_t_y_list = []

	for k in range( nPt ):
		shape_t_x_list.append( shape_t.pt[ 0, k ] )
		shape_t_y_list.append( shape_t.pt[ 1, k ] )

	shape_t_x_list.append( shape_t.pt[ 0, 0 ] )
	shape_t_y_list.append( shape_t.pt[ 1, 0 ] )


	plt.plot( shape_t_x_list, shape_t_y_list )

plt.show()

print( max_age )
print( min_age )

print( max_norm_age )
print( min_norm_age )

print( slope.tVector )