################################################################################################
################################################################################################
# Example - Frechet Mean
# Manifold - Kendall 2D Manifold 																   #
################################################################################################
################################################################################################

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
nObs = 3

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

# Exception List 
except_list = np.asarray( [ 246, 303, 356, 376, 636, 680, 934, 961, 998, 1005, 1033, 1039 ] )
except_list = except_list - 1

# Kendall Shape Space 
shape_list = []
shape_subj_list = []

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

mean_shape = sm.FrechetMean( shape_list )

# Visualization
mean_x_list = [] 
mean_y_list = []

for k in range( nPt ):
	mean_x_list.append( mean_shape.pt[ 0, k ] )
	mean_y_list.append( mean_shape.pt[ 1, k ] )

mean_x_list.append( mean_shape.pt[ 0, 0 ] )
mean_y_list.append( mean_shape.pt[ 1, 0 ] )

plt.figure()
plt.plot( mean_x_list, mean_y_list )
plt.show()





# # Check Construction
# # Construction Test
# shape_0 = manifolds.kendall2D( nPt )
# print( len( shape_list[ 0 ] ) ) 
# print( nPt )

# shape_0.SetPoint( shape_list[ 0 ] )
# print( shape_0.pt[ 0, 1 ] )

# # Log Map Check 
# shape0 = manifolds.kendall2D( nPt )
# shape0.SetPoint( shape_list[ 0 ] )

# shape1 = manifolds.kendall2D( nPt )
# shape1.SetPoint( shape_list[ 1 ] )

# print( "Construction Done" )
# tVec_check = shape0.LogMap( shape1 )

# print( tVec_check.tVector ) 

# # Exp Map Check
# shape1_mapped = shape0.ExponentialMap( tVec_check )

# print( shape1.pt )
# print( shape1_mapped.pt )

# print( "Mapped vs Shape 1 " )
# print( np.linalg.norm( np.subtract( shape1.pt, shape1_mapped.pt ) ) )

# print( "Shape 1 : Norm " )
# print( np.linalg.norm( shape1.pt ) )

# print( "Shape 0 : Norm " )
# print( np.linalg.norm( shape0.pt ) )

# print( "shape1 - shape0 " )
# print( np.linalg.norm( np.subtract( shape1.pt, shape0.pt ) ) )

# # Parallel Translate Check 
# shape2 = manifolds.kendall2D( nPt )
# shape2.SetPoint( shape_list[ 2 ] )

# tVec1_2 = shape1.LogMap( shape2 )

# tVec1_2_pTo0 = shape1.ParallelTranslateAtoB( shape1, shape0, tVec1_2 )
