#####################################################################################################################################
#####################################################################################################################################
# Example - Geodesic Regression																					  					#		
# Manifold - Kendall 2D Manifold																				  					#
# y : Corpus Callosum Shapes - 2D Kendall Shape Space				 											  					#
# t : Age  																										  					#	 
# Covariates : sex, hr, lr, autism, autismSpec, healthy 														  					#
# Model : y = beta0 + beta1 * s + beta2 * hr + beta3 * lr + beta4 * autism + beta5 * autismSpec * beta6 * healthy 					#
#                ( gamma0 + gamma1 * s + gamma2 * hr + gamma3 * lr + gamma4 * autism + gamma5 * autismSpec + gamma 6 * healthy )t   #
#####################################################################################################################################
#####################################################################################################################################

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


# Read Subject ID
subj_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/subjects.csv", header=None )
subj_list = []

for i in range( len( subj_csv_list[ 0 ] ) ):
	subj_list.append( subj_csv_list[ 0 ][ i ][ :6 ] )

# Unique Subject IDs
subj_set = list( set( subj_list ) )

# Read Sex 
male_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/males.csv", header=None )
male_list = []

for i in range( len( male_csv_list[ 0 ] ) ):
	male_list.append( int( male_csv_list[ 0 ][ i ] ) )

female_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/females.csv", header=None )
female_list = []

for i in range( len( female_csv_list[ 0 ] ) ):
	female_list.append( int( female_csv_list[ 0 ][ i ] ) )

# Read Autism Diagnosed 
autism_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/autism.csv", header=None )
autism_list = []
for i in range( len( autism_csv_list[ 0 ] ) ):
	autism_list.append( int( autism_csv_list[ 0 ][ i ] ) )

# Read Autism Spectrum Diagnosed
autismspec_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/autismspec.csv", header=None )
autismspec_list = []

for i in range( len( autismspec_csv_list[ 0 ] ) ):
	autismspec_list.append( int( autismspec_csv_list[ 0 ][ i ] ) )

# Read High risk 
hr_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/hr.csv", header=None )
hr_list = [] 
for i in range( len( hr_csv_list[ 0 ] ) ):
	hr_list.append( int( hr_csv_list[ 0 ][ i ] ) )

# Read Low risk
lr_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/lr.csv", header=None )
lr_list = [] 

for i in range( len( lr_csv_list[ 0 ] ) ):
	lr_list.append( int( lr_csv_list[ 0 ][ i ] ) )

# Read Healthy 
none_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/none.csv", header=None )
none_list = []

for i in range( len( none_csv_list[ 0 ] ) ):
	none_list.append( int( none_csv_list[ 0 ][ i ] ) )

# Read Time
age_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/age.csv", header=None )
age_list = [] 

for i in range( len( age_csv_list[ 0 ] ) ):
	age_list.append( float( age_csv_list[ 0 ][ i ] ) )

# Exception List 
except_list = np.asarray( [ 246, 303, 356, 376, 636, 680, 934, 961, 998, 1005, 1033, 1039 ] )
except_list = except_list - 1


# Kendall Shape Space 
shape_list = []
shape_age_list = []
shape_subj_list = subj_set 
shape_nObs_list = []
cov_int_list = []
cov_slope_list = []
subj_s_list = []
subj_autism_list = []
subj_autism_spec_list = []
subj_hr_list = []
subj_lr_list = []
subj_none_list = []



for i in range( len( subj_set ) ):
	shape_list.append( [] )
	shape_age_list.append( [] )
	shape_nObs_list.append( 0 )

	cov_int_list.append( [] )
	cov_slope_list.append( [] )

	subj_s_list.append( 0 )
	subj_autism_list.append( 0 )
	subj_autism_spec_list.append( 0 )
	subj_hr_list.append( 0 )
	subj_lr_list.append( 0 )
	subj_none_list.append( 0 )


#########################################################
# Covariates : sex, hr, lr, autism, autismSpec, healthy #
#########################################################

for i in range( nObs ):
	# Exception leist
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

	pt_mat_i = np.zeros( [ 2, nPt ] )

	for j in range( nPt ):
		pt_ij = output_i.GetPoint( j )

		pt_ij_norm = [ ( pt_ij[ 0 ] - center_of_mass_i ) / size_i, ( pt_ij[ 1 ] - center_of_mass_i ) / size_i ]

		pt_mat_i[ 0, j ] = pt_ij_norm[ 0 ]
		pt_mat_i[ 1, j ] = pt_ij_norm[ 1 ] 

	shape_i = manifolds.kendall2D( nPt )
	shape_i.SetPoint( pt_mat_i )

	# Covariates
	if i in male_list:
		s_i = 1
	else:
		s_i = 0

	if i in autism_list:
		autism_i = 1
	else:
		autism_i = 0

	if i in autismspec_list:
		autismspec_i = 1
	else:
		autismspec_i = 0

	if i in hr_list:
		hr_i = 1
	else:
		hr_i = 0

	if i in lr_list:
		lr_i = 1
	else:
		lr_i = 0

	if i in none_list:
		none_i = 1
	else:
		none_i = 0

	# Find Subject Index
	subj_i = subj_list[ i ]

	for j in range( len( subj_set ) ):
		if subj_i == subj_set[ j ]:
			shape_list[ j ].append( shape_i )
			shape_age_list[ j ].append( float( age_list[ i ] ) )

			if shape_nObs_list[ j ] == 0:
				cov_int_list[ j ].append( s_i )
				cov_int_list[ j ].append( hr_i )
				cov_int_list[ j ].append( lr_i )
				cov_int_list[ j ].append( autism_i )
				cov_int_list[ j ].append( autismspec_i )
				cov_int_list[ j ].append( none_i )

				cov_slope_list[ j ].append( s_i )
				cov_slope_list[ j ].append( hr_i )
				cov_slope_list[ j ].append( lr_i )
				cov_slope_list[ j ].append( autism_i )
				cov_slope_list[ j ].append( autismspec_i )
				cov_slope_list[ j ].append( none_i )

				subj_s_list[ j ] = s_i
				subj_autism_list[ j ] = autism_i
				subj_autism_spec_list[ j ] = autismspec_i
				subj_hr_list[ j ] = hr_i
				subj_lr_list[ j ] = lr_i
				subj_none_list[ j ] = none_i

			shape_nObs_list[ j ] = shape_nObs_list[ j ] + 1 


print( len( cov_int_list ) )
print( len( cov_slope_list ) )
print( len( shape_list ) )

print( shape_nObs_list )
print( np.sum( shape_nObs_list ) )

print( cov_int_list[ 0 ] )
print( cov_slope_list[ 0 ] )


shape_nObs_list_a = [] 
shape_list_a = []
shape_age_list_a = []
shape_subj_list_a = []

cov_int_list_a = []
cov_slope_list_a = []

subj_s_list_a = []
subj_autism_list_a = []
subj_autism_spec_list_a = []
subj_hr_list_a = []
subj_lr_list_a = []
subj_none_list_a = []


# Shape
shape_list_all = []
# Subjects
subj_list_all = []
# Age
t_list_all = [] 
# Sex
s_list_all = [] 
# HR
hr_list_all = []
# LR
lr_list_all = [] 
# Autism Diagnosed
autism_list_all = []
# Autism Spectrum Diagnosed
autismspec_list_all = []
# Healthy
none_list_all = []


# For Single Geodesic Model
cov_int_list_MLSG = []
cov_slope_list_MLSG = []

for i in range( len( shape_nObs_list ) ):
	nObs_i = shape_nObs_list[ i ] 

	if nObs_i > 1:
		shape_list_a.append( shape_list[ i ] )
		shape_age_list_a.append( shape_age_list[ i ] )
		shape_subj_list_a.append( shape_subj_list[ i ] )
		subj_s_list_a.append( subj_s_list[ i ] )
		subj_autism_list_a.append( subj_autism_list[ i ] )
		subj_autism_spec_list_a.append( subj_autism_spec_list[ i ] )
		subj_hr_list_a.append( subj_hr_list[ i ] )
		subj_lr_list_a.append( subj_lr_list[ i ] )
		subj_none_list_a.append( subj_none_list[ i ] )

		cov_int_list_a.append( cov_int_list[ i ] )
		cov_slope_list_a.append( cov_slope_list[ i ] )

		cov_int_list_MLSG.append( [ 0 ] )
		cov_slope_list_MLSG.append( [ 0 ] )

		shape_nObs_list_a.append( shape_nObs_list[ i ] )

		for j in range( len( shape_list[ i ] ) ):
			shape_list_all.append( shape_list[ i ][ j ] )
			t_list_all.append( shape_age_list[ i ][ j ] )
			subj_list_all.append( shape_subj_list[ i ] )
			s_list_all.append( subj_s_list[ i ] )
			hr_list_all.append( subj_hr_list[ i ] )
			lr_list_all.append( subj_lr_list[ i ] )
			autism_list_all.append( subj_autism_list[ i ] )
			autismspec_list_all.append( subj_autism_spec_list[ i ] )
			none_list_all.append( subj_none_list[ i ] )

data_list_pd = pd.DataFrame( { 'SubjectID': subj_list_all, 'Age':t_list_all, 'Sex':s_list_all, 'HR':hr_list_all, 'LR':lr_list_all, 'Autism':autism_list_all, 'AutismSpec':autismspec_list_all, 'None':none_list_all } )
data_list_pd.to_csv( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/Polished_Data_List.csv' )

# est_beta0, tangent_intercept_arr, tangent_slope_arr = sm.MultivariateLinearizedGeodesicRegression_Kendall2D_BottomUp( shape_age_list_a, shape_list_a, cov_int_list_a, cov_slope_list_a, max_iter=10, verbose=True )

# #####################################################################################################################################
# # Model : y = beta0 + beta1 * s + beta2 * hr + beta3 * lr + beta4 * autism + beta5 * autismSpec * beta6 * healthy 					#
# #                ( gamma0 + gamma1 * s + gamma2 * hr + gamma3 * lr + gamma4 * autism + gamma5 * autismSpec + gamma 6 * healthy )t   #
# #####################################################################################################################################
nParam = 7

# est_beta0.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta0.pt')

# # beta1 * s
# est_beta1 = tangent_intercept_arr[ 0 ]
# est_beta1.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta1.tVec')


# # beta2 * hr
# est_beta2 = tangent_intercept_arr[ 1 ]
# est_beta2.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta2.tVec')

# # beta3 * lr
# est_beta3 = tangent_intercept_arr[ 2 ]
# est_beta3.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta3.tVec')

# # beta4 * autism
# est_beta4 = tangent_intercept_arr[ 3 ]
# est_beta4.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta4.tVec')

# # beta5 * autismSpec
# est_beta5 = tangent_intercept_arr[ 4 ]
# est_beta5.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta5.tVec')

# # beta6 * healthy
# est_beta6 = tangent_intercept_arr[ 5 ]
# est_beta6.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta6.tVec')

# # gamma0 t
# est_gamma0 = tangent_slope_arr[ 6 ]
# est_gamma0.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma0.tVec')

# # gamma1 * s t
# est_gamma1 = tangent_slope_arr[ 0 ]
# est_gamma1.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma1.tVec')

# # gamma2 * hr t
# est_gamma2 = tangent_slope_arr[ 1 ]
# est_gamma2.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma2.tVec')

# # gamma3 * lr t
# est_gamma3 = tangent_slope_arr[ 2 ]
# est_gamma3.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma3.tVec')

# # gamma4 * autism t
# est_gamma4 = tangent_slope_arr[ 3 ]
# est_gamma4.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma4.tVec')

# # gamma5 * autismSpec t
# est_gamma5 = tangent_slope_arr[ 4 ]
# est_gamma5.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma5.tVec')

# # gamma6 * healthy t
# est_gamma6 = tangent_slope_arr[ 5 ]
# est_gamma6.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma6.tVec')

print( "# of Subjects" )
print( len( shape_list_a ) )

# Read 
est_beta0 = manifolds.kendall2D( nPt )
est_beta0.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta0.pt' )

est_beta1 = manifolds.kendall2D_tVec( nPt )
est_beta1.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta1.tVec' )

est_beta2 = manifolds.kendall2D_tVec( nPt )
est_beta2.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta2.tVec' )

est_beta3 = manifolds.kendall2D_tVec( nPt )
est_beta3.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta3.tVec' )

est_beta4 = manifolds.kendall2D_tVec( nPt )
est_beta4.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta4.tVec' )

est_beta5 = manifolds.kendall2D_tVec( nPt )
est_beta5.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta5.tVec' )

est_beta6 = manifolds.kendall2D_tVec( nPt )
est_beta6.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_beta6.tVec' )


est_gamma0 = manifolds.kendall2D_tVec( nPt )
est_gamma0.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma0.tVec' )

est_gamma1 = manifolds.kendall2D_tVec( nPt )
est_gamma1.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma1.tVec' )

est_gamma2 = manifolds.kendall2D_tVec( nPt )
est_gamma2.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma2.tVec' )

est_gamma3 = manifolds.kendall2D_tVec( nPt )
est_gamma3.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma3.tVec' )

est_gamma4 = manifolds.kendall2D_tVec( nPt )
est_gamma4.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma4.tVec' )

est_gamma5 = manifolds.kendall2D_tVec( nPt )
est_gamma5.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma5.tVec' )

est_gamma6 = manifolds.kendall2D_tVec( nPt )
est_gamma6.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/est_gamma6.tVec' )

# Calculate R2 and adjusted R2
sqDist_MLMG_sum = 0
sqVar_sum = 0

# mean_shape = sm.FrechetMean( shape_list_all )
# mean_shape.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/mean_shape.pt' )

# Read
mean_shape = manifolds.kendall2D( nPt )
mean_shape.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/MLMG/mean_shape.pt' )

for i in range( len( shape_list_all ) ):
	shape_i = shape_list_all[ i ]
	t_i = t_list_all[ i ]

	s_i = s_list_all[ i ]
	hr_i = hr_list_all[ i ]
	lr_i = lr_list_all[ i ]
	autism_i = autism_list_all[ i ]
	autismSpec_i = autismspec_list_all[ i ]
	healthy_i = none_list_all[ i ]

	beta_i = manifolds.kendall2D_tVec( nPt )

	for k in range( 2 ):
		for j in range( nPt ):
			beta_i.tVector[ k, j ] = ( est_beta1.tVector[ k, j ] * s_i + est_beta2.tVector[ k, j ] * hr_i + est_beta3.tVector[ k, j ] * lr_i + est_beta4.tVector[ k, j ] * autism_i  + est_beta5.tVector[ k, j ] * autismSpec_i + est_beta6.tVector[ k, j ] * healthy_i )

	p1_i = est_beta0.ExponentialMap( beta_i )

	gamma0_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma0 )
	gamma1_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma1 )
	gamma2_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma2 )
	gamma3_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma3 )
	gamma4_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma4 )
	gamma5_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma5 )
	gamma6_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma6 )

	gamma_i = manifolds.kendall2D_tVec( nPt )
	for k in range( 2 ):
		for j in range( nPt ):
			gamma_i.tVector[ k, j ] = ( gamma0_p1_i.tVector[ k, j ] + gamma1_p1_i.tVector[ k, j ] * s_i + gamma2_p1_i.tVector[ k, j ] * hr_i + gamma3_p1_i.tVector[ k, j ] * lr_i + gamma4_p1_i.tVector[ k, j ] * autism_i  + gamma5_p1_i.tVector[ k, j ] * autismSpec_i + gamma6_p1_i.tVector[ k, j ] * healthy_i ) * t_i

	est_p_i = p1_i.ExponentialMap( gamma_i )

	tVec_est_p_n_to_p_n = est_p_i.LogMap( shape_i ) 

	sqDist_n = tVec_est_p_n_to_p_n.normSquared() 

	sqDist_MLMG_sum += sqDist_n

	tVec_mean_to_p_n = mean_shape.LogMap( shape_i ) 

	sqVar_n = tVec_mean_to_p_n.normSquared()

	sqVar_sum += sqVar_n

nObs_a = len( shape_list_all )

R2_MLMG = 1 - ( sqDist_MLMG_sum / sqVar_sum )

nParam_MLMG = 7 

adjustedR2_MLMG = R2_MLMG - ( ( 1 - R2_MLMG ) * nParam_MLMG / ( nObs_a - nParam_MLMG - 1 ) )

# Geodesic Regression
nParam_geo = 1
sqDist_SG_sum = 0 

intercept, slope = sm.LinearizedGeodesicRegression( t_list_all, shape_list_all, max_iter=1000, stepSize=0.0005, step_tol=1e-12, verbose=True )

for i in range( len( shape_list_all ) ):
	p_i = shape_list_all[ i ]
	t_i = t_list_all[ i ]

	slope_t_i = slope.ScalarMultiply( t_i )
	est_p_i = intercept.ExponentialMap( slope_t_i )

	tVec_est_p_i_to_p_i = est_p_i.LogMap( p_i )

	sqDist_i = tVec_est_p_i_to_p_i.normSquared() 

	sqDist_SG_sum += sqDist_i

R2_SG = 1 - ( sqDist_SG_sum / sqVar_sum )


adjustedR2_SG = R2_SG - ( ( 1 - R2_SG ) * nParam_geo / ( nObs_a - nParam_geo - 1 ) )


# MultiLv Single Geodesic Model
est_beta0_MLSG, tangent_intercept_arr_MLSG, tangent_slope_arr_MLSG = sm.MultivariateLinearizedGeodesicRegression_Kendall2D_BottomUp( shape_age_list_a, shape_list_a, cov_int_list_MLSG, cov_slope_list_MLSG, max_iter=10, verbose=True )

slope_MLSG = tangent_slope_arr_MLSG[ 1 ] 

nParam_MLSG = 1
sqDist_MLSG_sum = 0 
sqVar_MLSG_sum = 0

for i in range( len( shape_list_all ) ):
	p_i = shape_list_all[ i ]
	t_i = t_list_all[ i ]

	slope_t_i = slope_MLSG.ScalarMultiply( t_i )
	est_p_i = est_beta0_MLSG.ExponentialMap( slope_t_i )

	tVec_est_p_i_to_p_i = est_p_i.LogMap( p_i )

	sqDist_i = tVec_est_p_i_to_p_i.normSquared() 

	sqDist_MLSG_sum += sqDist_i

R2_MLSG = 1 - ( sqDist_MLSG_sum / sqVar_sum )
adjustedR2_MLSG = R2_MLSG - ( ( 1 - R2_MLSG ) * nParam_MLSG / ( nObs_a - nParam_MLSG - 1 ) )

print( "MLMG" )
print( R2_MLMG )
print( adjustedR2_MLMG )

print( "MLSG" )
print( R2_MLSG )
print( adjustedR2_MLSG )	

print( "SG" )
print( R2_SG )
print( adjustedR2_SG )	

'''
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

'''