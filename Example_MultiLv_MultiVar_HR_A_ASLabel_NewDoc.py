#####################################################################################################################################
#####################################################################################################################################
# Example - Geodesic Regression																					  					#		
# Manifold - Kendall 2D Manifold																				  					#
# y : Corpus Callosum Shapes - 2D Kendall Shape Space				 											  					#
# t : Age  																										  					#	 
# Covariates : sex, autism, autismSpec, ADOS 			 														  					#
# Model : y = beta0 + beta1 * s + beta2 * autism + beta3 * ados 												 					#
#                ( gamma0 + gamma1 * s + gamma2 * autism + gamma3 * ados )t   													    #
#####################################################################################################################################
#####################################################################################################################################

import manifolds 
import numpy as np
import StatsModel as sm

import matplotlib.pyplot as plt
# Visualization
import vtk

import pandas as pd

from colour import Color 

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
visit_label_list = []

for i in range( len( subj_csv_list[ 0 ] ) ):
	subj_list.append( subj_csv_list[ 0 ][ i ][ :6 ] )
	visit_label_list.append( subj_csv_list[ 0 ][ i ][ 7:10 ] )
	print( subj_csv_list[ 0 ][ i ][ 7:10 ] )

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

# # Read Autism Diagnosed 
# autism_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/autism.csv", header=None )
# autism_list = []
# for i in range( len( autism_csv_list[ 0 ] ) ):
# 	autism_list.append( int( autism_csv_list[ 0 ][ i ] ) )

# # Read Autism Spectrum Diagnosed
# autismspec_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/autismspec.csv", header=None )
# autismspec_list = []

# for i in range( len( autismspec_csv_list[ 0 ] ) ):
# 	autismspec_list.append( int( autismspec_csv_list[ 0 ][ i ] ) )

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

# # Read Healthy 
# none_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/none.csv", header=None )
# none_list = []

# for i in range( len( none_csv_list[ 0 ] ) ):
# 	none_list.append( int( none_csv_list[ 0 ][ i ] ) )

# Read Time
age_csv_list = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/age.csv", header=None )
age_list = [] 

for i in range( len( age_csv_list[ 0 ] ) ):
	age_list.append( float( age_csv_list[ 0 ][ i ] ) )

# Exception List 
except_list = np.asarray( [ 246, 303, 356, 376, 636, 680, 934, 961, 998, 1005, 1033, 1039 ] )
except_list = except_list - 1

# All information CSV
info_dict = pd.read_csv( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/data-2019-02-27T18_18_32.075Z_with_ADOSseverity.csv" )

# Kendall Shape Space 
shape_list = []
shape_age_list = []
shape_subj_list = subj_set 
shape_visit_label_list = []
shape_ADOS_list = [] 

shape_nObs_list = []

cov_int_list = []
cov_slope_list = []

subj_s_list = []
subj_autism_list = []
subj_autism_spec_list = []
subj_none_list = []

subj_hr_list = []
subj_lr_list = [] 


for i in range( len( subj_set ) ):
	shape_list.append( [] )
	shape_visit_label_list.append( [] )
	shape_ADOS_list.append( [] )	

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
# Covariates : sex, autism, ADOS			            #
#########################################################
cnt_noDiag = 0

cnt_noRisk = 0 

cnt_Less2 = 0

cnt_noSex = 0

cnt_no24 = 0

cnt_noADOS = 0

cnt_none = 0

cnt_autism = 0

cnt_autismS = 0

for i in range( nObs ):
	# Exception list
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

	# if i in autism_list:
	# 	autism_i = 1
	# else:
	# 	autism_i = 0

	# if i in autismspec_list:
	# 	autismspec_i = 1
	# else:
	# 	autismspec_i = 0

	# if i in hr_list:
	# 	hr_i = 1
	# else:
	# 	hr_i = 0

	# if i in lr_list:
	# 	lr_i = 1
	# else:
	# 	lr_i = 0

	# if i in none_list:
	# 	none_i = 1
	# else:
	# 	none_i = 0

	# Find Subject Index
	subj_i = subj_list[ i ]

	for j in range( len( subj_set ) ):
		if subj_i == subj_set[ j ]:
			# if autismspec_i == 1:
			# 	autism_i = 1

			# if autism_i == 0:
			# 	none_i = 1				

			row_ij = info_dict.loc[ ( info_dict['demographics,CandID'] == int( subj_set[ j ] ) ) & ( info_dict[ 'demographics,Visit_label' ] == visit_label_list[ i ] ) ]

			# ADOS Score
			if row_ij[ 'ADOS_Derived,severity_score_lookup' ][ row_ij.index[ 0 ] ] == '.':				
				shape_ADOS_list[ j ].append( 0 )
			elif np.isnan( float( row_ij[ 'ADOS_Derived,severity_score_lookup' ][ row_ij.index[ 0 ] ] ) ):
				shape_ADOS_list[ j ].append( 0 )
			else:
				shape_ADOS_list[ j ].append( int( row_ij[ 'ADOS_Derived,severity_score_lookup' ][ row_ij.index[ 0 ] ] ) )

			row_i2m = info_dict.loc[ ( info_dict['demographics,CandID'] == int( subj_set[ j ] ) ) & ( info_dict[ 'demographics,Visit_label' ] == "V24" ) ] 			

			if row_i2m.empty:
				cnt_no24 += 1 
				continue

			# Autism Label
			if row_i2m[ 'ADOS_Derived,severity_score_lookup' ][ row_i2m.index[ 0 ] ] == ".":
				cnt_noADOS += 1
				continue
			elif np.isnan( float( row_i2m[ 'ADOS_Derived,severity_score_lookup' ][ row_i2m.index[ 0 ] ] ) ):
				cnt_noADOS += 1
				continue

			if row_i2m[ 'ADOS_Derived,ADOS_classification' ][ row_i2m.index[ 0 ] ] == "autism":
				cnt_autism += 1 
				autism_i = 1
				autismspec_i = 0
				none_i = 0
			elif row_i2m[ 'ADOS_Derived,ADOS_classification' ][ row_i2m.index[ 0 ] ] == "autism spectrum":
				cnt_autismS += 1 
				autism_i = 0
				autismspec_i = 1
				none_i = 0
			elif row_i2m[ 'ADOS_Derived,ADOS_classification' ][ row_i2m.index[ 0 ] ] == "none":
				cnt_none += 1 
				none_i = 1
				autism_i = 0
				autismspec_i = 0
				# continue
			else:
				cnt_noDiag += 1 
				autism_i = 0
				none_i = 0
				autismspec_i = 0
				continue

			if row_i2m[ 'demographics,Risk' ][ row_i2m.index[ 0 ] ] == "LR": 
				lr_i = 1 
				hr_i = 0
				continue
			elif  row_i2m[ 'demographics,Risk' ][ row_i2m.index[ 0 ] ] == "HR": 
				hr_i = 1 
				lr_i = 0
			else:
				cnt_noRisk += 1 
				hr_i = 0
				lr_i = 0
				continue 

			shape_list[ j ].append( shape_i )
			shape_age_list[ j ].append( float( age_list[ i ] ) )
			shape_visit_label_list[ j ].append( visit_label_list[ i ] )

			shape_nObs_list[ j ] = shape_nObs_list[ j ] + 1 

			if shape_nObs_list[ j ] == 1:
				cov_int_list[ j ] = [ s_i ]
				cov_slope_list[ j ] = [ s_i ]

				subj_s_list[ j ] = s_i
				subj_autism_list[ j ] = autism_i
				subj_autism_spec_list[ j ] = autismspec_i
				subj_hr_list[ j ] = hr_i
				subj_lr_list[ j ] = lr_i
				subj_none_list[ j ] = none_i


# print( len( cov_int_list ) )
# print( len( cov_slope_list ) )
# print( len( shape_list ) )

# print( shape_nObs_list )
# print( np.sum( shape_nObs_list ) )

# print( cov_int_list[ 0 ] )
# print( cov_slope_list[ 0 ] )


shape_nObs_list_a = [] 
shape_list_a = []
shape_age_list_a = []
shape_subj_list_a = []
shape_visit_label_list_a = []
shape_ADOS_list_a = []


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
# ADOS
ADOS_list_all = []

# For Single Geodesic Model
cov_int_list_MLSG = []
cov_slope_list_MLSG = []


cnt_check = 0

cnt_noDiag2 = 0 

cnt_Less2_0 = 0

cnt_subjHealthy = 0

cnt_subjAS = 0

cnt_subjMale = 0

cnt_subjFemale = 0


for i in range( len( shape_nObs_list ) ):
	nObs_i = shape_nObs_list[ i ] 

	if nObs_i > 1:
		# if subj_hr_list[ i ] == subj_lr_list[ i ]:
		# 	continue

		if ( subj_none_list[ i ] == subj_autism_list[ i ] and subj_none_list[ i ] == subj_autism_spec_list[ i ] ):
			cnt_noDiag2 += 1
			continue
	
		# if subj_s_list[ i ] == 1:
		# 	continue

		# if subj_autism_list[ i ] == 1:
		# 	continue

		if not 'V24' in shape_visit_label_list[ i ]:
			cnt_check += 1
			continue

		# if subj_none_list[ i ] == 1:
		# 	# cnt_check += 1 
		# 	continue

		# if shape_ADOS_list[ i ][ -1 ] == 0:
		# 	continue

		shape_list_a.append( shape_list[ i ] )
		shape_age_list_a.append( shape_age_list[ i ] )
		shape_subj_list_a.append( shape_subj_list[ i ] )
		shape_visit_label_list_a.append( shape_visit_label_list[ i ] )

		subj_s_list_a.append( subj_s_list[ i ] )
		subj_autism_list_a.append( subj_autism_list[ i ] )
		subj_autism_spec_list_a.append( subj_autism_spec_list[ i ] )
		subj_hr_list_a.append( subj_hr_list[ i ] )
		subj_lr_list_a.append( subj_lr_list[ i ] )
		subj_none_list_a.append( subj_none_list[ i ] )
		shape_ADOS_list_a.append( shape_ADOS_list[ i ][ -1 ] )

		# Covariates Add
		cov_int_list[ i ].append( subj_autism_list[ i ] )
		cov_slope_list[ i ].append( subj_autism_list[ i ] )

		cov_int_list[ i ].append( subj_autism_spec_list[ i ] )
		cov_slope_list[ i ].append( subj_autism_spec_list[ i ] )

		
		cov_int_list_a.append( cov_int_list[ i ] )
		cov_slope_list_a.append( cov_slope_list[ i ] )

		cov_int_list_MLSG.append( [ 0 ] )
		cov_slope_list_MLSG.append( [ 0 ] )

		shape_nObs_list_a.append( shape_nObs_list[ i ] )

		if subj_none_list[ i ]  == 1 :
			cnt_subjHealthy += 1 
		else:
			cnt_subjAS += 1 

		if subj_s_list[ i ] == 1:
			cnt_subjMale += 1 
		else:
			cnt_subjFemale += 1 

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
			ADOS_list_all.append( shape_ADOS_list[ i ][ -1 ] )
	elif nObs_i == 1:
		cnt_Less2 += 1
	else:
		cnt_Less2_0 += 1 


data_list_pd = pd.DataFrame( { 'SubjectID': subj_list_all, 'Age':t_list_all, 'Sex':s_list_all, 'HR':hr_list_all, 'LR':lr_list_all, 'Autism':autism_list_all, 'AutismSpec':autismspec_list_all, 'None':none_list_all, 'ADOS':ADOS_list_all } )
data_list_pd.to_csv( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/Polished_Data_List.csv' )

print( len( shape_list_a ) )
print( len( shape_list_all ) )

min_age = np.min( t_list_all )
max_age = np.max( t_list_all )

t_list_all = np.divide( np.subtract( t_list_all, min_age ), max_age - min_age )
max_norm_age = np.max( t_list_all )
min_norm_age = np.min( t_list_all )

for i in range( len( shape_age_list_a ) ):
	for j in range( len( shape_age_list_a[ i ] ) ):
		shape_age_list_a[ i ][ j ] = ( shape_age_list_a[ i ][ j ] - min_age ) / ( max_age - min_age )

est_beta0, tangent_intercept_arr, tangent_slope_arr = sm.MultivariateLinearizedGeodesicRegression_Kendall2D_BottomUp( shape_age_list_a, shape_list_a, cov_int_list_a, cov_slope_list_a, max_iter=10, verbose=True )

#####################################################################################################################################
# Model : y = beta0 + beta1 * s + beta2 * hr + beta3 * autism +beta4 * autismspec								 					#
#                ( gamma0 + gamma1 * s + gamma2 * hr + gamma3 * autism + gamma4 * autismspec )t									    #
#####################################################################################################################################
nParam = 6

est_beta0.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_beta0.pt')

# beta1 * s
est_beta1 = tangent_intercept_arr[ 0 ]
est_beta1.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_beta1.tVec')

# beta2 * a
est_beta2 = tangent_intercept_arr[ 1 ]
est_beta2.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_beta2.tVec')

# beta3 * as
est_beta3 = tangent_intercept_arr[ 2 ]
est_beta3.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_beta3.tVec')

# gamma0 t
est_gamma0 = tangent_slope_arr[ 3 ]
est_gamma0.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma0.tVec')

# gamma1 * s t
est_gamma1 = tangent_slope_arr[ 0 ]
est_gamma1.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma1.tVec')

# gamma2 * a t
est_gamma2 = tangent_slope_arr[ 1 ]
est_gamma2.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma2.tVec')

# gamma3 * as t
est_gamma3 = tangent_slope_arr[ 2 ]
est_gamma3.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma3.tVec')


# # Read 
# est_beta0 = manifolds.kendall2D( nPt )
# est_beta0.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_beta0.pt' )

# est_beta1 = manifolds.kendall2D_tVec( nPt )
# est_beta1.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_beta1.tVec' )

# est_beta2 = manifolds.kendall2D_tVec( nPt )
# est_beta2.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_beta2.tVec' )

# est_gamma0 = manifolds.kendall2D_tVec( nPt )
# est_gamma0.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma0.tVec' )

# est_gamma1 = manifolds.kendall2D_tVec( nPt )
# est_gamma1.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma1.tVec' )

# est_gamma2 = manifolds.kendall2D_tVec( nPt )
# est_gamma2.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma2.tVec' )

# est_gamma3 = manifolds.kendall2D_tVec( nPt )
# est_gamma3.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/est_gamma3.tVec' )

# Calculate R2 and adjusted R2
sqDist_MLMG_sum = 0
sqVar_sum = 0

mean_shape = sm.FrechetMean( shape_list_all, maxIter=1000 )
mean_shape.Write( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/mean_shape.pt' )

# # Read
# mean_shape = manifolds.kendall2D( nPt )
# mean_shape.Read( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/mean_shape.pt' )

for i in range( len( shape_list_all ) ):
	shape_i = shape_list_all[ i ]
	t_i = t_list_all[ i ]

	s_i = s_list_all[ i ]
	hr_i = hr_list_all[ i ]
	lr_i = lr_list_all[ i ]
	autism_i = autism_list_all[ i ]
	autismSpec_i = autismspec_list_all[ i ]
	healthy_i = none_list_all[ i ]
	ADOS_i = ADOS_list_all[ i ]

	beta_i = manifolds.kendall2D_tVec( nPt )

	for k in range( 2 ):
		for j in range( nPt ):
			beta_i.tVector[ k, j ] = ( est_beta1.tVector[ k, j ] * s_i + est_beta2.tVector[ k, j ] * autism_i + est_beta3.tVector[ k, j ] * autismSpec_i )

	p1_i = est_beta0.ExponentialMap( beta_i )

	gamma0_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma0 )
	gamma1_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma1 )
	gamma2_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma2 )
	gamma3_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma3 )

	gamma_i = manifolds.kendall2D_tVec( nPt )
	for k in range( 2 ):
		for j in range( nPt ):
			gamma_i.tVector[ k, j ] = ( gamma0_p1_i.tVector[ k, j ] + gamma1_p1_i.tVector[ k, j ] * s_i + gamma2_p1_i.tVector[ k, j ] * autism_i + gamma3_p1_i.tVector[ k, j ] * autismSpec_i ) * t_i

	est_p_i = p1_i.ExponentialMap( gamma_i )

	tVec_est_p_n_to_p_n = est_p_i.LogMap( shape_i ) 

	sqDist_n = tVec_est_p_n_to_p_n.normSquared() 

	sqDist_MLMG_sum += sqDist_n

	tVec_mean_to_p_n = mean_shape.LogMap( shape_i ) 

	sqVar_n = tVec_mean_to_p_n.normSquared()

	sqVar_sum += sqVar_n

nObs_a = len( shape_list_all )

R2_MLMG = 1 - ( sqDist_MLMG_sum / sqVar_sum )

nParam_MLMG = 6 
adjustedR2_MLMG = R2_MLMG - ( ( 1 - R2_MLMG ) * nParam_MLMG / ( nObs_a - nParam_MLMG - 1 ) )

# Geodesic Regression
nParam_geo = 2
sqDist_SG_sum = 0 

intercept, slope = sm.LinearizedGeodesicRegression( t_list_all, shape_list_all, max_iter=1000, stepSize=0.0005, step_tol=1e-12, verbose=False )

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
est_beta0_MLSG, tangent_intercept_arr_MLSG, tangent_slope_arr_MLSG = sm.MultivariateLinearizedGeodesicRegression_Kendall2D_BottomUp( shape_age_list_a, shape_list_a, cov_int_list_MLSG, cov_slope_list_MLSG, max_iter=10, verbose=False )
slope_MLSG = tangent_slope_arr_MLSG[ -1 ] 

nParam_MLSG = 2
sqDist_MLSG_sum = 0 

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

for s in range( 2 ):
	for a in range( 2 ):
		for i_as in range( 2 ):
			beta_i = manifolds.kendall2D_tVec( nPt )

			for k in range( 2 ):
				for j in range( nPt ):
					beta_i.tVector[ k, j ] = ( est_beta1.tVector[ k, j ] * s + est_beta2.tVector[ k, j ] * a + est_beta3.tVector[ k, j ] * i_as )

			p1_i = est_beta0.ExponentialMap( beta_i )

			gamma0_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma0 )
			gamma1_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma1 )
			gamma2_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma2 )
			gamma3_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma3 )

			gamma_i = manifolds.kendall2D_tVec( nPt )
			for k in range( 2 ):
				for j in range( nPt ):
					gamma_i.tVector[ k, j ] = ( gamma0_p1_i.tVector[ k, j ] + gamma1_p1_i.tVector[ k, j ] * s + gamma2_p1_i.tVector[ k, j ] * a + gamma3_p1_i.tVector[ k, j ] * i_as )
			
			# plt.figure( figsize =( 20, 10 ) )
			fig = plt.figure()
			ax = fig.add_subplot( 1, 1, 1 )
			plt.xlim( -0.2, 0.2 )
			plt.ylim( -0.1, 0.1 )

			ax.spines[ 'left' ].set_position( 'center' )
			ax.spines[ 'bottom' ].set_position( 'center' )

			ax.spines[ 'right' ].set_color( 'none' )
			ax.spines[ 'top' ].set_color( 'none' )

			x_tick_arr = np.arange( -0.2, 0.21, 0.05 )
			x_tick_arr = x_tick_arr[ np.nonzero( x_tick_arr ) ]

			y_tick_arr = np.arange( -0.1, 0.11, 0.05 ) 
			y_tick_arr = y_tick_arr[ np.nonzero( y_tick_arr ) ]

			plt.xticks( x_tick_arr )
			plt.yticks( y_tick_arr )
			plt.axis('off')

			plt.tight_layout()

			plt.gca().set_aspect('equal', adjustable='box')

			green = Color( "blue" )
			colors = list( green.range_to( Color( "yellow" ), 11 ) )

			for i_t in range( 11 ):
				t = float( i_t ) / 10

				gamma_i_t = gamma_i.ScalarMultiply( t )

				p_t = p1_i.ExponentialMap( gamma_i_t )

				x_t = []
				y_t = [] 

				for i_p in range( nPt ):
					x_t.append( p_t.pt[ 0, i_p ] )
					y_t.append( p_t.pt[ 1, i_p ] )

				x_t.append( p_t.pt[ 0, 0 ] )
				y_t.append( p_t.pt[ 1, 0 ] )


				plt.plot( x_t, y_t, color=colors[ i_t ].rgb, alpha=0.6 )

			plt.savefig( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/results_s' + str( s ) + 'a' + str( a ) + 'as' + str( i_as ) + '.png' )
			plt.close()

green = Color( "blue" )
colors_2 = list( green.range_to( Color( "yellow" ), 11 ) ) 

for i_t in range( 11 ):
	t = float( i_t ) / 10

	for s in range( 2 ):
		# plt.figure( figsize =( 20, 10 ) )
		fig = plt.figure()
		ax = fig.add_subplot( 1, 1, 1 )
		plt.xlim( -0.2, 0.2 )
		plt.ylim( -0.1, 0.1 )

		ax.spines[ 'left' ].set_position( 'center' )
		ax.spines[ 'bottom' ].set_position( 'center' )

		ax.spines[ 'right' ].set_color( 'none' )
		ax.spines[ 'top' ].set_color( 'none' )

		x_tick_arr = np.arange( -0.2, 0.21, 0.05 )
		x_tick_arr = x_tick_arr[ np.nonzero( x_tick_arr ) ]

		y_tick_arr = np.arange( -0.1, 0.11, 0.05 ) 
		y_tick_arr = y_tick_arr[ np.nonzero( y_tick_arr ) ]

		plt.xticks( x_tick_arr )
		plt.yticks( y_tick_arr )
		plt.axis('off')

		plt.tight_layout()

		plt.gca().set_aspect('equal', adjustable='box')

		for i_as in range( 2 ):
			for a in range( 2 ):
				beta_i = manifolds.kendall2D_tVec( nPt )

				for k in range( 2 ):
					for j in range( nPt ):
						beta_i.tVector[ k, j ] = ( est_beta1.tVector[ k, j ] * s + est_beta2.tVector[ k, j ] * a + est_beta3.tVector[ k, j ] * i_as )

				p1_i = est_beta0.ExponentialMap( beta_i )

				gamma0_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma0 )
				gamma1_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma1 )
				gamma2_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma2 )
				gamma3_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_i, est_gamma3 )

				gamma_i = manifolds.kendall2D_tVec( nPt )
				for k in range( 2 ):
					for j in range( nPt ):
						gamma_i.tVector[ k, j ] = ( gamma0_p1_i.tVector[ k, j ] + gamma1_p1_i.tVector[ k, j ] * s + gamma2_p1_i.tVector[ k, j ] * a + gamma3_p1_i.tVector[ k, j ] * i_as )
			
				gamma_i_t = gamma_i.ScalarMultiply( t )

				p_t = p1_i.ExponentialMap( gamma_i_t )

				x_t = []
				y_t = []

				for i_p in range( nPt ):
					x_t.append( p_t.pt[ 0, i_p ] )
					y_t.append( p_t.pt[ 1, i_p ] )

				x_t.append( p_t.pt[ 0, 0 ] )
				y_t.append( p_t.pt[ 1, 0 ] )

				plt.plot( x_t, y_t, color=colors_2[ a ].rgb, alpha=0.6, label=("s" + str( s ) + "a" + str( a ) + str( "i_as " ) ) )
				# plt.legend()

			plt.savefig( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/results_' + 's'+ str( s ) + 't' + str( i_t ) + '.png' )
			plt.close()

# Subject Specific Trajectory
# Individual geodesic estimation
for i in range( len( shape_age_list_a ) ):
	age_list_i = shape_age_list_a[ i ] 
	shape_list_i = shape_list_a[ i ] 

	p_i, v_i = sm.LinearizedGeodesicRegression( age_list_i, shape_list_i, verbose=False )

	for j in range( len( shape_list_i ) ):
		age_ij = shape_age_list_a[ i ][ j ]
		s_ij = subj_s_list_a[ i ]
		autism_ij = subj_autism_list_a[ i ]
		autismSpec_ij = subj_autism_spec_list_a[ i ]		
		ados_ij = shape_ADOS_list_a[ i ]

		subj_ij = shape_subj_list_a[ i ]
		visit_label_ij = shape_visit_label_list_a[ i ][ j ]

		fig = plt.figure()
		ax = fig.add_subplot( 1, 1, 1 )
		plt.xlim( -0.2, 0.2 )
		plt.ylim( -0.1, 0.1 )

		ax.spines[ 'left' ].set_position( 'center' )
		ax.spines[ 'bottom' ].set_position( 'center' )

		ax.spines[ 'right' ].set_color( 'none' )
		ax.spines[ 'top' ].set_color( 'none' )

		x_tick_arr = np.arange( -0.2, 0.21, 0.05 )
		x_tick_arr = x_tick_arr[ np.nonzero( x_tick_arr ) ]

		y_tick_arr = np.arange( -0.1, 0.11, 0.05 ) 
		y_tick_arr = y_tick_arr[ np.nonzero( y_tick_arr ) ]

		plt.xticks( x_tick_arr )
		plt.yticks( y_tick_arr )
		plt.axis('off')
		plt.tight_layout()

		plt.gca().set_aspect('equal', adjustable='box')

		# Observation
		shape_obs_ij = shape_list_a[ i ][ j ]
		shape_obs_ij_x = []
		shape_obs_ij_y = []

		for i_p in range( nPt ):
			shape_obs_ij_x.append( shape_obs_ij.pt[ 0, i_p ] )
			shape_obs_ij_y.append( shape_obs_ij.pt[ 1, i_p ] )

		shape_obs_ij_x.append( shape_obs_ij_x[ 0 ] )
		shape_obs_ij_y.append( shape_obs_ij_y[ 0 ] )

		plt.plot( shape_obs_ij_x, shape_obs_ij_y, color="black", label="Obs", lw=5 )

		# Subject-wise estimation - Mixed Effects
		shape_ind_est_ij = p_i.ExponentialMap( v_i.ScalarMultiply( age_ij ) )
		shape_ind_est_ij_x = []
		shape_ind_est_ij_y = []

		for i_p in range( nPt ):
			shape_ind_est_ij_x.append( shape_ind_est_ij.pt[ 0, i_p ] )
			shape_ind_est_ij_y.append( shape_ind_est_ij.pt[ 1, i_p ] )

		shape_ind_est_ij_x.append( shape_ind_est_ij_x[ 0 ] )
		shape_ind_est_ij_y.append( shape_ind_est_ij_y[ 0 ] )

		# plt.plot( shape_ind_est_ij_x, shape_ind_est_ij_y, color="cyan", label="Subj-Spec" )

		# Single geodesic regression estimation 
		shape_sg_est_ij = intercept.ExponentialMap( slope.ScalarMultiply( age_ij ) )
		shape_sg_est_ij_x = []
		shape_sg_est_ij_y = []

		for i_p in range( nPt ):
			shape_sg_est_ij_x.append( shape_sg_est_ij.pt[ 0, i_p ] )
			shape_sg_est_ij_y.append( shape_sg_est_ij.pt[ 1, i_p ] )

		shape_sg_est_ij_x.append( shape_sg_est_ij_x[ 0 ] )
		shape_sg_est_ij_y.append( shape_sg_est_ij_y[ 0 ] )

		plt.plot( shape_sg_est_ij_x, shape_sg_est_ij_y, color="red", label="SG", lw=5 )

		# Multi-Level single geodeisc regression estimation
		shape_mlsg_est_ij = est_beta0_MLSG.ExponentialMap( slope_MLSG.ScalarMultiply( age_ij ) )
		shape_mlsg_est_ij_x = []
		shape_mlsg_est_ij_y = []

		for i_p in range( nPt ):
			shape_mlsg_est_ij_x.append( shape_mlsg_est_ij.pt[ 0, i_p ] )
			shape_mlsg_est_ij_y.append( shape_mlsg_est_ij.pt[ 1, i_p ] )

		shape_mlsg_est_ij_x.append( shape_mlsg_est_ij_x[ 0 ] )
		shape_mlsg_est_ij_y.append( shape_mlsg_est_ij_y[ 0 ] )

		plt.plot( shape_mlsg_est_ij_x, shape_mlsg_est_ij_y, color="blue", label="MLSG", lw=2 )

		# MLMG - Fixed Effects
		beta_ij = manifolds.kendall2D_tVec( nPt )

		for kk in range( 2 ):
			for jj in range( nPt ):
				beta_ij.tVector[ kk, jj ] = ( est_beta1.tVector[ kk, jj ] * s_ij + est_beta2.tVector[ kk, jj ] * autism_ij + est_beta3.tVector[ kk, jj ] * autismSpec_ij )

		p1_ij = est_beta0.ExponentialMap( beta_ij )

		gamma0_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_ij, est_gamma0 )
		gamma1_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_ij, est_gamma1 )
		gamma2_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_ij, est_gamma2 )
		gamma3_p1_i = est_beta0.ParallelTranslateAtoB( est_beta0, p1_ij, est_gamma3 )

		gamma_ij = manifolds.kendall2D_tVec( nPt )
		for kk in range( 2 ):
			for jj in range( nPt ):
				gamma_ij.tVector[ kk, jj ] = ( gamma0_p1_i.tVector[ kk, jj ] + gamma1_p1_i.tVector[ kk, jj ] * s_ij + gamma2_p1_i.tVector[ kk, jj ] * autism_ij + gamma3_p1_i.tVector[ kk, jj ] * autismSpec_ij ) * age_ij

		shape_mlmg_est_ij = p1_ij.ExponentialMap( gamma_ij )
		shape_mlmg_est_ij_x = []
		shape_mlmg_est_ij_y = []

		for i_p in range( nPt ):
			shape_mlmg_est_ij_x.append( shape_mlmg_est_ij.pt[ 0, i_p ] )
			shape_mlmg_est_ij_y.append( shape_mlmg_est_ij.pt[ 1, i_p ] )

		shape_mlmg_est_ij_x.append( shape_mlmg_est_ij_x[ 0 ] )
		shape_mlmg_est_ij_y.append( shape_mlmg_est_ij_y[ 0 ] )

		plt.plot( shape_mlmg_est_ij_x, shape_mlmg_est_ij_y, color="cyan", label="MLMG", lw=2 )
		plt.legend()

		plt.savefig( '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/ACE_IBIS_CC/Data/HMG_HR_AAS_New/Subjects/' + subj_ij + "_" + visit_label_ij + "_s" + str( s_ij ) + "a" + str( autism_ij ) + "as" + str( autismSpec_ij ) + '.png' )
		plt.close()


print( "No. Obs " ) 
print( nObs )

print( "No Sex" )
print( cnt_noSex )

print( "less than 2" )
print( cnt_Less2 )

print( "Obs 0" ) 
print( cnt_Less2_0 )

print( "No Risk" )
print( cnt_noRisk )

print( "No Diag" )
print( cnt_noDiag )

print( "No Diag2" ) 
print( cnt_noDiag2 )

print( "No 24" ) 
print( cnt_no24 )

print( "No 24_?" ) 
print( cnt_check )

print( "No ADOS" ) 
print( cnt_noADOS )

print( "No. Exception")
print( len( except_list ) )

print( "======== Subjects ========" )

print( "Female" )
print( cnt_subjFemale )
print( "Male" ) 
print( cnt_subjMale )


print( "Healthy" ) 
print( cnt_subjHealthy ) 

print( "AS" )
print( cnt_subjAS )

print( "No. All Subject" )
print( len( subj_set ) ) 

print( "No. Subj in Analysis" )
numObs = len( shape_list_a ) 
print( numObs )

print( "Excluded Subj" )
allExclusion =  cnt_Less2_0 + cnt_Less2 + cnt_check
print( allExclusion )

print( "Lost in Space" )
print( len( subj_set ) - ( numObs + allExclusion ) )


print( "MLMG" )
print( R2_MLMG )
print( adjustedR2_MLMG )

print( "MLSG" )
print( R2_MLSG )
print( adjustedR2_MLSG )	

print( "SG" )
print( R2_SG )
print( adjustedR2_SG )	

