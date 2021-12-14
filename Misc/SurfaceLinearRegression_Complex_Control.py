import os
import sys
import csv
import vtk

import numpy as np
import pandas as pd

import math 

# Stats Model
import statsmodels.api as sm

import matplotlib.pyplot as plt



def MeasurePointwiseR2( t_list, data_list_x, data_list_y, data_list_z, est_model_x, est_model_y, est_model_z ):
	# Calculate a mean 
	nData = len( data_list_x )

	mu_x = 0
	mu_y = 0
	mu_z = 0

	for i in range( len( data_list_x ) ):
		mu_x += float( data_list_x[ i ] ) / float( nData )
		mu_y += float( data_list_y[ i ] ) / float( nData )
		mu_z += float( data_list_z[ i ] ) / float( nData )
	
	# Calculate variance
	var_tot = 0 

	for i in range( nData ):
		var_tot += ( ( ( data_list_x[ i ] - mu_x ) ** 2 + ( data_list_y[ i ] - mu_y ) ** 2 + ( data_list_z[ i ] - mu_z ) ** 2 ) / float( nData ) )

	interp_x = est_model_x.params[ 0 ]
	slope_x = est_model_x.params[ 1 ] 

	interp_y = est_model_y.params[ 0 ]
	slope_y = est_model_y.params[ 1 ] 

	interp_z = est_model_z.params[ 0 ]
	slope_z = est_model_z.params[ 1 ] 

	var_explained = 0

	for i in range( nData ):
		est_x_t = interp_x + slope_x * t_list[ i ]
		est_y_t = interp_y + slope_y * t_list[ i ]
		est_z_t = interp_z + slope_z * t_list[ i ]

		var_explained += ( ( ( data_list_x[ i ] - est_x_t ) ** 2 + ( data_list_y[ i ] - est_y_t ) ** 2 + ( data_list_z[ i ] - est_z_t ) ** 2 ) / float( nData ) ) 

	R2 = 1 - ( var_explained / var_tot )

	return R2

# Data Information Class
class dataInfo:
	def __init__( self ):
		self.ID = ''
		self.LabelList = []
		self.AgeList = []
		self.CAPGroupList = []
		self.CAPList = [] 


	def __repr__(self):
		return "dataInfo Class \n ID : %s \n LabelList : %s, \n AgeList : %s, \n CAPGroupList : %s \n" % ( self.ID, self.LabelList, self.AgeList, self.CAPGroupList )

# Data Excel Sheet
# Read Subject Information
dataInfoList = []
csvPath = '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep/DataPath_2014_Volumes_Polished_Diagnosis.csv'

csvFile = open( csvPath )
reader = csv.DictReader( csvFile )

for row in reader:
	id_str = row[ 'ID' ]

	bIsInList = 0

	for k in range( len( dataInfoList ) ):
		dataInfo_k = dataInfoList[ k ]

		if id_str[ -5: ] == dataInfo_k.ID:
			label_str = row[ 'Label' ]
			age_str = row[ 'Scan_Age' ]
			CAPG_str = row[ 'CAP Group' ]
			CAP_str = row[ 'CAP' ] 

			if CAPG_str == '':
				CAPG_str = 'cont'
			if CAP_str == '':				
				CAP_str = '-1'

			dataInfo_k.LabelList.append( label_str )
			dataInfo_k.AgeList.append( float( age_str ) )
			dataInfo_k.CAPGroupList.append( CAPG_str )
			dataInfo_k.CAPList.append( float( CAP_str ) )			

			bIsInList = 1
			break

	if bIsInList == 0:
		label_str = row[ 'Label' ]
		age_str = row[ 'Scan_Age' ]
		CAPG_str = row[ 'CAP Group' ]
		CAP_str = row[ 'CAP' ] 

		if CAPG_str == '':
			CAPG_str = 'cont'
		if CAP_str == '':				
			CAP_str = '-1'

		dataInfo_new = dataInfo()
		dataInfo_new.ID = id_str[ -5: ]
		dataInfo_new.LabelList.append( label_str )
		dataInfo_new.AgeList.append( float( age_str ) )
		dataInfo_new.CAPGroupList.append( CAPG_str )
		dataInfo_new.CAPList.append( float( CAP_str ) )

		dataInfoList.append( dataInfo_new )

# Data Folder
dataFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep/subjects/"

# Estimated Output Folder Path
outFolderPath = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/Surface_Complex_CTRL_ShapeWork/"


# Anatomy list
anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]
# anatomy_list = [ 'left_caudate' ]

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


# For all subjects
for anatomy in anatomy_list:
	corr_pt_vtk_list = [] 
	ageList = []
	subjIDList = []


	for i in range( len( dataInfoList )  ):
	# for i in range( 10 ):
		subj_dataFolder = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID
		if not os.path.isdir( subj_dataFolder ):
			# print( 'PHD-AS1-' + dataInfoList[i].ID + "does not exist" )
			continue

		# Skip if there is only one shape in the list 
		if len( dataInfoList[i].AgeList ) < 2:
			# print( dataInfoList[i].ID + "has less than 2 data" )
			continue

		for j in range( len( dataInfoList[i].LabelList ) ):
			if j > 0:
				break

			if not dataInfoList[i].CAPGroupList[ j ] == 'cont':
				continue
			
			subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

			# print( "ID : " + dataInfoList[i].ID + ", Label : " + dataInfoList[i].LabelList[j ] )

			input_path = subj_i_label_j_folderPath + anatomy + ".mha"

			if not os.path.isfile( input_path ):
				continue

			corr_pts_vtk_path = subj_i_label_j_folderPath + anatomy + "corr_pts.vtk"
			reader_ij = vtk.vtkPolyDataReader()
			reader_ij.SetFileName( corr_pts_vtk_path )
			reader_ij.Update()

			data_ij = reader_ij.GetOutput()

			corr_pt_vtk_list.append( data_ij )
			ageList.append( dataInfoList[ i ].AgeList[ j ] )
			subjIDList.append( dataInfoList[ i ].ID )


for anatomy in anatomy_list:	
	data0 = corr_pt_vtk_list[ 0 ]

	nData = len( corr_pt_vtk_list )
	nPt = data0.GetNumberOfPoints()

	print( nData )

	est_pt_vtk_list = []

	minAge = 20
	maxAge = 80
	est_age_list = np.arange( minAge, maxAge + 1, 1 )

	for t_i, t in enumerate( est_age_list ):
		est_pt_vtk_t = vtk.vtkPolyData()
		est_pt_vtk_t.DeepCopy( data0 )

		est_pt_vtk_list.append( est_pt_vtk_t )		

	R2Arr = vtk.vtkFloatArray()
	R2Arr.SetNumberOfValues( nPt )
	R2Arr.SetName( "R2" )

	maxR2 = -1.0 
	maxR2_idx = 0

	estModel_x_list = []
	estModel_y_list = []
	estModel_z_list = []


	for p in range( nPt ):
		pt_list_x = []
		pt_list_y = []
		pt_list_z = []

		for i in range( nData ):
			data_i = corr_pt_vtk_list[ i ]
			pt_i_p = data_i.GetPoint( p )

			pt_list_x.append( pt_i_p[ 0 ] )
			pt_list_y.append( pt_i_p[ 1 ] )
			pt_list_z.append( pt_i_p[ 2 ] )
		
		# Linear Regression
		t_list_sm_x = sm.add_constant( ageList )
		LS_model_x = sm.OLS( pt_list_x, t_list_sm_x )
		est_x = LS_model_x.fit()
		est_x_interp = est_x.params[ 0 ]
		est_x_slope = est_x.params[ 1 ] 

		t_list_sm_y = sm.add_constant( ageList )
		LS_model_y = sm.OLS( pt_list_y, t_list_sm_y )
		est_y = LS_model_y.fit()
		est_y_interp = est_y.params[ 0 ]
		est_y_slope = est_y.params[ 1 ] 
		
		t_list_sm_z = sm.add_constant( ageList )
		LS_model_z = sm.OLS( pt_list_z, t_list_sm_z )
		est_z = LS_model_z.fit()
		est_z_interp = est_z.params[ 0 ]
		est_z_slope = est_z.params[ 1 ] 

		estModel_x_list.append( est_x )
		estModel_y_list.append( est_y )
		estModel_z_list.append( est_z )

		pointR2 = MeasurePointwiseR2( ageList, pt_list_x, pt_list_y, pt_list_z, est_x, est_y, est_z )

		if pointR2 > maxR2:
			maxR2 = pointR2 
			maxR2_idx = p

		R2Arr.SetValue( p, pointR2 )

		# print( pointR2 )		

		est_age_list = np.arange( minAge, maxAge + 1, 1 )

		for t_i, t in enumerate( est_age_list ):
			est_x_t = est_x_interp + est_x_slope * t
			est_y_t = est_y_interp + est_y_slope * t 
			est_z_t = est_z_interp + est_z_slope * t 

			est_pt_vtk_list[ t_i ].GetPoints().SetPoint( p, [ est_x_t, est_y_t, est_z_t ] )
			est_pt_vtk_list[ t_i ].GetPointData().AddArray( R2Arr )

	# # Plot Graph at maximum R2 idx 
	# print( "Max R2 Position : " + anatomy )
	# print( "R2 : " + str( maxR2 ) )
	# print( "Idx : " + str( maxR2_idx ) )

	# maxR2_data_x_list = []
	# maxR2_data_y_list = []
	# maxR2_data_z_list = []

	# for i in range( nData ):
	# 	data_i = corr_pt_vtk_list[ i ]
	# 	pt_i_p = data_i.GetPoint( maxR2_idx )

	# 	pt_i_p_x = pt_i_p[ 0 ]
	# 	pt_i_p_y = pt_i_p[ 1 ]
	# 	pt_i_p_z = pt_i_p[ 2 ]

	# 	maxR2_data_x_list.append( pt_i_p_x )
	# 	maxR2_data_y_list.append( pt_i_p_y )
	# 	maxR2_data_z_list.append( pt_i_p_z )

	# estTimeArr = np.arange( 20, 81, 1 )

	# maxR2_est_model_x = estModel_x_list[ maxR2_idx ] 
	# maxR2_est_model_y = estModel_y_list[ maxR2_idx ] 
	# maxR2_est_model_z = estModel_z_list[ maxR2_idx ] 
	
	# maxR2_est_x_list = []
	# maxR2_est_y_list = []
	# maxR2_est_z_list = []
	
	# for t in estTimeArr:
	# 	maxR2_est_x_t = maxR2_est_model_x.params[0] + t * maxR2_est_model_x.params[ 1 ]
	# 	maxR2_est_y_t = maxR2_est_model_y.params[0] + t * maxR2_est_model_y.params[ 1 ]
	# 	maxR2_est_z_t = maxR2_est_model_z.params[0] + t * maxR2_est_model_z.params[ 1 ]

	# 	maxR2_est_x_list.append( maxR2_est_x_t )
	# 	maxR2_est_y_list.append( maxR2_est_y_t )
	# 	maxR2_est_z_list.append( maxR2_est_z_t )
	# # max R2 plotting
	# plt.figure( figsize=( 6, 6 ) )
	# plt.scatter( ageList, maxR2_data_x_list, c=colors[ 0 ], alpha=0.5, label= "Data X" )
	# plt.plot( estTimeArr, maxR2_est_x_list, c=est_colors[ 0 ], label="LReg" ) 
	# plt.xlabel('Time') 
	# plt.ylabel('X' )
	# plt.title( anatomy )
	# plt.legend()
	# plt.tight_layout()
		
	# plt.figure( figsize=( 6, 6 ) )
	# plt.scatter( ageList, maxR2_data_y_list, c=colors[ 0 ], alpha=0.5, label= "Data Y" )
	# plt.plot( estTimeArr, maxR2_est_y_list, c=est_colors[ 0 ], label="LReg" ) 
	# plt.xlabel('Time') 
	# plt.ylabel('Y' )
	# plt.title( anatomy )
	# plt.legend()
	# plt.tight_layout()

	# plt.figure( figsize=( 6, 6 ) )
	# plt.scatter( ageList, maxR2_data_z_list, c=colors[ 0 ], alpha=0.5, label= "Data Z" )
	# plt.plot( estTimeArr, maxR2_est_z_list, c=est_colors[ 0 ], label="LReg" ) 
	# plt.xlabel('Time') 
	# plt.ylabel('Z' )
	# plt.title( anatomy )
	# plt.legend()
	# plt.tight_layout()

	est_age_list = np.arange( minAge, maxAge + 1, 1 )

	for t_i, t in enumerate( est_age_list ):
		filePath = outFolderPath + "Surface_ShapeWork_CTRL_Complex_" + anatomy + "_" + str( t_i ) + ".vtk"

		est_writer = vtk.vtkPolyDataWriter()
		est_writer.SetInputData( est_pt_vtk_list[ t_i ] )
		est_writer.SetFileName( filePath )
		est_writer.Update()
		est_writer.Write()

	for age_idx in range( len( ageList ) ):
		age_i = ageList[ age_idx ]
		subject_i = subjIDList[ age_idx ]

		est_vtk_i = vtk.vtkPolyData()
		est_vtk_i.DeepCopy( data0 )

		for p in range( nPt ):
			x_intercept = estModel_x_list[ p ].params[ 0 ]
			x_slope = estModel_x_list[ p ].params[ 1 ]

			y_intercept = estModel_y_list[ p ].params[ 0 ]
			y_slope = estModel_y_list[ p ].params[ 1 ]

			z_intercept = estModel_z_list[ p ].params[ 0 ]
			z_slope = estModel_z_list[ p ].params[ 1 ]

			est_x_t = x_intercept + x_slope * age_i
			est_y_t = y_intercept + y_slope * age_i
			est_z_t = z_intercept + z_slope * age_i

			est_vtk_i.GetPoints().SetPoint( p, [ est_x_t, est_y_t, est_z_t ] )

		est_vtk_i.Modified()

		filePath_est_i = outFolderPath + "Surface_ShapeWork_CTRL_Complex_" + anatomy + "_LinearModel_" + subject_i + "_" + str( age_i ) + ".vtk"
		
		writer_i = vtk.vtkPolyDataWriter()		
		writer_i.SetFileName( filePath_est_i ) 
		writer_i.SetInputData( est_vtk_i )
		writer_i.Update()
		writer_i.Write()

		byuFilePath_est_i = outFolderPath + "Surface_ShapeWork_CTRL_Complex_" + anatomy + "_LinearModel_" + subject_i + "_" + str( age_i ) + ".byu"

		byuWriter_i = vtk.vtkBYUWriter()
		byuWriter_i.SetGeometryFileName( byuFilePath_est_i )
		byuWriter_i.SetInputData( est_vtk_i )
		byuWriter_i.Update()
		byuWriter_i.Write()

		stlFilePath_est_i = outFolderPath + "Surface_ShapeWork_CTRL_Complex_" + anatomy + "_LinearModel_" + subject_i + "_" + str( age_i ) + ".stl"

		stlWriter_i = vtk.vtkSTLWriter()
		stlWriter_i.SetFileName( stlFilePath_est_i )
		stlWriter_i.SetInputData( est_vtk_i )
		stlWriter_i.Update()
		stlWriter_i.Write()



# plt.show() 