import os
import sys
import csv
import subprocess
import vtk

import numpy as np

# Visualization
import pylab

# PCA for Comparison
from sklearn.decomposition import PCA, KernelPCA

# Stats Model
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Riemannian Stats model
import manifolds
import StatsModel as rsm

from scipy import stats

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

# Anatomy list
anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]
# anatomy_list = [ 'left_caudate' ]

# Control Group List
CMRepDataList_CTRL = [ [], [], [], [] ]
riskGroupList_CTRL = []
ageList_CTRL = [] 
CAPList_CTRL = []
SubjectList_CTRL = []

# High Group List
CMRepDataList_HIGH = [ [], [], [], [] ]
riskGroupList_HIGH = []
ageList_HIGH = [] 
CAPList_HIGH = []
SubjectList_HIGH = []

CMRepDataList_All = [ [], [], [], [] ]

# vtkPolyData for Intrinsic Mean
meanPolyData = vtk.vtkPolyData()

# For all subjects
cnt = 0
for i in range( len( dataInfoList )  ):
# for i in range( 5 ):
	subj_dataFolder = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID
	if not os.path.isdir( subj_dataFolder ):
		print( 'PHD-AS1-' + dataInfoList[i].ID + "does not exist" )
		continue

	# Skip if there is only one shape in the list 
	if len( dataInfoList[i].AgeList ) < 2:
		print( dataInfoList[i].ID + "has less than 2 data" )
		continue

	for j in range( len( dataInfoList[i].LabelList ) ):
		if j > 0:
			break

		if not ( dataInfoList[i].CAPGroupList[ j ] == 'high' or dataInfoList[i].CAPGroupList[ j ] == 'cont' ):
			continue
			
		subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

		isAllAnatomy = True

		# Set CM-Rep Abstract Point
		for a, anatomy in enumerate( anatomy_list ):
			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"
			anatomy_cmrep_surface_cen_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med_centered.vtk" 
			anatomy_cmrep_surface_cen_norm_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med_centered_normalized.vtk" 

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				isAllAnatomy = False
				continue

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( anatomy_cmrep_surface_path )
			reader.Update()
			polyData = reader.GetOutput()

			if cnt == 0:
				meanPolyData.DeepCopy( polyData )	
				# print( polyData )

			nAtoms = polyData.GetNumberOfPoints() 

			cenOfMassFilter = vtk.vtkCenterOfMass()
			cenOfMassFilter.SetInputData( polyData )
			cenOfMassFilter.SetUseScalarsAsWeights( False )			
			cenOfMassFilter.Update()

			cenOfMass = cenOfMassFilter.GetCenter()

			polyData_cen = vtk.vtkPolyData()
			polyData_cen.DeepCopy( polyData )

			polyData_norm = vtk.vtkPolyData()
			polyData_norm.DeepCopy( polyData )

			pos_centered_arr = []
			rad_arr = []

			normal1_arr = []
			normal2_arr = []

			for k in range( nAtoms ):
				pos = polyData.GetPoint( k )
				pos_cen = np.subtract( pos, cenOfMass )
				polyData_cen.GetPoints().SetPoint( k, pos_cen )

				pos_centered_arr.append( pos_cen )				


				spoke1 = polyData.GetPointData().GetArray( "Spoke1" ).GetTuple( k )
				spoke1 = np.divide( spoke1, np.linalg.norm( spoke1 ) )

				spoke2 = polyData.GetPointData().GetArray( "Spoke2" ).GetTuple( k )
				spoke2 = np.divide( spoke2, np.linalg.norm( spoke2 ) )

				normal1_k = manifolds.sphere( 3 )
				normal2_k = manifolds.sphere( 3 )

				normal1_k.SetPoint( spoke1 )
				normal2_k.SetPoint( spoke2 )

				normal1_arr.append( normal1_k )
				normal2_arr.append( normal2_k )

				rad = polyData.GetPointData().GetArray( "Radius Function" ).GetValue( k )
				rad_arr.append( rad )


			pos_centered_arr_flatten = np.matrix( pos_centered_arr ).flatten()
			print( pos_centered_arr_flatten.shape )

			length = np.linalg.norm( pos_centered_arr_flatten )

			pos_cen_norm_arr = []

			for k in range( nAtoms ):
				pos_norm = np.divide( pos_centered_arr[ k ], length )
				pos_cen_norm_arr.append( pos_norm )
				polyData_norm.GetPoints().SetPoint( k, pos_norm )

			pos_cen_norm_arr_flatten = np.matrix( pos_cen_norm_arr ).flatten()

			polyData_norm.Modified()
			polyData_cen.Modified()

			writer_cen = vtk.vtkPolyDataWriter()
			writer_cen.SetInputData( polyData_cen )
			writer_cen.SetFileName( anatomy_cmrep_surface_cen_path )
			writer_cen.Update()
			writer_cen.Write()

			writer_norm = vtk.vtkPolyDataWriter()
			writer_norm.SetInputData( polyData_norm )
			writer_norm.SetFileName( anatomy_cmrep_surface_cen_norm_path )
			writer_norm.Update()
			writer_norm.Write()

			print( "***==============================================================***" )
			print( anatomy_cmrep_surface_path )
			print( "Normalized Vector Length" )
			print( np.linalg.norm( pos_cen_norm_arr_flatten ))
			print( "==============================================================" )

			# Project positions to pre-shape space
			pos_cen_matrix = np.matrix( pos_centered_arr )
			print( pos_cen_matrix.shape ) 

			# Create Helmert submatrix
			H = rsm.HelmertSubmatrix( nAtoms )

			# Normalize
			HX = np.dot( H, pos_cen_matrix )
			print( HX.shape )

			# H_THX = np.dot( H.T, HX ) 

			# print( H_THX[ 0, : ] )
			# print( pos_cen_matrix[ 0, : ] )

			length_HX = np.linalg.norm( HX )

			Z_H = np.divide( HX, length_HX )

			# Preshape Array
			Z_H_flatten = Z_H.flatten()

			print( np.linalg.norm( Z_H_flatten ) )

			print( length )
			print( length_HX )

			# Set CM-Rep Abstract Point
			cmrep_ij = manifolds.cmrep_abstract_normal( nAtoms )
			
			# Center			
			center_ij = manifolds.euclidean( 3 )
			# print( "Center Of Mass " )
			center_ij.SetPoint( cenOfMass )
							
			# Scale 
			scale_ij = manifolds.pos_real( 1 )
			# print( "Scale" )

			scale_ij.SetPoint( length_HX )

			# Abstract Position
			pos_ij = manifolds.sphere( 3 * ( nAtoms - 1 ) )

			# print( "Abstract Position" )
			# print( np.array( Z_H_flatten ).flatten().shape ) 			
			# print( 3 * ( nAtoms - 1 ) ) 

			pos_ij.SetPoint( np.array( Z_H_flatten ).flatten() )

			# Radius 
			rad_ij = manifolds.pos_real( nAtoms )
			# print( "Radius" )
			rad_ij.SetPoint( rad_arr )

			# CM-Rep Point
			pt_abs = [ center_ij, scale_ij, pos_ij, rad_ij, normal1_arr, normal2_arr ]
			# print( "CMRep Abstract Point" )
			cmrep_ij.SetPoint( pt_abs )

			if dataInfoList[i].CAPGroupList[ j ] == 'cont':
				CMRepDataList_CTRL[ a ].append( cmrep_ij )
			else:
				CMRepDataList_HIGH[ a ].append( cmrep_ij )
	
			CMRepDataList_All[ a ].append( cmrep_ij )

		if not isAllAnatomy:
			continue

		if dataInfoList[i].CAPGroupList[ j ] == 'cont':
			# CMRepDataList.append( cmrep_ij )
			riskGroupList_CTRL.append( dataInfoList[i].CAPGroupList[ j ] )
			ageList_CTRL.append( dataInfoList[i].AgeList[ j ] )
			SubjectList_CTRL.append( dataInfoList[i].ID )
			CAPList_CTRL.append( dataInfoList[i].CAPList[j] )
		else:
			# CMRepDataList.append( cmrep_ij )
			riskGroupList_HIGH.append( dataInfoList[i].CAPGroupList[ j ] )
			ageList_HIGH.append( dataInfoList[i].AgeList[ j ] )
			SubjectList_HIGH.append( dataInfoList[i].ID )
			CAPList_HIGH.append( dataInfoList[i].CAPList[j] )


		cnt +=1 

# Manifold Dimension
nManDim = CMRepDataList_CTRL[0][0].nDim

nData_CTRL = len( CMRepDataList_CTRL[ 0 ] )
nData_HIGH = len( CMRepDataList_HIGH[ 0 ] )
nData_All = nData_CTRL + nData_HIGH
refFilePath = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_HIGH_Group_NewCode_Linearized_Complex/CMRep_Abstract_Normal_Regression_HIGH_left_caudate_0.vtk"

reader = vtk.vtkPolyDataReader()
reader.SetFileName( refFilePath )
reader.Update()
meanPolyData = reader.GetOutput()

# Manifold Dimension
nManDim = meanPolyData.GetNumberOfPoints()

# Anatomy list
anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]

# # mu_arr_All = []
# mu_arr_CTRL = []
# mu_arr_HIGH = []

# for a, anatomy in enumerate( anatomy_list ):
# 	# outPath_mean_All_a = "CMRep_Abstract_Normal_FrechetMean_ALL_" + anatomy + ".rpt"
# 	outPath_mean_CTRL_a = "CMRep_Abstract_Normal_FrechetMean_CTRL_" + anatomy + ".rpt"
# 	outPath_mean_HIGH_a = "CMRep_Abstract_Normal_FrechetMean_HIGH_" + anatomy + ".rpt"

# 	# mu_all_a = manifolds.cmrep_abstract_normal( nManDim )
# 	# mu_all_a.Read( outPath_mean_All_a )

# 	mu_CTRL_a = manifolds.cmrep_abstract_normal( nManDim )
# 	mu_CTRL_a.Read( outPath_mean_CTRL_a )

# 	mu_HIGH_a = manifolds.cmrep_abstract_normal( nManDim )
# 	mu_HIGH_a.Read( outPath_mean_HIGH_a ) 

# 	# mu_arr_All.append( mu_all_a ) 
# 	mu_arr_CTRL.append( mu_CTRL_a )
# 	mu_arr_HIGH.append( mu_HIGH_a )

# 	# [ a ].Read( outPath_mean_All_a )
	# mu_arr_CTRL[ a ].Read( outPath_mean_CTRL_a )
	# mu_arr_HIGH[ a ].Read( outPath_mean_HIGH_a )


# w_All, v_All, mu_arr_All = rsm.TangentPGA_CMRep_Abstract_Normal_Arr( CMRepDataList_All )
# w_CTRL, v_CTRL, mu_arr_CTRL = rsm.TangentPGA_CMRep_Abstract_Normal_Arr( CMRepDataList_CTRL )
# w_HIGH, v_HIGH, mu_arr_HIGH = rsm.TangentPGA_CMRep_Abstract_Normal_Arr( CMRepDataList_HIGH ) 


mu_arr_All = rsm.TangentPGA_CMRep_Abstract_Normal_Arr( CMRepDataList_All )
mu_arr_CTRL = rsm.TangentPGA_CMRep_Abstract_Normal_Arr( CMRepDataList_CTRL )
mu_arr_HIGH = rsm.TangentPGA_CMRep_Abstract_Normal_Arr( CMRepDataList_HIGH ) 

for a, anatomy in enumerate( anatomy_list ):
	outPath_mean_All_a = "CMRep_Abstract_Normal_FrechetMean_ALL_" + anatomy + ".rpt"
	outPath_mean_CTRL_a = "CMRep_Abstract_Normal_FrechetMean_CTRL_" + anatomy + ".rpt"
	outPath_mean_HIGH_a = "CMRep_Abstract_Normal_FrechetMean_HIGH_" + anatomy + ".rpt"

	mu_arr_All[ a ].Write( outPath_mean_All_a )
	mu_arr_CTRL[ a ].Write( outPath_mean_CTRL_a )
	mu_arr_HIGH[ a ].Write( outPath_mean_HIGH_a )


# outPath_w_All = "CMRep_Abstract_Normal_PGA_W_ALL.npy" 
# outPath_w_CTRL = "CMRep_Abstract_Normal_PGA_W_CTRL.npy" 
# outPath_w_HIGH = "CMRep_Abstract_Normal_PGA_W_HIGH.npy" 

# outPath_v_All = "CMRep_Abstract_Normal_PGA_V_ALL.npy" 
# outPath_v_CTRL = "CMRep_Abstract_Normal_PGA_V_CTRL.npy" 
# outPath_v_HIGH = "CMRep_Abstract_Normal_PGA_V_HIGH.npy" 

# np.save( outPath_w_All, w_All )
# np.save( outPath_w_CTRL, w_CTRL )
# np.save( outPath_w_HIGH, w_HIGH )

# np.save( outPath_v_All, v_All )
# np.save( outPath_v_CTRL, v_CTRL )
# np.save( outPath_v_HIGH, v_HIGH )

# print( "========================================" )
# print( "v All")
# print( v_All )
# print( "========================================" )
# print( "v CTRL")
# print( v_CTRL )
# print( "========================================" )
# print( "v HIGH")
# print( v_HIGH )
# print( "========================================" )


# print( "========================================" )
# print( "w All")
# print( w_All )
# print( "========================================" )
# print( "w CTRL")
# print( w_CTRL )
# print( "========================================" )
# print( "w HIGH")
# print( w_HIGH )
# print( "========================================" )




'''
w_All, v_All, mu_arr_All = rsm.TangentPGA_CMRep_Abstract_Normal_Mu_Arr( CMRepDataList_All )

w_CTRL, v_CTRL, mu_arr_CTRL = rsm.TangentPGA_CMRep_Abstract_Normal_Mu_Arr( CMRepDataList_CTRL )

w_HIGH, v_HIGH, mu_arr_HIGH = rsm.TangentPGA_CMRep_Abstract_Normal_Mu_Arr( CMRepDataList_HIGH ) 


for a, anatomy in enumerate( anatomy_list ):
	outPath_mean_All_a = "CMRep_Abstract_Normal_FrechetMean_ALL_" + anatomy + ".rpt"
	outPath_mean_CTRL_a = "CMRep_Abstract_Normal_FrechetMean_CTRL_" + anatomy + ".rpt"
	outPath_mean_HIGH_a = "CMRep_Abstract_Normal_FrechetMean_HIGH_" + anatomy + ".rpt"

	mu_arr_All[ a ].Write( outPath_mean_All_a )
	mu_arr_CTRL[ a ].Write( outPath_mean_CTRL_a )
	mu_arr_HIGH[ a ].Write( outPath_mean_HIGH_a )

outPath_w_All = "CMRep_Abstract_Normal_PGA_W_ALL.npy" 
outPath_w_CTRL = "CMRep_Abstract_Normal_PGA_W_CTRL.npy" 
outPath_w_HIGH = "CMRep_Abstract_Normal_PGA_W_HIGH.npy" 

outPath_v_All = "CMRep_Abstract_Normal_PGA_V_ALL.npy" 
outPath_v_CTRL = "CMRep_Abstract_Normal_PGA_V_CTRL.npy" 
outPath_v_HIGH = "CMRep_Abstract_Normal_PGA_V_HIGH.npy" 

np.save( outPath_w_All, w_All )
np.save( outPath_w_CTRL, w_CTRL )
np.save( outPath_w_HIGH, w_HIGH )

np.save( outPath_v_All, v_All )
np.save( outPath_v_CTRL, v_CTRL )
np.save( outPath_v_HIGH, v_HIGH )

print( "========================================" )
print( "v All")
print( v_All )
print( "========================================" )
print( "v CTRL")
print( v_CTRL )
print( "========================================" )
print( "v HIGH")
print( v_HIGH )
print( "========================================" )


print( "========================================" )
print( "w All")
print( w_All )
print( "========================================" )
print( "w CTRL")
print( w_CTRL )
print( "========================================" )
print( "w HIGH")
print( w_HIGH )
print( "========================================" )
'''