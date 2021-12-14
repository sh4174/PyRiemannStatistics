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
# import manifolds
# import StatsModel as rsm

import time
from scipy.spatial.distance import directed_hausdorff 

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

# M-Rep Lists
CMRepDataList = [ [], [], [], [] ]
riskGroupList = []
ageList = [] 
CAPList = []
SubjectList = []
labelList = []

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

		if not dataInfoList[i].CAPGroupList[ j ] == 'cont':
			continue
			
		subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

		# CMRepDataList.append( cmrep_ij )
		ageList.append( dataInfoList[i].AgeList[ j ] )
		SubjectList.append( dataInfoList[i].ID )
		CAPList.append( dataInfoList[i].CAPList[j] )
		labelList.append( dataInfoList[ i ].LabelList[ j ] )
		cnt +=1 


# Geodesic Regression
print( "# of Subjects " ) 
print( len( ageList ) )
# print( CMRepDataList )


# Synth
reconBinPath = "/media/shong/IntHard1/4DAnalysis/Code/SPTSkeleton/CMRepReg/deformetrica/bin_cmrep_recon/ReconstructBoundary"

nTimePt = 60

anatomy_list = [ 'left_caudate', 'right_caudate', 'left_putamen', 'right_putamen' ]

# Input Folders
dataFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep/subjects/"

# Estimated Folders
EGRegFolderPath = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_CTRL_Group_NewCode_Linearized_Complex/Subjects/"
PC_LRegFolderPath = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/Surface_Complex_CTRL_ShapeWork/"
DiffeoFolderPath = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/Surface_Complex_CTRL_Deformetrica/output_fullRes/"

subjArr = np.array( SubjectList )
ageArr = np.array( ageList )
labelArr = np.array( labelList )

idxArr = ageArr.argsort()

sorted_ageArr = ageArr[ idxArr ]
sorted_subjArr = subjArr[ idxArr ] 
sorted_labelArr = labelArr[ idxArr ] 

Haus_EGReg = 0 
Haus_LReg = 0
Haus_DiffeoReg = 0

for idx in range( len( sorted_subjArr ) ):
	subj_i = sorted_subjArr[ idx ]
	age_i = sorted_ageArr[ idx ] 	
	label_i = sorted_labelArr[ idx ] 

	subjFolderPath = dataFolderPath + "PHD-AS1-" + subj_i + "/" + label_i + "/surfaces/decimated_aligned/"

	Haus_EGReg_i = 0
	Haus_LReg_i = 0
	Haus_DiffeoReg_i = 0

	for a in range( len( anatomy_list ) ) :
		anatomy = anatomy_list[ a ] 

		# Input Paths 
		inputPath_i = subjFolderPath + "cmrep_" + anatomy + "/mesh/def3_target.vtk"

		inputReader = vtk.vtkPolyDataReader()
		inputReader.SetFileName( inputPath_i )
		inputReader.Update()

		input_i = inputReader.GetOutput()

		# Estimated Paths
		EGRegPath_i = EGRegFolderPath + "CMRep_EGReg_Recon_" + anatomy + "_Subjects_" + subj_i + "_t_" + str( age_i ) + ".vtk"
		EGReg_reader = vtk.vtkPolyDataReader()
		EGReg_reader.SetFileName( EGRegPath_i )
		EGReg_reader.Update()

		EGReg_est = EGReg_reader.GetOutput()

		LRegPath_i = PC_LRegFolderPath + "Surface_ShapeWork_CTRL_Complex_" + anatomy + "_LinearModel_" + subj_i + "_" + str( age_i ) + ".vtk"
		LReg_reader = vtk.vtkPolyDataReader()
		LReg_reader.SetFileName( LRegPath_i )
		LReg_reader.Update()

		LReg_est = LReg_reader.GetOutput()

		transform = vtk.vtkTransform()
		transform.RotateX( 180 )
		transform.RotateY( 180 )
		transform.Update()

		LReg_tFilter = vtk.vtkTransformPolyDataFilter()
		LReg_tFilter.SetTransform( transform )
		LReg_tFilter.SetInputData( LReg_est )
		LReg_tFilter.Update()

		LReg_est_t = LReg_tFilter.GetOutput()

		DiffeoRegPath_i = DiffeoFolderPath + "GeodesicRegression__Reconstruction__" + anatomy + "__tp_" + str( idx ) + "__age_" + ("%.2f" % age_i ) + ".vtk"
		DiffeoReg_reader = vtk.vtkPolyDataReader()
		DiffeoReg_reader.SetFileName( DiffeoRegPath_i )
		DiffeoReg_reader.Update()

		DiffeoReg_est = DiffeoReg_reader.GetOutput()

		DiffeoReg_tFilter = vtk.vtkTransformPolyDataFilter()
		DiffeoReg_tFilter.SetTransform( transform )
		DiffeoReg_tFilter.SetInputData( DiffeoReg_est )
		DiffeoReg_tFilter.Update()

		DiffeoReg_est_t = DiffeoReg_tFilter.GetOutput()

		input_pt_list = [] 

		for pp in range( input_i.GetNumberOfPoints() ):
			pt_pp = input_i.GetPoint( pp )
			input_pt_list.append( pt_pp ) 

		EGReg_est_pt_list = []

		for pp in range( EGReg_est.GetNumberOfPoints() ):
			EGReg_est_pt_list.append( EGReg_est.GetPoint( pp ) )

		LReg_est_pt_list = [] 

		for pp in range( LReg_est_t.GetNumberOfPoints() ):
			LReg_est_pt_list.append( LReg_est_t.GetPoint( pp ) )

		Diffeo_est_pt_list = [] 

		for pp in range( DiffeoReg_est_t.GetNumberOfPoints() ):
			Diffeo_est_pt_list.append( DiffeoReg_est_t.GetPoint( pp ) )

		Haus_EGReg_a = max( directed_hausdorff( input_pt_list, EGReg_est_pt_list )[0], directed_hausdorff( EGReg_est_pt_list, input_pt_list )[ 0 ] )

		Haus_EGReg_i += ( Haus_EGReg_a / 4.0 )

		Haus_LReg_a = max( directed_hausdorff( input_pt_list, LReg_est_pt_list )[0], directed_hausdorff( LReg_est_pt_list, input_pt_list )[ 0 ] )

		Haus_LReg_i += ( Haus_LReg_a / 4.0 )

		Haus_DiffeoReg_a = max( directed_hausdorff( input_pt_list, Diffeo_est_pt_list )[0], directed_hausdorff( Diffeo_est_pt_list, input_pt_list )[ 0 ] )

		Haus_DiffeoReg_i += ( Haus_DiffeoReg_a / 4.0 )

	Haus_EGReg += Haus_EGReg_i
	Haus_LReg += Haus_LReg_i
	Haus_DiffeoReg += Haus_DiffeoReg_i


Haus_EGReg = Haus_EGReg / float( len( sorted_subjArr ) )
Haus_LReg = Haus_LReg / float( len( sorted_subjArr ) )
Haus_DiffeoReg = Haus_DiffeoReg / float( len( sorted_subjArr ) )


print( "EGReg Average Hausdorff Distance" ) 
print( Haus_EGReg )

print( "LReg Average Hausdorff Distance" ) 
print( Haus_LReg )

print( "DiffeoReg Average Hausdorff Distance" ) 
print( Haus_DiffeoReg )
