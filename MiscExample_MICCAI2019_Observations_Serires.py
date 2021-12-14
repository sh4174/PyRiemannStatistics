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

import time


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

		isAll = True

		# Set CM-Rep Abstract Point
		for a, anatomy in enumerate( anatomy_list ):
			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"
			anatomy_cmrep_surface_cen_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med_centered.vtk" 
			anatomy_cmrep_surface_cen_norm_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med_centered_normalized.vtk" 

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				isAll = False 
				continue

		if not isAll:
			continue

		print( dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] )
		
		if dataInfoList[i].ID == "51095" or dataInfoList[i].ID == "51451" or dataInfoList[i].ID =="52050" or dataInfoList[i].ID =="52838" or dataInfoList[i].ID == "51284" :
			continue

		for a, anatomy in enumerate( anatomy_list ):
			bndr_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.bnd.vtk"
			target_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3_target.vtk" 

			if not os.path.isfile( bndr_path ):
				print( bndr_path )				
				print( "File doesn't exist" )
				continue

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( bndr_path )
			reader.Update()
			bndr = reader.GetOutput()
			
			nPtArr = bndr.GetPointData().GetNumberOfArrays()

			for p_a in range( nPtArr ):
				bndr.GetPointData().RemoveArray( bndr.GetPointData().GetArrayName( p_a ) )

			nCellArr = bndr.GetCellData().GetNumberOfArrays()

			for p_a in range( nCellArr ):
				bndr.GetCellData().RemoveArray( bndr.GetCellData().GetArrayName( p_a ) )			


			tReader = vtk.vtkPolyDataReader()
			tReader.SetFileName( target_path )
			tReader.Update()
			target = tReader.GetOutput()

			bndrOutFileName = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/MICCAI2019_CTRL_Complex_Subjects_Series/CMRep_bndr_" + anatomy + "_" + str( cnt ) + ".vtk"
			targetOutFileName = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/MICCAI2019_CTRL_Complex_Subjects_Series/Obs_" + anatomy + "_" + str( cnt ) + ".vtk"

			writer = vtk.vtkPolyDataWriter()
			writer.SetFileName( bndrOutFileName )
			writer.SetInputData( bndr )
			writer.Update()
			writer.Write()

			writer_target = vtk.vtkPolyDataWriter()
			writer_target.SetFileName( targetOutFileName )
			writer_target.SetInputData( target )
			writer_target.Update()

			writer_target.Write()			

		cnt = cnt + 1

