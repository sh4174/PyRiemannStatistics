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
dataFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep_LC_Aligned/subjects/"

# Anatomy list
# anatomy_list = [ 'left_caudate', 'left_putamen' ]
anatomy_list = [ 'left_caudate' ]

# M-Rep Lists
CMRepDataList = []
riskGroupList = []
ageList = [] 
CAPList = []
SubjectList = []

# vtkPolyData for Intrinsic Mean
meanPolyData = vtk.vtkPolyData()

# For all subjects
cnt = 0
for i in range( len( dataInfoList )  ):
# for i in range( 20 ):
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

		for anatomy in anatomy_list:
			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				continue

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( anatomy_cmrep_surface_path )
			reader.Update()
			polyData = reader.GetOutput()

			if cnt == 0:
				meanPolyData.DeepCopy( polyData )	
				# print( polyData )

			nAtoms = polyData.GetNumberOfPoints() 

			cmrep_ij = manifolds.cmrep_bndr_normals( nAtoms )

			for k in range( nAtoms ):
				pos = polyData.GetPoint( k )
				rad = polyData.GetPointData().GetArray( "Radius Function" ).GetValue( k )

				spoke1 = polyData.GetPointData().GetArray( "Spoke1" ).GetTuple( k )
				spoke1 = np.divide( spoke1, np.linalg.norm( spoke1 ) )

				spoke2 = polyData.GetPointData().GetArray( "Spoke2" ).GetTuple( k )
				spoke2 = np.divide( spoke2, np.linalg.norm( spoke2 ) )

				if np.linalg.norm( np.subtract( spoke1, spoke2 ) ) < 1e-10:
					cmrep_ij.edge[ k ] = 1

				cmrep_ij.SetPosition( k, pos )
				cmrep_ij.SetRadius( k, rad )
				cmrep_ij.SetSpoke1( k, spoke1 )
				cmrep_ij.SetSpoke2( k, spoke2 )				

			cmrep_ij.UpdateMeanRadius()

			CMRepDataList.append( cmrep_ij )
			riskGroupList.append( dataInfoList[i].CAPGroupList[ j ] )
			ageList.append( dataInfoList[i].AgeList[ j ] )
			SubjectList.append( dataInfoList[i].ID )
			CAPList.append( dataInfoList[i].CAPList[j] )
			cnt +=1 

# Manifold Dimension
nManDim = CMRepDataList[0].nDim
nData = len( CMRepDataList )

# # Intrinsic Mean
# mu = rsm.FrechetMean( CMRepDataList )
# mu.Write( '/media/shong/IntHard1/4DAnalysis/NIPS2019/CMRepCorrelationTest/CMRep_LC_Mean.pt' ) 
mu = manifolds.cmrep_bndr_normals( nAtoms )
mu.Read( '/media/shong/IntHard1/4DAnalysis/NIPS2019/CMRepCorrelationTest/CMRep_LC_Mean.pt' ) 




