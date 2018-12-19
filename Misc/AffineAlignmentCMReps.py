import os
import sys
import csv
import vtk

import numpy as np

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
anatomy_list = [ 'left_caudate' ]

# M-Rep Lists
CMRepDataList = []
riskGroupList = []
ageList = [] 
CAPList = []
SubjectList = []

# vtkPolyData for Intrinsic Mean
outFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep_LC_AffineAligned/"
meanPolyDataFilePath = "intrinsic_mean_mrep_LC_HDRiskGroup.vtk"

meanDataPath = outFolderPath + meanPolyDataFilePath

reader_mu = vtk.vtkPolyDataReader()
reader_mu.SetFileName( meanDataPath )
reader_mu.Update()

mu = reader_mu.GetOutput() 

# For all subjects
cnt = 0
for i in range( len( dataInfoList )  ):
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
	
		sujFolderPath = 'PHD-AS1-' + dataInfoList[i].ID + "/"
		subj_i_label_j_folderPath = dataFolderPath + sujFolderPath  + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

		outSubjFolderPath = outFolderPath + sujFolderPath
		os.system( "mkdir " + outSubjFolderPath )

		outLabelFolderPath = outSubjFolderPath + dataInfoList[ i ].LabelList[ j ]
		os.system( "mkdir " + outLabelFolderPath )

		outSurfPath = outLabelFolderPath + "/surfaces/"
		os.system( "mkdir " + outSurfPath )

		outDecimatedAlignPath = outSurfPath + "decimated_aligned/"
		os.system( "mkdir " + outDecimatedAlignPath )

		nAtoms = 0

		for a in range( len( anatomy_list ) ):
			anatomy = anatomy_list[ a ] 

			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"

			outAnatomyPath = outDecimatedAlignPath + "cmrep_" + anatomy
			os.system( "mkdir " + outAnatomyPath )

			outMeshPath = outAnatomyPath + "/mesh/"
			os.system( "mkdir " + outMeshPath )

			outAnatomy_cmrep_surface_path = outDecimatedAlignPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk" 

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				break

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( anatomy_cmrep_surface_path )
			reader.Update()
			polyData = reader.GetOutput()


			icp = vtk.vtkIterativeClosestPointTransform()
			icp.SetSource( polyData )
			icp.SetTarget( mu )
			icp.GetLandmarkTransform().SetModeToAffine()
			icp.SetMaximumNumberOfIterations( 100 )
			icp.Modified()
			icp.Update()

			icpTransform = vtk.vtkTransformPolyDataFilter()
			icpTransform.SetInputData( polyData )
			icpTransform.SetTransform( icp )
			icpTransform.Update()

			aligned_data = icpTransform.GetOutput()

			writer = vtk.vtkPolyDataWriter()
			writer.SetFileName( outAnatomy_cmrep_surface_path )
			writer.SetInputData( aligned_data )
			writer.Update()
			writer.Write()



