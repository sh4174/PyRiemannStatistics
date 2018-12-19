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
dataFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep/subjects/"

# Anatomy list
anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]

# M-Rep Lists
CMRepDataList = []
riskGroupList = []
ageList = [] 
CAPList = []
SubjectList = []

# vtkPolyData for Intrinsic Mean
meanPolyDataList = []

for d in range( len( anatomy_list ) ):
	meanPolyData_d = vtk.vtkPolyData()
	meanPolyDataList.append( meanPolyData_d )	

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

		if not dataInfoList[i].CAPGroupList[ j ] == 'cont':
			continue
			
		subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

		nAtoms = 0
		cmrep_ij = manifolds.cmrep( 0 )

		for a in range( len( anatomy_list ) ):
			anatomy = anatomy_list[ a ] 

			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				break

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( anatomy_cmrep_surface_path )
			reader.Update()
			polyData = reader.GetOutput()

			if cnt == 0:
				meanPolyDataList[ a ].DeepCopy( polyData )	
				# print( polyData )

			nAtoms_a = polyData.GetNumberOfPoints() 
			nAtoms += nAtoms_a

			for k in range( nAtoms_a ):
				pos = polyData.GetPoint( k )
				rad = polyData.GetPointData().GetArray( "Radius Function" ).GetValue( k )

				cmrep_ij_pos_a_k = manifolds.euclidean( 3 )
				cmrep_ij_rad_a_k = manifolds.pos_real( 1 )

				cmrep_ij_pos_a_k.SetPoint( pos )
				cmrep_ij_rad_a_k.SetPoint( rad )

				cmrep_ij.AppendAtom( [ cmrep_ij_pos_a_k, cmrep_ij_rad_a_k ] )

			# cmrep_ij.UpdateMeanRadius()

		CMRepDataList.append( cmrep_ij )
		riskGroupList.append( dataInfoList[i].CAPGroupList[ j ] )
		ageList.append( dataInfoList[i].AgeList[ j ] )
		SubjectList.append( dataInfoList[i].ID )
		CAPList.append( dataInfoList[i].CAPList[j] )
		cnt +=1 

nDataList = []
nDimStartlist = [ 0 ]

for d in range( len( anatomy_list ) ):
	meanPolyData_d = meanPolyDataList[ d ]
	nDataList.append( meanPolyData_d.GetNumberOfPoints() )

	if d > 0:
		nDimStartlist.append( nDataList[ d - 1 ] + nDimStartlist[ d - 1 ] )

# Manifold Dimension
nManDim = CMRepDataList[0].nDim
nData = len( CMRepDataList )

base = manifolds.cmrep( nManDim )
base.Read( "CMRep_LinearizedGeodesicRegression_Complex_CTRL_base.rpt" )

tangent = manifolds.cmrep_tVec( nManDim ) 
tangent.Read(  "CMRep_LinearizedGeodesicRegression_Complex_CTRL_tangent.tVec"  )

t0 = 20 
tN = 80
nTimePt = 30
est_time_list_i = []
est_rad_pt_list_i = []

regression_output_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2018/GeodesicRegressionResults/CMRep_CTRL_Complex_NewCode_Linearized/'

for a in range( len( anatomy_list ) ):
	anatomy = anatomy_list[ a ] 
	
	inFileName = 'CMRep_Regression_CTRL_Comlex_' + anatomy + '_' + str( 0 ) + '.vtk' 
	outFileName = 'CMRep_Regression_CTRL_Comlex_' + anatomy + '_' + str( 0 ) + '_test.vtk' 

	input_path = regression_output_folder_path + inFileName 
	output_path = regression_output_folder_path + outFileName  

	reader = vtk.vtkPolyDataReader()
	reader.SetFileName( input_path )
	reader.Update()

	polyData = reader.GetOutput() 
	nDim_a = polyData.GetNumberOfPoints() 

	posVecArr = vtk.vtkFloatArray()
	posVecArr.SetNumberOfComponents( 3 )
	posVecArr.SetNumberOfTuples( nDim_a )
	posVecArr.SetName( "Pos_tVec" ) 

	posMagArr = vtk.vtkFloatArray()
	posMagArr.SetNumberOfValues( nDim_a ) 
	posMagArr.SetName( "Pos_tVec_Mag" )

	radVecArr = vtk.vtkFloatArray()
	radVecArr.SetNumberOfValues( nDim_a )
	radVecArr.SetName( "Rad_tVec" )


	for k in range( nDim_a ):
		k_cmrep = nDimStartlist[ a ] + k 

		tVec_a_k_pos = tangent.tVector[ k_cmrep ][ 0 ].tVector

		tVec_a_k_rad = tangent.tVector[ k_cmrep ][ 1 ].tVector[ 0 ]

		posVecArr.SetTuple( k, tVec_a_k_pos )
		posMagArr.SetValue( k, np.linalg.norm( tVec_a_k_pos ) )
		radVecArr.SetValue( k, tVec_a_k_rad )


	posVecArr.Modified()
	posMagArr.Modified()
	radVecArr.Modified()

	polyData.GetPointData().AddArray( posVecArr )
	polyData.GetPointData().AddArray( posMagArr )
	polyData.GetPointData().AddArray( radVecArr ) 

	polyData.Modified()

	writer_t = vtk.vtkPolyDataWriter() 
	writer_t.SetFileName( output_path )
	writer_t.SetInputData( polyData )
	writer_t.Update()
	writer_t.Write() 
