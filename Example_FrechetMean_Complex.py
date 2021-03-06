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

		# if dataInfoList[i].CAPGroupList[ j ] == 'cont':
		# 	continue
			
		subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

		nAtoms = 0
		cmrep_ij = manifolds.cmrep( 0 )

		IsAllAnatomy = True

		for a in range( len( anatomy_list ) ):
			anatomy = anatomy_list[ a ] 

			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				IsAllAnatomy = False
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

		if not IsAllAnatomy:
			continue
			
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

print( nManDim )
print( nData )
print( np.min( CAPList ) )
print( np.max( CAPList ) )

# Intrinsic Mean
mu = rsm.FrechetMean( CMRepDataList )

output_folder_path = '/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_CorrespondenceTest/CorrespondenceFromMu_AllAnatomy/'

for a, anatomy in enumerate( anatomy_list ):
	meanPolyData = meanPolyDataList[ a ] 

	meanPolyData.GetPointData().RemoveArray( "normals" )
	meanPolyData.GetPointData().RemoveArray( "Texture Coordinates" )
	meanPolyData.GetPointData().RemoveArray( "Covariant Tensor Determinant" )
	meanPolyData.GetPointData().RemoveArray( "Rho Function" )
	meanPolyData.GetPointData().RemoveArray( "Radius Function" )
	meanPolyData.GetPointData().RemoveArray( "Phi" )
	meanPolyData.GetPointData().RemoveArray( "Dummy1" )
	meanPolyData.GetPointData().RemoveArray( "Bending Energy" )
	meanPolyData.GetPointData().RemoveArray( "Regularity Penalty" )
	meanPolyData.GetPointData().RemoveArray( "Metric Angle" )
	meanPolyData.GetPointData().RemoveArray( "U Coordinate" )
	meanPolyData.GetPointData().RemoveArray( "V Coordinate" )
	meanPolyData.GetPointData().RemoveArray( "Mean Curvature" )
	meanPolyData.GetPointData().RemoveArray( "Gauss Curvature" )
	meanPolyData.GetPointData().RemoveArray( "Kappa1" )
	meanPolyData.GetPointData().RemoveArray( "Kappa2" )
	meanPolyData.GetPointData().RemoveArray( "Atom Normal" )
	meanPolyData.GetPointData().RemoveArray( "Stretch" )
	meanPolyData.GetPointData().RemoveArray( "Curvature Penalty Feature" )
	meanPolyData.GetPointData().RemoveArray( "Area Element" )
	meanPolyData.GetPointData().RemoveArray( "Grad R Magnitude (original)" )
	meanPolyData.GetPointData().RemoveArray( "Rs2" )
	meanPolyData.GetPointData().RemoveArray( "Spoke1" )
	meanPolyData.GetPointData().RemoveArray( "Spoke2" )
	meanPolyData.GetPointData().RemoveArray( "LaplaceBasis" )
	meanPolyData.GetPointData().RemoveArray( "Off Diagonal Term of Contravariant MT" )
	meanPolyData.GetPointData().RemoveArray( "Xu" )
	meanPolyData.GetPointData().RemoveArray( "Xv" )
	meanPolyData.GetPointData().RemoveArray( "GradR" )

	nDim_a = meanPolyData.GetNumberOfPoints() 

	for a, anatomy in enumerate( anatomy_list ):
		meanPolyData = meanPolyDataList[ a ] 

		outFileName = 'Intrinsic_Mean_Comlex_' + anatomy + '.vtk' 

		output_path = output_folder_path + outFileName 

		polyData_t = vtk.vtkPolyData()
		polyData_t.DeepCopy( meanPolyData ) 

		radiusArr_t_vtk = vtk.vtkFloatArray() 
		radiusArr_t_vtk.SetNumberOfValues( polyData_t.GetNumberOfPoints() )
		radiusArr_t_vtk.SetName( 'Radius' )

		for k in range( nDim_a ):
			k_cmrep = nDimStartlist[ a ] + k 			
			polyData_t.GetPoints().SetPoint( k, mu.pt[ k_cmrep ][ 0 ].pt )
			radiusArr_t_vtk.SetValue( k, mu.pt[ k_cmrep ][ 1 ].pt[ 0 ] )

		polyData_t.GetPointData().AddArray( radiusArr_t_vtk ) 
		
		polyData_t.Modified() 

		writer_t = vtk.vtkPolyDataWriter() 
		writer_t.SetFileName( output_path )
		writer_t.SetInputData( polyData_t )
		writer_t.Update()
		writer_t.Write() 
