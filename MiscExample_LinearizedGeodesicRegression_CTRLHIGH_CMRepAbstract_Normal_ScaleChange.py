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

refFilePath = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_HIGH_Group_NewCode_Linearized_Complex/CMRep_Abstract_Normal_Regression_HIGH_left_caudate_0.vtk"

reader = vtk.vtkPolyDataReader()
reader.SetFileName( refFilePath )
reader.Update()
meanPolyData = reader.GetOutput()

# Manifold Dimension
nManDim = meanPolyData.GetNumberOfPoints()

# Anatomy list
anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]

for a, anatomy in enumerate( anatomy_list ):
	# base, tangent = rsm.LinearizedGeodesicRegression( ageList, CMRepDataList[ a ], max_iter, step_size, step_tol, False, False )

	# base.Write( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_base.rpt" )
	# tangent.Write( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_tangent.tVec" )

	print( "===============================" )
	print( " Reading.... " ) 
	print( "===============================" )

	base = manifolds.cmrep_abstract_normal( nManDim )
	base.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_CTRL_" + anatomy + "_base.rpt" )

	tangent = manifolds.cmrep_abstract_normal_tVec( nManDim ) 
	tangent.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_CTRL_" + anatomy + "_tangent.tVec" )

	base_HR = manifolds.cmrep_abstract_normal( nManDim )
	base_HR.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_HIGH_" + anatomy + "_base.rpt" )

	tangent_HR = manifolds.cmrep_abstract_normal_tVec( nManDim ) 
	tangent_HR.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_HIGH_" + anatomy + "_tangent.tVec" )

	# Estimate CM-Rep Surface Trajectory to VTK
	# Reference VTK Poly data 
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

	t0 = 20 
	tN = 80

	nTimePt = 61
	est_time_list_i = []
	est_rad_pt_list_i = []

	# Scale Change
	change_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_CTRLHIGH_Group_NewCode_Linearized_Complex/ScaleChange/'

	outFileName = 'CMRep_Abstract_Normal_Regression_HIGH_' + anatomy + '_Scale_Change.png' 

	output_path = change_folder_path + outFileName 

	time_trend_arr = []

	ctrl_trend_scale = []
	high_trend_scale = [] 

	ctrl_time_list = []
	high_time_list = []

	ctrl_list_scale = []
	high_list_scale = []

	for n in range( nTimePt ):
		time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

		position_matrix_0 = base.GetEuclideanLocations()

		tVec_t = tangent.ScalarMultiply( time_pt )
		est_cmrep_t = base.ExponentialMap( tVec_t )

		tVec_HR_t = tangent_HR.ScalarMultiply( time_pt )
		est_cmrep_HR_t = base_HR.ExponentialMap( tVec_HR_t )


		time_trend_arr.append( time_pt )

		ctrl_trend_scale.append( est_cmrep_t.pt[ 1 ].pt[0] )
		high_trend_scale.append( est_cmrep_HR_t.pt[ 1 ].pt[0] )


	for d in range( len( CMRepDataList_CTRL[ a ] ) ):
		ctrl_list_scale.append( CMRepDataList_CTRL[ a ][ d ].pt[ 1 ].pt[0] )
		ctrl_time_list.append( ageList_CTRL[ d ] )

	for d in range( len( CMRepDataList_HIGH[ a ] ) ):
		high_list_scale.append( CMRepDataList_HIGH[ a ][ d ].pt[ 1 ].pt[0] )
		high_time_list.append( ageList_HIGH[ d ] )


	fig = plt.figure()
	plt.plot( time_trend_arr, ctrl_trend_scale, c='b', linewidth=8, label="CTRL" )
	plt.plot( time_trend_arr, high_trend_scale, c='r', linewidth=8, label="High" )

	plt.scatter( ctrl_time_list, ctrl_list_scale, c='b', s=25, alpha=0.5 )
	plt.scatter( high_time_list, high_list_scale, c='r', s=25, alpha=0.5 )
	plt.xlabel('Age (y)', fontweight='bold')
	plt.ylabel('Scale (mm^2)', fontweight='bold')
	plt.tight_layout()
	plt.legend(prop={'weight':'bold'})	
	fig.savefig( output_path )



plt.show()


