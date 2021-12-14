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

S_C_OverAll = 0
S_1_OverAll = 0
S_2_OverAll = 0

dfn = 2 
dfd = nData_CTRL + nData_HIGH - 4


for a, anatomy in enumerate( anatomy_list ):
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

	# base, tangent = rsm.LinearizedGeodesicRegression( ageList, CMRepDataList[ a ], max_iter, step_size, step_tol, False, False )

	# base.Write( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_base.rpt" )
	# tangent.Write( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_tangent.tVec" )

	print( "===============================" )
	print( " Reading.... " ) 
	print( "===============================" )

	base_CTRL = manifolds.cmrep_abstract_normal( nManDim )
	base_CTRL.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_CTRL_" + anatomy + "_base.rpt" )

	tangent_CTRL = manifolds.cmrep_abstract_normal_tVec( nManDim ) 
	tangent_CTRL.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_CTRL_" + anatomy + "_tangent.tVec" )

	base_HIGH = manifolds.cmrep_abstract_normal( nManDim )
	base_HIGH.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_HIGH_" + anatomy + "_base.rpt" )

	tangent_HIGH = manifolds.cmrep_abstract_normal_tVec( nManDim ) 
	tangent_HIGH.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_HIGH_" + anatomy + "_tangent.tVec" )

	base_All = manifolds.cmrep_abstract_normal( nManDim )
	base_All.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_CTRLHIGH_" + anatomy + "_base.rpt" )

	tangent_All = manifolds.cmrep_abstract_normal_tVec( nManDim )
	tangent_All.Read( "CMRep_Abstract_Normal_LinearizedGeodesicRegression_CTRLHIGH_" + anatomy + "_tangent.tVec" )

	# Combined Squared Residuals
	# Anatomy-wise Overall
	S_C = 0
	# Center 
	S_C_center = 0
	# Scale 
	S_C_scale = 0
	# PreShape
	S_C_PreShape = 0
	# Radius Field
	S_C_Radius = np.zeros( nManDim )
	# Normal 1 Field
	S_C_Normal1 = np.zeros( nManDim )
	# Normal 2 Field
	S_C_Normal2 = np.zeros( nManDim )

	# Control Group Squared Residuals
	S_1 = 0
	# Center 
	S_1_center = 0
	# Scale 
	S_1_scale = 0
	# PreShape
	S_1_PreShape = 0
	# Radius Field
	S_1_Radius = np.zeros( nManDim )
	# Normal 1 Field
	S_1_Normal1 = np.zeros( nManDim )
	# Normal 2 Field
	S_1_Normal2 = np.zeros( nManDim )

	# High risk group squared residuals
	S_2 = 0 
	# Center 
	S_2_center = 0
	# Scale 
	S_2_scale = 0
	# PreShape
	S_2_PreShape = 0
	# Radius Field
	S_2_Radius = np.zeros( nManDim )
	# Normal 1 Field
	S_2_Normal1 = np.zeros( nManDim )
	# Normal 2 Field
	S_2_Normal2 = np.zeros( nManDim )

	# Local Chow Statistics Array 
	Chow_Radius = np.zeros( nManDim )
	Chow_Normal1 = np.zeros( nManDim )
	Chow_Normal2 = np.zeros( nManDim )

	# Local Chow Statistics P-Value
	P_Chow_Radius = np.zeros( nManDim )
	P_Chow_Normal1 = np.zeros( nManDim )
	P_Chow_Normal2 = np.zeros( nManDim )


	# Calculate mean area and mean radius 
	mean_area_s = 0
	mean_radius = 0

	for d in range( nData_CTRL ):
		mean_area_s += ( CMRepDataList_CTRL[ a ][ d ].pt[ 1 ].pt[ 0 ] / float( nData_All ) )
		CMRepDataList_CTRL[ a ][ d ].UpdateMeanRadius()
		mean_radius += ( CMRepDataList_CTRL[ a ][ d ].meanRadius / float( nData_All ) )

	for d in range( nData_HIGH ):
		mean_area_s += ( CMRepDataList_HIGH[ a ][ d ].pt[ 1 ].pt[ 0 ] / float( nData_All ) )
		CMRepDataList_HIGH[ a ][ d ].UpdateMeanRadius()
		mean_radius += ( CMRepDataList_HIGH[ a ][ d ].meanRadius / float( nData_All ) )

	# Control Group Data
	for d in range( len( CMRepDataList_CTRL[ a ] ) ):
		ctrl_t_d = ageList_CTRL[ d ]

		# Calculate Variance from overall Trajectory
		tVec_All_t = tangent_All.ScalarMultiply( ctrl_t_d )
		est_cmrep_All_t = base_All.ExponentialMap( tVec_All_t )

		tVec_est_All_to_y_d = est_cmrep_All_t.LogMap( CMRepDataList_CTRL[ a ][ d ] )
		tVec_est_All_to_y_d.SetMeanRadius( mean_radius )
		tVec_est_All_to_y_d.SetMeanScale( np.sqrt( mean_area_s ) * ( 1.0 / 3.0 ) )

		S_C += ( tVec_est_All_to_y_d.normSquared() )
		# Center Residual
		S_C_center += tVec_est_All_to_y_d.tVector[ 0 ].normSquared()  

		# Scale Residual
		S_C_scale += tVec_est_All_to_y_d.tVector[ 1 ].normSquared()  

		# Scale Residual
		S_C_PreShape += tVec_est_All_to_y_d.tVector[ 2 ].normSquared()  

		# Calculate Variance from overall Trajectory
		tVec_CTRL_t = tangent_CTRL.ScalarMultiply( ctrl_t_d )
		est_cmrep_CTRL_t = base_CTRL.ExponentialMap( tVec_CTRL_t )

		tVec_est_CTRL_to_y_d = est_cmrep_CTRL_t.LogMap( CMRepDataList_CTRL[ a ][ d ] )
		tVec_est_CTRL_to_y_d.SetMeanRadius( mean_radius )
		tVec_est_CTRL_to_y_d.SetMeanScale( np.sqrt( mean_area_s ) * ( 1.0 / 3.0 ) )

		S_1 += ( tVec_est_CTRL_to_y_d.normSquared() )

		# Center Residual
		S_1_center += tVec_est_CTRL_to_y_d.tVector[ 0 ].normSquared()  

		# Scale Residual
		S_1_scale += tVec_est_CTRL_to_y_d.tVector[ 1 ].normSquared()  

		# Scale Residual
		S_1_PreShape += tVec_est_CTRL_to_y_d.tVector[ 2 ].normSquared()  

		# Radius Fields
		# Boundary Normal 1 Fields
		# Boundary Normal 2 Fields
		for i_a in range( nManDim ):
			tVec_est_All_to_y_d_rad_i_a = manifolds.pos_real_tVec( 1 )
			tVec_est_All_to_y_d_rad_i_a.SetTangentVector( tVec_est_All_to_y_d.tVector[ 3 ].tVector[ i_a ] )

			S_C_Radius[ i_a ] += tVec_est_All_to_y_d_rad_i_a.normSquared()
			S_C_Normal1[ i_a ] += tVec_est_All_to_y_d.tVector[ 4 ][ i_a ].normSquared()
			S_C_Normal2[ i_a ] += tVec_est_All_to_y_d.tVector[ 5 ][ i_a ].normSquared()

			tVec_est_CTRL_to_y_d_rad_i_a = manifolds.pos_real_tVec( 1 )
			tVec_est_CTRL_to_y_d_rad_i_a.SetTangentVector( tVec_est_CTRL_to_y_d.tVector[ 3 ].tVector[ i_a ] )

			S_1_Radius[ i_a ] += tVec_est_CTRL_to_y_d_rad_i_a.normSquared()
			S_1_Normal1[ i_a ] += tVec_est_CTRL_to_y_d.tVector[ 4 ][ i_a ].normSquared()
			S_1_Normal2[ i_a ] += tVec_est_CTRL_to_y_d.tVector[ 5 ][ i_a ].normSquared()

	# High Risk Group Data
	for d in range( len( CMRepDataList_HIGH[ a ] ) ):
		high_t_d = ageList_HIGH[ d ]

		# Calculate Variance from overall Trajectory
		tVec_All_t = tangent_All.ScalarMultiply( high_t_d )
		est_cmrep_All_t = base_All.ExponentialMap( tVec_All_t )

		tVec_est_All_to_y_d = est_cmrep_All_t.LogMap( CMRepDataList_HIGH[ a ][ d ] )
		tVec_est_All_to_y_d.SetMeanRadius( mean_radius )
		tVec_est_All_to_y_d.SetMeanScale( np.sqrt( mean_area_s ) * ( 1.0 / 3.0 ) )

		S_C += ( tVec_est_All_to_y_d.normSquared() )
		# Center Residual
		S_C_center += tVec_est_All_to_y_d.tVector[ 0 ].normSquared()  

		# Scale Residual
		S_C_scale += tVec_est_All_to_y_d.tVector[ 1 ].normSquared()  

		# Scale Residual
		S_C_PreShape += tVec_est_All_to_y_d.tVector[ 2 ].normSquared()  

		# Calculate Variance from overall Trajectory
		tVec_HIGH_t = tangent_HIGH.ScalarMultiply( high_t_d )
		est_cmrep_HIGH_t = base_HIGH.ExponentialMap( tVec_HIGH_t )

		tVec_est_HIGH_to_y_d = est_cmrep_HIGH_t.LogMap( CMRepDataList_HIGH[ a ][ d ] )
		tVec_est_HIGH_to_y_d.SetMeanRadius( mean_radius )
		tVec_est_HIGH_to_y_d.SetMeanScale( np.sqrt( mean_area_s ) * ( 1.0 / 3.0 ) )

		S_2 += ( tVec_est_HIGH_to_y_d.normSquared() )

		# Center Residual
		S_2_center += tVec_est_HIGH_to_y_d.tVector[ 0 ].normSquared()  

		# Scale Residual
		S_2_scale += tVec_est_HIGH_to_y_d.tVector[ 1 ].normSquared()  

		# Scale Residual
		S_2_PreShape += tVec_est_HIGH_to_y_d.tVector[ 2 ].normSquared()  

		for i_a in range( nManDim ):
			tVec_est_All_to_y_d_rad_i_a = manifolds.pos_real_tVec( 1 )
			tVec_est_All_to_y_d_rad_i_a.SetTangentVector( tVec_est_All_to_y_d.tVector[ 3 ].tVector[ i_a ] )

			S_C_Radius[ i_a ] += tVec_est_All_to_y_d_rad_i_a.normSquared()
			S_C_Normal1[ i_a ] += tVec_est_All_to_y_d.tVector[ 4 ][ i_a ].normSquared()
			S_C_Normal2[ i_a ] += tVec_est_All_to_y_d.tVector[ 5 ][ i_a ].normSquared()

			tVec_est_HIGH_to_y_d_rad_i_a = manifolds.pos_real_tVec( 1 )
			tVec_est_HIGH_to_y_d_rad_i_a.SetTangentVector( tVec_est_HIGH_to_y_d.tVector[ 3 ].tVector[ i_a ] )

			S_2_Radius[ i_a ] += tVec_est_HIGH_to_y_d_rad_i_a.normSquared()
			S_2_Normal1[ i_a ] += tVec_est_HIGH_to_y_d.tVector[ 4 ][ i_a ].normSquared()
			S_2_Normal2[ i_a ] += tVec_est_HIGH_to_y_d.tVector[ 5 ][ i_a ].normSquared()

	S_C_OverAll += S_C
	S_1_OverAll += S_1
	S_2_OverAll += S_2

	chow_stat_anatomy = ( ( S_C - ( S_1 + S_2 ) ) / 2 ) / ( ( S_1 + S_2 ) / ( nData_CTRL + nData_HIGH - 4 ) )
	p_anatomy = 1.0 - stats.f.cdf(chow_stat_anatomy, dfn, dfd )

	print( "================================================" )
	print( "Anatomy Chow Stat: " + anatomy)
	print( chow_stat_anatomy )
	print( "Anatomy p-value : " + anatomy )
	print( p_anatomy )

	chow_stat_center =  ( ( S_C_center - ( S_1_center + S_2_center ) ) / 2 ) / ( ( S_1_center + S_2_center ) / ( nData_CTRL + nData_HIGH - 4 ) )
	p_center = 1.0 - stats.f.cdf(chow_stat_center, dfn, dfd )

	chow_stat_scale =  ( ( S_C_scale - ( S_1_scale + S_2_scale ) ) / 2 ) / ( ( S_1_scale + S_2_scale ) / ( nData_CTRL + nData_HIGH - 4 ) )
	p_scale = 1.0 - stats.f.cdf(chow_stat_scale, dfn, dfd )

	chow_stat_preshape =  ( ( S_C_PreShape - ( S_1_PreShape + S_2_PreShape ) ) / 2 ) / ( ( S_1_PreShape + S_2_PreShape ) / ( nData_CTRL + nData_HIGH - 4 ) )
	p_preshape = 1.0 - stats.f.cdf(chow_stat_preshape, dfn, dfd )

	print( "Anatomy Center Chow Stat: " + anatomy )
	print( chow_stat_center )
	print( "Anatomy Center p-value: " + anatomy )
	print( p_center )

	print( "Anatomy Scale Chow Stat: " + anatomy )
	print( chow_stat_scale )
	print( "Anatomy Scale p-value: " + anatomy )
	print( p_scale )

	print( "Anatomy PreShape Chow Stat: " + anatomy )
	print( chow_stat_preshape )
	print( "Anatomy Preshape p-value: " + anatomy )
	print( p_preshape )
	print( "================================================" )


	for i_a in range( nManDim ):
		Chow_Radius[ i_a ] = ( ( S_C_Radius[ i_a ] - ( S_1_Radius[ i_a ] + S_2_Radius[ i_a ] ) ) / 2 ) / ( ( S_1_Radius[ i_a ] + S_2_Radius[ i_a ] ) / ( nData_CTRL + nData_HIGH - 4 ) )
		Chow_Normal1[ i_a ] = ( ( S_C_Normal1[ i_a ] - ( S_1_Normal1[ i_a ] + S_2_Normal1[ i_a ] ) ) / 2 ) / ( ( S_1_Normal1[ i_a ] + S_2_Normal1[ i_a ] ) / ( nData_CTRL + nData_HIGH - 4 ) )
		Chow_Normal2[ i_a ] = ( ( S_C_Normal2[ i_a ] - ( S_1_Normal2[ i_a ] + S_2_Normal2[ i_a ] ) ) / 2 ) / ( ( S_1_Normal2[ i_a ] + S_2_Normal2[ i_a ] ) / ( nData_CTRL + nData_HIGH - 4 ) )

		P_Chow_Radius[ i_a ] = 1.0 - stats.f.cdf(Chow_Radius[ i_a ], dfn, dfd ) 
		P_Chow_Normal1[ i_a ] = 1.0 - stats.f.cdf(Chow_Normal1[ i_a ], dfn, dfd ) 
		P_Chow_Normal2[ i_a ] = 1.0 - stats.f.cdf(Chow_Normal2[ i_a ], dfn, dfd ) 


	outputFolder = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_CTRLHIGH_Group_NewCode_Linearized_Complex/ChowStatistics/"
	outputFileName = "CMRep_Abstract_Normal_Regression_CTRL_HIGH_Chow_Stats_" + anatomy + ".vtk"

	outputPath = outputFolder + outputFileName

	# Write Local Chow Statistics (Radius, Bndr Normal1/2) on a template polyData 
	polyData_t = vtk.vtkPolyData()
	polyData_t.DeepCopy( meanPolyData ) 

	radChowArr = vtk.vtkFloatArray()
	radChowArr.SetName( "Chow - Radius" )	
	radChowArr.SetNumberOfValues( nManDim )

	normal1ChowArr = vtk.vtkFloatArray()
	normal1ChowArr.SetName( "Chow - Normal 1" )	
	normal1ChowArr.SetNumberOfValues( nManDim )

	normal2ChowArr = vtk.vtkFloatArray()
	normal2ChowArr.SetName( "Chow - Normal 2" )	
	normal2ChowArr.SetNumberOfValues( nManDim )

	P_radChowArr = vtk.vtkFloatArray()
	P_radChowArr.SetName( "Chow p-value - Radius" )	
	P_radChowArr.SetNumberOfValues( nManDim )

	P_normal1ChowArr = vtk.vtkFloatArray()
	P_normal1ChowArr.SetName( "Chow p-value - Normal 1" )	
	P_normal1ChowArr.SetNumberOfValues( nManDim )

	P_normal2ChowArr = vtk.vtkFloatArray()
	P_normal2ChowArr.SetName( "Chow p-value - Normal 2" )	
	P_normal2ChowArr.SetNumberOfValues( nManDim )

	# Position - Template CTRL Age 20
	position_matrix_0 = base_CTRL.GetEuclideanLocations()
	for k in range( nManDim ):
		# Position Points + Center Change
		polyData_t.GetPoints().SetPoint( k, position_matrix_0[ k, : ] )
		radChowArr.SetValue( k, Chow_Radius[ k ] )
		normal1ChowArr.SetValue( k, Chow_Normal1[ k ] )
		normal2ChowArr.SetValue( k, Chow_Normal2[ k ] )

		P_radChowArr.SetValue( k, P_Chow_Radius[ k ] )
		P_normal1ChowArr.SetValue( k, P_Chow_Normal1[ k ] )
		P_normal2ChowArr.SetValue( k, P_Chow_Normal2[ k ] )

	polyData_t.GetPointData().AddArray( radChowArr )
	polyData_t.GetPointData().AddArray( normal1ChowArr )
	polyData_t.GetPointData().AddArray( normal2ChowArr )

	polyData_t.GetPointData().AddArray( P_radChowArr )
	polyData_t.GetPointData().AddArray( P_normal1ChowArr )
	polyData_t.GetPointData().AddArray( P_normal2ChowArr )

	polyData_t.Modified() 		

	writer_t = vtk.vtkPolyDataWriter() 
	writer_t.SetFileName( outputPath )
	writer_t.SetInputData( polyData_t )
	writer_t.Update()
	writer_t.Write() 

S_C_OverAll += S_C
S_1_OverAll += S_1
S_2_OverAll += S_2

chow_stat_overall = ( ( S_C_OverAll - ( S_1_OverAll + S_2_OverAll ) ) / 2 ) / ( ( S_1_OverAll + S_2_OverAll ) / ( nData_CTRL + nData_HIGH - 4 ) )

print( "========================================" )
print( "Overall Chow Statistics" )
print( chow_stat_overall )

alpha = 0.05 
p_value = 1.0 - stats.f.cdf(chow_stat_overall, dfn, dfd )

print( "Overall p-value" ) 
print( p_value )
print( "========================================" )
