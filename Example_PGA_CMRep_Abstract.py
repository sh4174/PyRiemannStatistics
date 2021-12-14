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

		for anatomy in anatomy_list:
			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"
			anatomy_cmrep_surface_cen_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med_centered.vtk" 
			anatomy_cmrep_surface_cen_norm_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med_centered_normalized.vtk" 

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

			for k in range( nAtoms ):
				pos = polyData.GetPoint( k )
				pos_cen = np.subtract( pos, cenOfMass )
				polyData_cen.GetPoints().SetPoint( k, pos_cen )

				pos_centered_arr.append( pos_cen )				

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
			cmrep_ij = manifolds.cmrep_abstract( nAtoms )
			
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
			pt_abs = [ center_ij, scale_ij, pos_ij, rad_ij ]
			# print( "CMRep Abstract Point" )
			cmrep_ij.SetPoint( pt_abs )

			CMRepDataList.append( cmrep_ij )
			riskGroupList.append( dataInfoList[i].CAPGroupList[ j ] )
			ageList.append( dataInfoList[i].AgeList[ j ] )
			SubjectList.append( dataInfoList[i].ID )
			CAPList.append( dataInfoList[i].CAPList[j] )
			cnt +=1 

# Manifold Dimension
nManDim = CMRepDataList[0].nDim
nData = len( CMRepDataList )

# Intrinsic Mean
# mu = rsm.FrechetMean( CMRepDataList )

#######################
#    PGA      #
#######################

# Geodesic Regression
max_iter = 100
step_size = 0.01
step_tol = 1e-6

w, v, mu = rsm.TangentPGA( CMRepDataList, max_iter, step_tol, step_size )

print( w )
print( v )


print( "=======================================" )
print( " PGA Results " )
print( "=======================================" )

sum_w = np.sum( w )
w_accum = 0

print( "=======================================" )
print( "First 10 Components Eigenvalues" )
print( "=======================================" )

for i in range( 10 ):
	w_accum += w[ i ]
	print( str( i + 1 ) + "th PG : " + str( w_accum / sum_w ) )

print( "=============================" ) 
print( " PGA Done " ) 
print( "=============================" )

print( "====================================" )
print( " Project data to PGs" )
print( "====================================" ) 

tVec_list = []
w_1 = [] 
w_2 = []
w_3 = []
w_4 = [] 
w_5 = []
w_6 = []
w_7 = []
w_8 = [] 
w_9 = []
w_10 = []


# Covariance matrix on a tangent vector space 
nCenterDim = CMRepDataList[ 0 ].pt[ 0 ].nDim
nScaleDim = CMRepDataList[ 0 ].pt[ 1 ].nDim
nPreShapeDim = CMRepDataList[ 0 ].pt[ 2 ].nDim 
nRadiusDim = CMRepDataList[ 0 ].pt[ 3 ].nDim 

# Total Dimension
nManDim_Cov = nCenterDim + nScaleDim + nPreShapeDim + nRadiusDim

for j in range( nData ):
	print( CMRepDataList[ j ].meanRadius )

	tVec_j = mu.LogMap( CMRepDataList[ j ] )

	u_j_mat = np.zeros( [ 1, nManDim_Cov ] )

	u_j_mat_center = np.asarray( tVec_j.tVector[ 0 ].tVector ).flatten()
	u_j_mat_scale = np.asarray( tVec_j.tVector[ 1 ].tVector ).flatten()
	u_j_mat_preshape = np.asarray( tVec_j.tVector[ 2 ].tVector ).flatten()
	u_j_mat_radius = np.asarray( tVec_j.tVector[ 3 ].tVector ).flatten()
	
	for d in range( nCenterDim ):
		u_j_mat[ 0, d ] = u_j_mat_center[ d ]

	for d in range( nScaleDim ):
		# u_j_mat[ 0, d + nCenterDim ] = CMRepDataList[ j ].meanRadius * u_j_mat_scale[ d ]
		u_j_mat[ 0, d + nCenterDim ] = u_j_mat_scale[ d ]

	for d in range( nPreShapeDim ):
		# u_j_mat[ 0, d + nCenterDim + nScaleDim ] = CMRepDataList[ j ].meanRadius * u_j_mat_preshape[ d ]
		u_j_mat[ 0, d + nCenterDim + nScaleDim ] = u_j_mat_preshape[ d ]

	for d in range( nRadiusDim ):
		# u_j_mat[ 0, d + nCenterDim + nScaleDim + nPreShapeDim ] = CMRepDataList[ j ].meanRadius * u_j_mat_radius[ d ]
		u_j_mat[ 0, d + nCenterDim + nScaleDim + nPreShapeDim ] = u_j_mat_radius[ d ]

	w_1_j = np.dot( u_j_mat, v[ :, 0 ] )
	w_2_j = np.dot( u_j_mat, v[ :, 1 ] )
	w_3_j = np.dot( u_j_mat, v[ :, 2 ] )
	w_4_j = np.dot( u_j_mat, v[ :, 3 ] )
	w_5_j = np.dot( u_j_mat, v[ :, 4 ] )
	w_6_j = np.dot( u_j_mat, v[ :, 5 ] )
	w_7_j = np.dot( u_j_mat, v[ :, 6 ] )
	w_8_j = np.dot( u_j_mat, v[ :, 7 ] )
	w_9_j = np.dot( u_j_mat, v[ :, 8 ] )
	w_10_j = np.dot( u_j_mat, v[ :, 9 ] )

	w_1.append( w_1_j )
	w_2.append( w_2_j )
	w_3.append( w_3_j )
	w_4.append( w_4_j )
	w_5.append( w_5_j )
	w_6.append( w_6_j )
	w_7.append( w_7_j )
	w_8.append( w_8_j )
	w_9.append( w_9_j )
	w_10.append( w_10_j )

w_1 = np.asarray( w_1 ) 
w_2 = np.asarray( w_2 )
w_3 = np.asarray( w_3 ) 
w_4 = np.asarray( w_4 )
w_5 = np.asarray( w_5 ) 
w_6 = np.asarray( w_6 )
w_7 = np.asarray( w_7 ) 
w_8 = np.asarray( w_8 )
w_9 = np.asarray( w_9 ) 
w_10 = np.asarray( w_10 )

###########################################
#####        PGA Results Write       ######   
###########################################
outputFolderPath = "/media/shong/IntHard1/4DAnalysis/ICCV2019/PGA_Results/CMRep_Abstract_NoCommensuration/"
muFilePath = outputFolderPath + "IntrinsicMean.vtk"

# Intrinsic Mean 
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

muPolyData = vtk.vtkPolyData()
muPolyData.DeepCopy( meanPolyData )

radiusArr_mu = vtk.vtkFloatArray() 
radiusArr_mu.SetNumberOfValues( muPolyData.GetNumberOfPoints() )
radiusArr_mu.SetName( 'Radius' )

position_matrix_mu = mu.GetEuclideanLocations()

for k in range( nManDim ):
	muPolyData.GetPoints().SetPoint( k, position_matrix_mu[ k, : ] )

	radiusArr_mu.SetValue( k, mu.pt[3].pt[ k ] )

muPolyData.GetPointData().AddArray( radiusArr_mu )
muPolyData.Modified()

muWriter = vtk.vtkPolyDataWriter()
muWriter.SetFileName( muFilePath )
muWriter.SetInputData( muPolyData )
muWriter.Update()
muWriter.Write()

# PG Trend Write 
for k in range( 10 ):
	nLambda = 30
	lambda_list = np.linspace( -3 * np.sqrt( w[ k ] ), 3 * np.sqrt( w[ k ] ), nLambda )

	PG_pt_list = []

	tVec_v_k = manifolds.cmrep_abstract_tVec( nManDim )
	tVec_v_k.SetTangentVectorFromArray( np.squeeze( np.asarray( v[ :, k ] ) ) )

	for n in range( nLambda ):
		lambda_n = lambda_list[ n ]

		tVec_v_k_n = tVec_v_k.ScalarMultiply( lambda_n )

		PG_pt = mu.ExponentialMap( tVec_v_k_n )
		PG_pt_position_matrix = PG_pt.GetEuclideanLocations()

		PGPolyData = vtk.vtkPolyData()
		PGPolyData.DeepCopy( meanPolyData )

		PGRadiusArray = vtk.vtkFloatArray()
		PGRadiusArray.SetNumberOfValues( PGPolyData.GetNumberOfPoints() )
		PGRadiusArray.SetName( 'Radius' )


		for p in range( nManDim ):
			PGPolyData.GetPoints().SetPoint( p, PG_pt_position_matrix[ p, : ] )

			PGRadiusArray.SetValue( p, PG_pt.pt[ 3 ].pt[ p ] )

		PGPolyData.GetPointData().AddArray( PGRadiusArray )
		PGPolyData.Modified()


		PGFilePath = outputFolderPath + "PG_" + str( k ) + "_Lambda" + str( n ) + ".vtk"
		PG_Writer = vtk.vtkPolyDataWriter()
		PG_Writer.SetFileName( PGFilePath )
		PG_Writer.SetInputData( PGPolyData )
		PG_Writer.Update()
		PG_Writer.Write()

#################################################################
#####        Linearized Geodesic Regression Results        ######   
#################################################################

print( "===============================" )
print( " LReg Results Reading.... " ) 
print( "===============================" )

base = manifolds.cmrep_abstract( nManDim )
base.Read( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_LC_base.rpt" )

tangent = manifolds.cmrep_abstract_tVec( nManDim ) 
tangent.Read(  "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_LC_tangent.tVec"  )
t0 = 20 
tN = 80

nTimePt = 60

LReg_time_array = []
LReg_coeff_PG1_array = []
LReg_coeff_PG2_array = []
LReg_coeff_PG3_array = []
LReg_coeff_PG4_array = []
LReg_coeff_PG5_array = []
LReg_coeff_PG6_array = []
LReg_coeff_PG7_array = []
LReg_coeff_PG8_array = []
LReg_coeff_PG9_array = []
LReg_coeff_PG10_array = []

for n in range( nTimePt ):
	time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

	LReg_time_array.append( time_pt ) 

	tVec_t = tangent.ScalarMultiply( time_pt )

	est_cmrep_t = base.ExponentialMap( tVec_t )

	tVec_t_at_mu = mu.LogMap( est_cmrep_t )

	u_j_mat = np.zeros( [ 1, nManDim_Cov ] )

	u_j_mat_center = np.asarray( tVec_t_at_mu.tVector[ 0 ].tVector ).flatten()
	u_j_mat_scale = np.asarray( tVec_t_at_mu.tVector[ 1 ].tVector ).flatten()
	u_j_mat_preshape = np.asarray( tVec_t_at_mu.tVector[ 2 ].tVector ).flatten()
	u_j_mat_radius = np.asarray( tVec_t_at_mu.tVector[ 3 ].tVector ).flatten()
	
	for d in range( nCenterDim ):
		u_j_mat[ 0, d ] = u_j_mat_center[ d ]

	for d in range( nScaleDim ):
		# u_j_mat[ 0, d + nCenterDim ] = CMRepDataList[ j ].meanRadius * u_j_mat_scale[ d ]
		u_j_mat[ 0, d + nCenterDim ] = u_j_mat_scale[ d ]

	for d in range( nPreShapeDim ):
		# u_j_mat[ 0, d + nCenterDim + nScaleDim ] = CMRepDataList[ j ].meanRadius * u_j_mat_preshape[ d ]
		u_j_mat[ 0, d + nCenterDim + nScaleDim ] = u_j_mat_preshape[ d ]

	for d in range( nRadiusDim ):
		# u_j_mat[ 0, d + nCenterDim + nScaleDim + nPreShapeDim ] = CMRepDataList[ j ].meanRadius * u_j_mat_radius[ d ]
		u_j_mat[ 0, d + nCenterDim + nScaleDim + nPreShapeDim ] = u_j_mat_radius[ d ]

	w_1_j = np.dot( u_j_mat, v[ :, 0 ] )
	w_2_j = np.dot( u_j_mat, v[ :, 1 ] )
	w_3_j = np.dot( u_j_mat, v[ :, 2 ] )
	w_4_j = np.dot( u_j_mat, v[ :, 3 ] )
	w_5_j = np.dot( u_j_mat, v[ :, 4 ] )
	w_6_j = np.dot( u_j_mat, v[ :, 5 ] )
	w_7_j = np.dot( u_j_mat, v[ :, 6 ] )
	w_8_j = np.dot( u_j_mat, v[ :, 7 ] )
	w_9_j = np.dot( u_j_mat, v[ :, 8 ] )
	w_10_j = np.dot( u_j_mat, v[ :, 9 ] )

	LReg_coeff_PG1_array.append( w_1_j )
	LReg_coeff_PG2_array.append( w_2_j )
	LReg_coeff_PG3_array.append( w_3_j )
	LReg_coeff_PG4_array.append( w_4_j )
	LReg_coeff_PG5_array.append( w_5_j )
	LReg_coeff_PG6_array.append( w_6_j )
	LReg_coeff_PG7_array.append( w_7_j )
	LReg_coeff_PG8_array.append( w_8_j )
	LReg_coeff_PG9_array.append( w_9_j )
	LReg_coeff_PG10_array.append( w_10_j )

LReg_coeff_PG1_array = np.asarray( LReg_coeff_PG1_array )
LReg_coeff_PG2_array = np.asarray( LReg_coeff_PG2_array )
LReg_coeff_PG3_array = np.asarray( LReg_coeff_PG3_array )
LReg_coeff_PG4_array = np.asarray( LReg_coeff_PG4_array )
LReg_coeff_PG5_array = np.asarray( LReg_coeff_PG5_array )
LReg_coeff_PG6_array = np.asarray( LReg_coeff_PG6_array )
LReg_coeff_PG7_array = np.asarray( LReg_coeff_PG7_array )
LReg_coeff_PG8_array = np.asarray( LReg_coeff_PG8_array )
LReg_coeff_PG9_array = np.asarray( LReg_coeff_PG9_array )
LReg_coeff_PG10_array = np.asarray( LReg_coeff_PG10_array )



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

# Regression line on PG 1 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_1, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG1_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 1 Coeff' )
plt.title( "PG1" )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.legend()
plt.tight_layout()

# Regression line on PG 2 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_2, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG2_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.ylabel('PG 2 Coeff' )
plt.title( "PG2" )
plt.legend()
plt.tight_layout()


# Regression line on PG 3 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_3, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG3_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 3 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG3" )
plt.legend()
plt.tight_layout()


# Regression line on PG 4 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_4, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG4_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 4 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG4" )
plt.legend()
plt.tight_layout()


# Regression line on PG 5 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_5, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG5_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 5 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG5" )
plt.legend()
plt.tight_layout()


# Regression line on PG 6 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_6, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG6_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 6 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG6" )
plt.legend()
plt.tight_layout()


# Regression line on PG 7 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_7, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG7_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 7 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG7" )
plt.legend()
plt.tight_layout()


# Regression line on PG 8 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_8, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG8_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 8 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG8" )
plt.legend()
plt.tight_layout()

# Regression line on PG 9 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_9, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG9_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 9 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG9" )
plt.legend()
plt.tight_layout()


# Regression line on PG 10 Space over time 
plt.figure( figsize=( 6, 6 ) )
plt.scatter( ageList, w_10, c=colors[ 0 ], alpha=0.5, label="Data" )
plt.plot( LReg_time_array, LReg_coeff_PG10_array, c=est_colors[ 0 ], label="LGReg" ) 
plt.xlabel('Age') 
plt.ylabel('PG 10 Coeff' )
plt.xlim( 20, 80 )
plt.ylim( -10, 10 )
plt.title( "PG10" )
plt.legend()
plt.tight_layout()


plt.show()