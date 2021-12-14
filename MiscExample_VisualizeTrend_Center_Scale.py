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

from mpl_toolkits.mplot3d import Axes3D 


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

		# Set CM-Rep Abstract Point
		for a, anatomy in enumerate( anatomy_list ):
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
			CMRepDataList[ a ].append( cmrep_ij )

		# CMRepDataList.append( cmrep_ij )
		riskGroupList.append( dataInfoList[i].CAPGroupList[ j ] )
		ageList.append( dataInfoList[i].AgeList[ j ] )
		SubjectList.append( dataInfoList[i].ID )
		CAPList.append( dataInfoList[i].CAPList[j] )
		cnt +=1 

# Manifold Dimension
nManDim = CMRepDataList[0][0].nDim
nData = len( CMRepDataList[0] )

# Intrinsic Mean
# mu = rsm.FrechetMean( CMRepDataList )

# Geodesic Regression
max_iter = 100
step_size = 0.01
step_tol = 1e-6

print( "# of Subjects " ) 
print( len( ageList ) )
# print( CMRepDataList )

# Anatomy list
anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]
# anatomy_list = [ 'left_caudate' ]

# anatomy_list = [ 'right_putamen' ]

# anatomy_list = [ 'right_caudate' ]

########################################
#####        Visualization        ######   
########################################
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)

colors = [ [ 0, 0, 1 ], [ 1.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.5, 1.0 ], [ 0, 0, 1.0 ] ] 
est_colors =[ [ 0, 1, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  
est_lReg_colors =[ [ 1, 0, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  


for a, anatomy in enumerate( anatomy_list ):
	# base, tangent = rsm.LinearizedGeodesicRegression( ageList, CMRepDataList[ a ], max_iter, step_size, step_tol, False, False )

	# base.Write( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_base.rpt" )
	# tangent.Write( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_tangent.tVec" )

	print( "===============================" )
	print( " Reading.... " ) 
	print( "===============================" )

	base = manifolds.cmrep_abstract( nManDim )
	base.Read( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_base.rpt" )

	tangent = manifolds.cmrep_abstract_tVec( nManDim ) 
	tangent.Read( "CMRep_Abstract_LinearizedGeodesicRegression_CTRL_" + anatomy + "_tangent.tVec" )

	print( "===============================" )
	print( " Read " ) 
	print( "===============================" )

	t0 = 20 
	tN = 80

	nTimePt = 60
	est_time_list_i = []
	est_rad_pt_list_i = []

	# Estimated Trend	
	est_time_list = [] 

	# Scale List
	est_scale_list = []

	# Center List
	est_center_x_list = []
	est_center_y_list = []
	est_center_z_list = []

	for n in range( nTimePt ):
		time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

		est_time_list.append( time_pt ) 		

		tVec_t = tangent.ScalarMultiply( time_pt )

		est_cmrep_t = base.ExponentialMap( tVec_t )

		# Euclidean PDM - Scale/Translation Included
		position_matrix_t = est_cmrep_t.GetEuclideanLocations()

		# Euclidean PDM - Scale/Translation Invariant
		position_abstract_matrix_t = est_cmrep_t.GetAbstractEuclideanLocations()

		# Center
		est_center_x_list.append( est_cmrep_t.pt[ 0 ].pt[ 0 ] )
		est_center_y_list.append( est_cmrep_t.pt[ 0 ].pt[ 1 ] )
		est_center_z_list.append( est_cmrep_t.pt[ 0 ].pt[ 2 ] )

		# Scale
		est_scale_list.append( est_cmrep_t.pt[ 1 ].pt[ 0 ] )

	# Data
	data_time_list = []
	data_scale_list = []
	data_center_x_list = []
	data_center_y_list = []
	data_center_z_list = []

	for n in range( nData ):
		data_time_list.append( ageList[ n ] )
		data_cmrep_n = CMRepDataList[ a ][ n ]

		data_center_x_list.append( data_cmrep_n.pt[ 0 ].pt[ 0 ] )
		data_center_y_list.append( data_cmrep_n.pt[ 0 ].pt[ 1 ] )
		data_center_z_list.append( data_cmrep_n.pt[ 0 ].pt[ 2 ] )

		data_scale_list.append( data_cmrep_n.pt[ 1 ].pt[ 0 ] )


	plt.figure( figsize=(6,6 ) )
	plt.scatter( data_time_list, data_scale_list, c=colors[ 3 ], label="Data Scale" )
	plt.plot( est_time_list, est_scale_list, c=est_colors[ 0 ], linestyle="-", linewidth=4, label="Estimated Scale" )
	plt.xlabel('Age(year)') 
	plt.ylabel('Scale Factor' )
	plt.title( anatomy )
	

	fig_center = plt.figure( figsize=(6,6 ) )
	ax_center = Axes3D( fig_center )

	ax_center.plot( est_center_x_list, est_center_y_list, est_center_z_list, label="Estimated Center" )
	ax_center.scatter( data_center_x_list, data_center_y_list, data_center_z_list, label="Data Center" ) 
	ax_center.set_xlabel('X') 
	ax_center.set_ylabel('Y')
	ax_center.set_ylabel('Z')
	ax_center.set_title( anatomy )


plt.legend()
plt.tight_layout()
plt.show()

