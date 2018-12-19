import os
import sys
import csv
import subprocess
import vtk

# M-Rep Manifolds 
import MReps 
import atom 
import numpy as np

# Visualization
import pylab

# PCA for Comparison
from sklearn.decomposition import PCA, KernelPCA

# Stats Model
import statsmodels.api as sm

import matplotlib.pyplot as plt

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
			cmrep_ij_anatomy = MReps.CMRep() 

			for k in range( nAtoms ):
				cmrep_atom_k = atom.cmrep_atom()

				pos = polyData.GetPoint( k )
				rad = polyData.GetPointData().GetArray( "Radius Function" ).GetValue( k )

				cmrep_atom_k.pos = pos
				cmrep_atom_k.rad = rad 

				cmrep_ij_anatomy.AppendAtom( cmrep_atom_k )

			CMRepDataList.append( cmrep_ij_anatomy )
			riskGroupList.append( dataInfoList[i].CAPGroupList[ j ] )
			ageList.append( dataInfoList[i].AgeList[ j ] )
			SubjectList.append( dataInfoList[i].ID )
			CAPList.append( dataInfoList[i].CAPList[j] )
			cnt +=1 

print( CAPList )
print( SubjectList )


print( cnt ) 
print( CMRepDataList[0].nAtoms )
print( CMRepDataList[0].atom_list[0].pos ) 
print( CMRepDataList[0].atom_list[0].rad ) 

print( CMRepDataList[10].nAtoms )
print( CMRepDataList[10].atom_list[10].pos ) 
print( CMRepDataList[10].atom_list[10].rad ) 

# Manifold Dimension
nManDim = CMRepDataList[0].nAtoms

nData = len( CMRepDataList )
nCMRepDim = 4 

# Calculate Intrinsic Mean
print( "====================================" )
print( "Calculate Intrinsic Mean" )
print( "====================================" ) 
max_iter = 1
tol = 0.1

# Initialize
mu = MReps.CMRep()
for k in range( nManDim ):
	mu_atom = atom.cmrep_atom()

	mu_atom.pos[ 0 ] = CMRepDataList[ 0 ].atom_list[k].pos[0]
	mu_atom.pos[ 1 ] = CMRepDataList[ 0 ].atom_list[k].pos[1]
	mu_atom.pos[ 2 ] = CMRepDataList[ 0 ].atom_list[k].pos[2]
	mu_atom.rad = CMRepDataList[ 0 ].atom_list[k].rad
	mu.AppendAtom( mu_atom )

for i in range( max_iter ):
	print( "=================================" ) 
	print( str( i ) + "th Iteration" )
	print( "=================================" )

	for k in range( nManDim ):
		dMu_k = atom.cmrep_tVec()

		for j in range( nData ):
			Log_mu_M_j_k = mu.atom_list[k].LogMap( CMRepDataList[ j ].atom_list[ k ] )

			dMu_k.tVector[ 0 ][0] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][0] )
			dMu_k.tVector[ 0 ][1] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][1] )
			dMu_k.tVector[ 0 ][2] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][2] )

			dMu_k.tVector[ 1 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 1 ] )
			
		Mu_k = mu.atom_list[ k ].ExponentialMap( dMu_k )
		mu.atom_list[ k ] = Mu_k

print( mu.atom_list[0].pos )
print( mu.atom_list[0].rad )

print( "====================================" )
print( " Project data to Intrinsic Mean" )
print( "====================================" ) 

tVec_pos_x_list = []
tVec_pos_y_list = []
tVec_pos_z_list = []

tVec_rad_list = []

# tVec_sphere1_x_list = []
# tVec_sphere1_y_list = []
# tVec_sphere1_z_list = []

# tVec_sphere2_x_list = []
# tVec_sphere2_y_list = []
# tVec_sphere2_z_list = []

for k in range( nManDim ):
	tVec_atom_pos_x_list = []
	tVec_atom_pos_y_list = []
	tVec_atom_pos_z_list = []

	tVec_atom_rad_list = []

	# tVec_atom_sphere1_x_list = []
	# tVec_atom_sphere1_y_list = []
	# tVec_atom_sphere1_z_list = []

	# tVec_atom_sphere2_x_list = []
	# tVec_atom_sphere2_y_list = []
	# tVec_atom_sphere2_z_list = []

	for j in range( nData ):
		tVec_j = mu.atom_list[k].LogMap( CMRepDataList[ j ].atom_list[ k ]  )

		tVec_atom_pos_x_list.append( tVec_j.tVector[0][0] )
		tVec_atom_pos_y_list.append( tVec_j.tVector[0][1] )
		tVec_atom_pos_z_list.append( tVec_j.tVector[0][2] )

		tVec_atom_rad_list.append( tVec_j.tVector[1] )

		# tVec_atom_sphere1_x_list.append( tVec_j.tVector[2][0] )
		# tVec_atom_sphere1_y_list.append( tVec_j.tVector[2][1] )
		# tVec_atom_sphere1_z_list.append( tVec_j.tVector[2][2] )

		# tVec_atom_sphere2_x_list.append( tVec_j.tVector[3][0] )
		# tVec_atom_sphere2_y_list.append( tVec_j.tVector[3][1] )
		# tVec_atom_sphere2_z_list.append( tVec_j.tVector[3][2] )

	tVec_pos_x_list.append( tVec_atom_pos_x_list )
	tVec_pos_y_list.append( tVec_atom_pos_y_list )
	tVec_pos_z_list.append( tVec_atom_pos_z_list )

	tVec_rad_list.append( tVec_atom_rad_list )

	# tVec_sphere1_x_list.append( tVec_atom_sphere1_x_list )
	# tVec_sphere1_y_list.append( tVec_atom_sphere1_y_list )
	# tVec_sphere1_z_list.append( tVec_atom_sphere1_z_list )

	# tVec_sphere2_x_list.append( tVec_atom_sphere2_x_list )
	# tVec_sphere2_y_list.append( tVec_atom_sphere2_y_list )
	# tVec_sphere2_z_list.append( tVec_atom_sphere2_z_list )

print( len( tVec_pos_x_list ) )
print( len( tVec_pos_x_list[0] ) )
print( len( ageList ) )

pos_x_lmodel_list = [] 
pos_y_lmodel_list = [] 
pos_z_lmodel_list = [] 

rad_lmodel_list = [] 

for k in range( nManDim ):
	t_list_sm = sm.add_constant( ageList )

	LS_model_pos_x = sm.OLS( tVec_pos_x_list[ k ], t_list_sm )
	LS_model_pos_y = sm.OLS( tVec_pos_y_list[ k ], t_list_sm )
	LS_model_pos_z = sm.OLS( tVec_pos_z_list[ k ], t_list_sm )

	LS_model_rad = sm.OLS( tVec_rad_list[ k ], t_list_sm )

	pos_x_est = LS_model_pos_x.fit()
	print( pos_x_est.summary() )

	pos_y_est = LS_model_pos_y.fit()
	print( pos_y_est.summary() )

	pos_z_est = LS_model_pos_z.fit()
	print( pos_z_est.summary() )

	rad_est = LS_model_rad.fit()
	print( rad_est.summary() )

	pos_x_lmodel_list.append( pos_x_est )
	pos_y_lmodel_list.append( pos_y_est )
	pos_z_lmodel_list.append( pos_z_est )

	rad_lmodel_list.append( rad_est )

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
nTimePt = 30
est_time_list_i = []
est_rad_pt_list_i = []

regression_output_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2018/GeodesicRegressionResults/CMRep_CTRL_Group/'

for n in range( nTimePt ):
	outFileName = 'CMRep_Regression_CTRL_' + str( n ) + '.vtk' 

	output_path = regression_output_folder_path + outFileName 

	time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

	polyData_t = vtk.vtkPolyData()
	polyData_t.DeepCopy( meanPolyData ) 

	radiusArr_t_vtk = vtk.vtkFloatArray() 
	radiusArr_t_vtk.SetNumberOfValues( polyData_t.GetNumberOfPoints() )
	radiusArr_t_vtk.SetName( 'Radius' )
	
	for k in range( nManDim ):
		# Position
		pos_x_model_i = pos_x_lmodel_list[ k ]
		pos_y_model_i = pos_y_lmodel_list[ k ]
		pos_z_model_i = pos_z_lmodel_list[ k ]


		est_time_fitted_pos_x_tVec_vector = np.add( np.multiply( time_pt, pos_x_model_i.params[1] ), pos_x_model_i.params[0] ) 
		est_time_fitted_pos_y_tVec_vector = np.add( np.multiply( time_pt, pos_y_model_i.params[1] ), pos_y_model_i.params[0] ) 
		est_time_fitted_pos_z_tVec_vector = np.add( np.multiply( time_pt, pos_z_model_i.params[1] ), pos_z_model_i.params[0] ) 

		pos_atom_at_mu = atom.euclidean_atom() 
		pos_atom_at_mu.pt = mu.atom_list[k].pos 

		est_time_fitted_pos_tVec = atom.euclidean_tVec() 
		est_time_fitted_pos_tVec.tVector[0] = est_time_fitted_pos_x_tVec_vector
		est_time_fitted_pos_tVec.tVector[1] = est_time_fitted_pos_y_tVec_vector
		est_time_fitted_pos_tVec.tVector[2] = est_time_fitted_pos_z_tVec_vector

		pt_pos_t = pos_atom_at_mu.ExponentialMap( est_time_fitted_pos_tVec )

		# Radius
		rad_model_i = rad_lmodel_list[ k ] 

		est_time_fitted_rad_tVec_vector = np.add( np.multiply( time_pt, rad_model_i.params[1] ), rad_model_i.params[0] )

		rad_atom_at_mu = atom.pos_real_atom() 
		rad_atom_at_mu.pt = mu.atom_list[k].rad

		est_time_fitted_rad_tVec = atom.pos_real_tVec() 
		est_time_fitted_rad_tVec.tVector = est_time_fitted_rad_tVec_vector
		pt_rad_t = rad_atom_at_mu.ExponentialMap( est_time_fitted_rad_tVec )

		radiusArr_t_vtk.SetValue( k, pt_rad_t.pt )
		polyData_t.GetPoints().SetPoint( k, pt_pos_t.pt  )

	polyData_t.GetPointData().AddArray( radiusArr_t_vtk ) 
	polyData_t.Modified() 

	writer_t = vtk.vtkPolyDataWriter() 
	writer_t.SetFileName( output_path )
	writer_t.SetInputData( polyData_t )
	writer_t.Update()
	writer_t.Write() 

# Matplotlib Plotting 
# Check result at a single atom
atom_idx = 420
rad_model_i = rad_lmodel_list[ atom_idx ] 
rad_data_list_i = [] 

# Data at a single atom
for j in range( nData ):
	rad_data_list_i.append( CMRepDataList[ j ].atom_list[ atom_idx ].rad )

rad_atom_at_mu = atom.pos_real_atom() 
rad_atom_at_mu.pt = mu.atom_list[atom_idx].rad

# Estimated Trend
for n in range( nTimePt ):
	time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0 
	est_time_list_i.append( time_pt ) 

	est_time_fitted_tVec_vector = np.add( np.multiply( time_pt, rad_model_i.params[1] ), rad_model_i.params[0] )
	est_time_fitted_tVec = atom.pos_real_tVec() 
	est_time_fitted_tVec.tVector = est_time_fitted_tVec_vector

	pt_rad_t = rad_atom_at_mu.ExponentialMap( est_time_fitted_tVec )
	est_rad_pt_list_i.append( pt_rad_t.pt ) 	

plt.figure()
plt.scatter( ageList, rad_data_list_i )
plt.plot( est_time_list_i, est_rad_pt_list_i, c='r' )
plt.show() 


# R^2 Statistics
# Intrinsic Mean 
print( "====================================" )
print( "Calculate Intrinsic Mean" )
print( "====================================" ) 
max_iter = 100
tol = 0.1

# Initialize
mu_Frechet = MReps.CMRep()
for k in range( nManDim ):
	mu_atom = atom.cmrep_atom()

	mu_atom.pos[ 0 ] = CMRepDataList[ 0 ].atom_list[k].pos[0]
	mu_atom.pos[ 1 ] = CMRepDataList[ 0 ].atom_list[k].pos[1]
	mu_atom.pos[ 2 ] = CMRepDataList[ 0 ].atom_list[k].pos[2]
	mu_atom.rad = CMRepDataList[ 0 ].atom_list[k].rad
	mu_Frechet.AppendAtom( mu_atom )

for i in range( max_iter ):
	print( "=================================" ) 
	print( str( i ) + "th Iteration" )
	print( "=================================" )

	for k in range( nManDim ):
		dMu_k = atom.cmrep_tVec()

		for j in range( nData ):
			Log_mu_M_j_k = mu_Frechet.atom_list[k].LogMap( CMRepDataList[ j ].atom_list[ k ] )

			dMu_k.tVector[ 0 ][0] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][0] )
			dMu_k.tVector[ 0 ][1] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][1] )
			dMu_k.tVector[ 0 ][2] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 0 ][2] )

			dMu_k.tVector[ 1 ] += ( ( 1.0 / nData ) * Log_mu_M_j_k.tVector[ 1 ] )
			
		Mu_k = mu_Frechet.atom_list[ k ].ExponentialMap( dMu_k )
		mu_Frechet.atom_list[ k ] = Mu_k

print( mu_Frechet.atom_list[0].pos )
print( mu_Frechet.atom_list[0].rad )

# Statistical Significance/Validations 
# Global R^2 Statistics / Mean Squared Error 
# Global Variance w.r.t mean 
variance_mu = 0

# Variance w.r.t the estimated trend
variance_Reg = 0

# RMSE w.r.t. the estiamted trend
rmse = 0

for j in range( nData ):
	dist_sq_i_mu  = 0
	t_j = ageList[ j ]

	for k in range( nManDim ):

		# Distance/Variance from the intrinsic mean
		Log_mu_M_j_k = mu_Frechet.atom_list[k].LogMap( CMRepDataList[ j ].atom_list[ k ] )
		norm_sq_j_k = mu_Frechet.atom_list[k].normSquared( Log_mu_M_j_k )
		dist_sq_i_mu += norm_sq_j_k


		# Estimated 
		pos_x_model_i = pos_x_lmodel_list[ k ]
		pos_y_model_i = pos_y_lmodel_list[ k ]
		pos_z_model_i = pos_z_lmodel_list[ k ]

		est_time_fitted_pos_x_tVec_vector = np.add( np.multiply( t_j, pos_x_model_i.params[1] ), pos_x_model_i.params[0] ) 
		est_time_fitted_pos_y_tVec_vector = np.add( np.multiply( t_j, pos_y_model_i.params[1] ), pos_y_model_i.params[0] ) 
		est_time_fitted_pos_z_tVec_vector = np.add( np.multiply( t_j, pos_z_model_i.params[1] ), pos_z_model_i.params[0] ) 


		# CMRep tangent vector at the anchor point( the intrinsic mean )
		est_time_fitted_cmrep_tVec = atom.cmrep_tVec() 

	 	# CM-Rep Position Tangent vector at the anchor point
		est_time_fitted_cmrep_tVec.tVector[0] = [ est_time_fitted_pos_x_tVec_vector, est_time_fitted_pos_y_tVec_vector, est_time_fitted_pos_z_tVec_vector ]

		# Radius
		rad_model_i = rad_lmodel_list[ k ] 

		est_time_fitted_rad_tVec_vector = np.add( np.multiply( t_j, rad_model_i.params[1] ), rad_model_i.params[0] )

		rad_atom_at_mu = atom.pos_real_atom() 
		rad_atom_at_mu.pt = mu.atom_list[k].rad

		est_time_fitted_rad_tVec = atom.pos_real_tVec() 
		est_time_fitted_rad_tVec.tVector = est_time_fitted_rad_tVec_vector
		pt_rad_t = rad_atom_at_mu.ExponentialMap( est_time_fitted_rad_tVec )

	# Distance/Variance from the intrinsic mean 
	dist_i_mu = np.sqrt( dist_sq_i_mu )
	variance_mu += dist_sq_i_mu

# Variance from the intrinsic mean
variance_mu = variance_mu / float( nData )

print( "Variance w.r.t. mu" )
print( variance_mu )


print( "RMSE w.r.t. mu" )
print( variance_mu )

for n in range( nTimePt ):
	outFileName = 'CMRep_Regression_CTRL_' + str( n ) + '.vtk' 

	output_path = regression_output_folder_path + outFileName 

	time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

	polyData_t = vtk.vtkPolyData()
	polyData_t.DeepCopy( meanPolyData ) 

	radiusArr_t_vtk = vtk.vtkFloatArray() 
	radiusArr_t_vtk.SetNumberOfValues( polyData_t.GetNumberOfPoints() )
	radiusArr_t_vtk.SetName( 'Radius' )
	
	for k in range( nManDim ):
		# Position
		pos_x_model_i = pos_x_lmodel_list[ k ]
		pos_y_model_i = pos_y_lmodel_list[ k ]
		pos_z_model_i = pos_z_lmodel_list[ k ]

		est_time_fitted_pos_x_tVec_vector = np.add( np.multiply( time_pt, pos_x_model_i.params[1] ), pos_x_model_i.params[0] ) 
		est_time_fitted_pos_y_tVec_vector = np.add( np.multiply( time_pt, pos_y_model_i.params[1] ), pos_y_model_i.params[0] ) 
		est_time_fitted_pos_z_tVec_vector = np.add( np.multiply( time_pt, pos_z_model_i.params[1] ), pos_z_model_i.params[0] ) 

		pos_atom_at_mu = atom.euclidean_atom() 
		pos_atom_at_mu.pt = mu.atom_list[k].pos 

		est_time_fitted_pos_tVec = atom.euclidean_tVec() 
		est_time_fitted_pos_tVec.tVector[0] = est_time_fitted_pos_x_tVec_vector
		est_time_fitted_pos_tVec.tVector[1] = est_time_fitted_pos_y_tVec_vector
		est_time_fitted_pos_tVec.tVector[2] = est_time_fitted_pos_z_tVec_vector

		pt_pos_t = pos_atom_at_mu.ExponentialMap( est_time_fitted_pos_tVec )

		# Radius
		rad_model_i = rad_lmodel_list[ k ] 

		est_time_fitted_rad_tVec_vector = np.add( np.multiply( time_pt, rad_model_i.params[1] ), rad_model_i.params[0] )

		rad_atom_at_mu = atom.pos_real_atom() 
		rad_atom_at_mu.pt = mu.atom_list[k].rad

		est_time_fitted_rad_tVec = atom.pos_real_tVec() 
		est_time_fitted_rad_tVec.tVector = est_time_fitted_rad_tVec_vector
		pt_rad_t = rad_atom_at_mu.ExponentialMap( est_time_fitted_rad_tVec )

		radiusArr_t_vtk.SetValue( k, pt_rad_t.pt )
		polyData_t.GetPoints().SetPoint( k, pt_pos_t.pt  )

		