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


refFilePath = "/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_CTRL_Group_NewCode_Linearized_Complex/CMRep_Abstract_Normal_Regression_CTRL_left_caudate_0.vtk"

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

	regression_output_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_CTRL_Group_NewCode_Linearized_Complex/CenterChange/'

	for n in range( nTimePt ):
		outFileName = 'CMRep_Abstract_Normal_Regression_CTRL_' + anatomy + '_Center_Change_' + str( n ) + '.vtk' 

		output_path = regression_output_folder_path + outFileName 

		time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

		polyData_t = vtk.vtkPolyData()
		polyData_t.DeepCopy( meanPolyData ) 

		# tVec_0 = tangent.ScalarMultiply( 0 )		
		# est_cmrep_0 = base.ExponentialMap( tVec_0 )

		position_matrix_0 = base.GetEuclideanLocations()

		tVec_t = tangent.ScalarMultiply( time_pt )
		est_cmrep_t = base.ExponentialMap( tVec_t )

		for k in range( nManDim ):
			# Position Points + Center Change
			polyData_t.GetPoints().SetPoint( k, np.add( position_matrix_0[ k, : ], np.subtract( est_cmrep_t.pt[ 0 ].pt, base.pt[ 0 ].pt ) ) )

		polyData_t.Modified()

		writer_t = vtk.vtkPolyDataWriter() 
		writer_t.SetFileName( output_path )
		writer_t.SetInputData( polyData_t )
		writer_t.Update()
		writer_t.Write() 

	# Radius Change
	radius_change_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_CTRL_Group_NewCode_Linearized_Complex/RadiusChange/'

	for n in range( nTimePt ):
		outFileName = 'CMRep_Abstract_Normal_Regression_CTRL_' + anatomy + '_Radius_Change_' + str( n ) + '.vtk' 

		output_path = radius_change_folder_path + outFileName 

		time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

		polyData_t = vtk.vtkPolyData()
		polyData_t.DeepCopy( meanPolyData ) 

		# tVec_0 = tangent.ScalarMultiply( 0 )		
		# est_cmrep_0 = base.ExponentialMap( tVec_0 )

		position_matrix_0 = base.GetEuclideanLocations()

		tVec_t = tangent.ScalarMultiply( time_pt )
		est_cmrep_t = base.ExponentialMap( tVec_t )

		for k in range( nManDim ):
			# Position Points + Center Change
			polyData_t.GetPoints().SetPoint( k, position_matrix_0[ k, : ] )
			polyData_t.GetPointData().GetArray( "Radius" ).SetValue( k, est_cmrep_t.pt[ 3 ].pt[ k ] )


		polyData_t.Modified() 		

		writer_t = vtk.vtkPolyDataWriter() 
		writer_t.SetFileName( output_path )
		writer_t.SetInputData( polyData_t )
		writer_t.Update()
		writer_t.Write() 

	# Bndr Normal1 Change
	normal2_change_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/CMRep_Abstract_Normal_CTRL_Group_NewCode_Linearized_Complex/BndrNormal1Change/'

	for n in range( nTimePt ):
		outFileName = 'CMRep_Abstract_Normal_Regression_CTRL_' + anatomy + '_BndrNormal1_Change_' + str( n ) + '.vtk' 

		output_path = normal2_change_folder_path + outFileName 

		time_pt = ( tN - t0 ) * n / ( nTimePt - 1 ) + t0  

		polyData_t = vtk.vtkPolyData()
		polyData_t.DeepCopy( meanPolyData ) 

		# tVec_0 = tangent.ScalarMultiply( 0 )		
		# est_cmrep_0 = base.ExponentialMap( tVec_0 )

		position_matrix_0 = base.GetEuclideanLocations()

		tVec_t = tangent.ScalarMultiply( time_pt )
		est_cmrep_t = base.ExponentialMap( tVec_t )

		for k in range( nManDim ):
			# Position Points + Center Change
			polyData_t.GetPoints().SetPoint( k, position_matrix_0[ k, : ] )
			polyData_t.GetPointData().GetArray( "BndrNormal1" ).SetTuple( k, est_cmrep_t.pt[ 4 ][ k ].pt )
			polyData_t.GetPointData().GetArray( "BndrNormal2" ).SetTuple( k, est_cmrep_t.pt[ 5 ][ k ].pt )

		polyData_t.Modified() 		

		writer_t = vtk.vtkPolyDataWriter() 
		writer_t.SetFileName( output_path )
		writer_t.SetInputData( polyData_t )
		writer_t.Update()
		writer_t.Write() 