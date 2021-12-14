import os
import sys
import csv
import vtk

import numpy as np


# Data Folder
dataFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep/subjects/"

# Anatomy list
anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]

# Intrinsic Mean 
mu_folderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_CorrespondenceTest/CorrespondenceFromMu_AllAnatomy/"

mu_list = []

for a in range( len( anatomy_list ) ):
	muPath = mu_folderPath + 'Intrinsic_Mean_Comlex_' + anatomy_list[ a ] + '.vtk' 	
	reader_m = vtk.vtkPolyDataReader()
	reader_m.SetFileName( muPath )
	reader_m.Update()
	mu = reader_m.GetOutput()

	idxArr = vtk.vtkIntArray()
	idxArr.SetName( "Index" )

	nPt = mu.GetNumberOfPoints() 

	for i in range( nPt ):
		idxArr.InsertNextValue( i )

	idxArr.Modified()
	mu.GetPointData().AddArray( idxArr )
	mu.Modified()

	muWriter = vtk.vtkPolyDataWriter()
	muWriter.SetFileName( muPath )
	muWriter.SetInputData( mu )
	muWriter.Update()
	muWriter.Write()

	mu_list.append( mu ) 

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

for a in range( len( anatomy_list ) ):
	# M-Rep Lists
	CMRepSurfaceList = []
	LandmarkDistanceList = []
	SubjectList = []
	LabelList = []

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
			
			subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy_list[ a ] + "/mesh/def3.med.vtk"

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				continue

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( anatomy_cmrep_surface_path )
			reader.Update()
			polyData = reader.GetOutput()

			mu = mu_list[ a ]
			dist_i = 0 

			for p in range( nPt ):
				pt_mu = mu.GetPoint( p )

				pt_i = polyData.GetPoint( p ) 

				dist_i_p = np.sqrt( ( pt_mu[ 0 ] - pt_i[ 0 ] ) ** 2 + ( pt_mu[ 1 ] - pt_i[ 1 ] ) ** 2 + ( pt_mu[ 2 ] - pt_i[ 2 ] ) ** 2 ) 

				dist_i += dist_i_p

			CMRepSurfaceList.append( polyData )
			SubjectList.append( dataInfoList[i].ID )
			LabelList.append( dataInfoList[i].LabelList[j] )
			LandmarkDistanceList.append( dist_i ) 
			cnt +=1 

		CMRepSurfaceListNP = np.array( CMRepSurfaceList )
		LandmarkDistanceListNP = np.array( LandmarkDistanceList )
		SubjectListNP = np.array( SubjectList )
		LabelListNP = np.array( LabelList )

		distArrInd = LandmarkDistanceListNP.argsort()
		sortedDistList = LandmarkDistanceListNP[ distArrInd[ ::-1 ] ] 
		sortedSubjectList = SubjectListNP[ distArrInd[ ::-1 ] ] 
		sortedLabelList =  LabelListNP[ distArrInd[ ::-1 ] ] 
		sortedCMRepSurfaceList = CMRepSurfaceListNP[ distArrInd[ ::-1 ] ] 

	cntFlip = 0 

	for j in range( len( sortedCMRepSurfaceList ) ):
		dataTest = sortedCMRepSurfaceList[ j ]

		landmarktransformCheck = vtk.vtkLandmarkTransform()
		landmarktransformCheck.SetSourceLandmarks( dataTest.GetPoints() )
		landmarktransformCheck.SetTargetLandmarks( mu.GetPoints() )
		landmarktransformCheck.SetModeToRigidBody()
		landmarktransformCheck.Update()

		transformCheck = vtk.vtkTransform()
		transformCheck.SetMatrix( landmarktransformCheck.GetMatrix() )
		transformCheck.Update()

		if np.abs( transformCheck.GetOrientation()[ 0 ] ) >= 90:
			print( sortedSubjectList[ j ] )
			cntFlip += 1 
		if np.abs( transformCheck.GetOrientation()[ 1 ] ) >= 90:
			print( sortedSubjectList[ j ] )
			cntFlip += 1 
		if np.abs( transformCheck.GetOrientation()[ 2 ] ) >= 90:
			print( sortedSubjectList[ j ] )
			cntFlip += 1 

	print( cntFlip )
	# print( sortedSubjectList[ :10 ] )
	# print( sortedDistList[ :10 ] )
	# print( len( sortedSubjectList ) )

	# Selected 10 most different CM-Rep surfaces from the intrinsic mean
	outFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_CorrespondenceTest/CorrespondenceFromMu_AllAnatomy/" + anatomy_list[a] + "/"

	for i in range( 10 ):
		outFileName = "data" + str( i ) + "_" + sortedSubjectList[ i ] + "_" + sortedLabelList[ i ] + "_tr.vtk"
		outFilePath = outFolderPath + outFileName

		data = sortedCMRepSurfaceList[ i ] 

		# Translation for Visualization
		translation = vtk.vtkTransform()
		translation.Translate( 25, 0, 0 )

		transformFilter = vtk.vtkTransformPolyDataFilter()
		transformFilter.SetInputData( data )
		transformFilter.SetTransform( translation )
		transformFilter.Update()
		data_trans = transformFilter.GetOutput() 

		idxArr_data = vtk.vtkIntArray() 
		idxArr_data.SetName( "Index" ) 
		for p in range( nPt ):
			idxArr_data.InsertNextValue( p )
		idxArr_data.Modified()

		data_trans.GetPointData().AddArray( idxArr_data )
		data_trans.Modified()

		writer_i = vtk.vtkPolyDataWriter()
		writer_i.SetFileName( outFilePath )
		writer_i.SetInputData( data_trans )
		writer_i.Update()
		writer_i.Write()

		# Draw lines
		outLineFileName = "line_mu_to_data" + str( i ) + "_" + sortedSubjectList[ i ] + "_" + sortedLabelList[ i ] + "_tr.vtk"
		outLineFilePath = outFolderPath + outLineFileName

		linePolyData = vtk.vtkPolyData()
		points = vtk.vtkPoints()
		lines = vtk.vtkCellArray()

		idxPoint = vtk.vtkIntArray()
		idxPoint.SetName( "Point Index" )

		idxLine = vtk.vtkIntArray()
		idxLine.SetName( "Line Index" )

		cnt = 0

		for i in range( 0, nPt, 10 ):
			points.InsertNextPoint( mu.GetPoint( i ) )
			points.InsertNextPoint( data_trans.GetPoint( i ) )
			idxPoint.InsertNextValue( cnt )
			idxPoint.InsertNextValue( cnt )

			line = vtk.vtkLine()
			line.GetPointIds().SetId( 0, 2 * cnt )
			line.GetPointIds().SetId( 1, 2 * cnt + 1 )

			lines.InsertNextCell( line )
			idxLine.InsertNextValue( cnt )

			cnt += 1

		idxPoint.Modified()
		idxLine.Modified()

		linePolyData.SetPoints( points )
		linePolyData.SetLines( lines )

		linePolyData.GetPointData().AddArray( idxPoint ) 
		linePolyData.GetCellData().AddArray( idxLine )

		linePolyData.Modified() 

		writer_line_i = vtk.vtkPolyDataWriter()
		writer_line_i.SetFileName( outLineFilePath )
		writer_line_i.SetInputData( linePolyData )
		writer_line_i.Update()
		writer_line_i.Write()
