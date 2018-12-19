import os
import sys
import csv
import vtk

import numpy as np

dataPath1 = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep_LC_Aligned/subjects/PHD-AS1-50007/42716/surfaces/decimated_aligned/cmrep_left_caudate/mesh/def3.med.vtk"
dataPath2 = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/Subjects_CMRep_LC_Aligned/subjects/PHD-AS1-52873/50329/surfaces/decimated_aligned/cmrep_left_caudate/mesh/def3.med.vtk"

reader1 = vtk.vtkPolyDataReader()
reader1.SetFileName( dataPath1 )
reader1.Update()

data1 = reader1.GetOutput() 

reader2 = vtk.vtkPolyDataReader()
reader2.SetFileName( dataPath2 ) 
reader2.Update()

data2 = reader2.GetOutput()

translation = vtk.vtkTransform()
translation.Translate( 25, 0, 0 )

transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetInputData( data2 )
transformFilter.SetTransform( translation )
transformFilter.Update()

data2_trans = transformFilter.GetOutput() 

nPt = data1.GetNumberOfPoints()

linePolyData = vtk.vtkPolyData()
points = vtk.vtkPoints()
lines = vtk.vtkCellArray()
cnt = 0

for i in range( 0, nPt, 10 ):
	points.InsertNextPoint( data1.GetPoint( i ) )
	points.InsertNextPoint( data2_trans.GetPoint( i ) )

	line = vtk.vtkLine()
	line.GetPointIds().SetId( 0, 2 * cnt )
	line.GetPointIds().SetId( 1, 2 * cnt + 1 )

	lines.InsertNextCell( line )

	cnt += 1

linePolyData.SetPoints( points )
linePolyData.SetLines( lines )
linePolyData.Modified() 


outFolderPath = "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_CorrespondenceTest/"
outData1Path = outFolderPath + "data1_50007.vtk"

writer1 = vtk.vtkPolyDataWriter()
writer1.SetFileName( outData1Path )
writer1.SetInputData( data1 )
writer1.Update()
writer1.Write()

outData2Path = outFolderPath + "data2_52873_tr.vtk"
writer2 = vtk.vtkPolyDataWriter()
writer2.SetFileName( outData2Path )
writer2.SetInputData( data2_trans )
writer2.Update()
writer2.Write()

outLinePath = outFolderPath + "lines_50007_52873.vtk"
writer_line = vtk.vtkPolyDataWriter()
writer_line.SetFileName( outLinePath )
writer_line.SetInputData( linePolyData )
writer_line.Update()
writer_line.Write()




