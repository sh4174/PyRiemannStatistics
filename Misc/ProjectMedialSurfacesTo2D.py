import vtk

folderPath = "/media/shong/IntHard1/4DAnalysis/IPMI2018/GeodesicRegressionResults/CMRep_CTRL_Complex_NewCode_Linearized/"
filePath_LC = folderPath + "CMRep_Regression_CTRL_Comlex_left_caudate_0.vtk"
filePath_RC = folderPath + "CMRep_Regression_CTRL_Comlex_right_caudate_0.vtk"
filePath_LP = folderPath + "CMRep_Regression_CTRL_Comlex_left_putamen_0.vtk"
filePath_RP = folderPath + "CMRep_Regression_CTRL_Comlex_right_putamen_0.vtk"

lc_org_reader = vtk.vtkPolyDataReader() 
lc_org_reader.SetFileName( filePath_LC )
lc_org_reader.Update()
lc_org = lc_org_reader.GetOutput()

rc_org_reader = vtk.vtkPolyDataReader() 
rc_org_reader.SetFileName( filePath_RC )
rc_org_reader.Update()
rc_org = rc_org_reader.GetOutput()

lp_org_reader = vtk.vtkPolyDataReader() 
lp_org_reader.SetFileName( filePath_LP )
lp_org_reader.Update()
lp_org = lp_org_reader.GetOutput()

rp_org_reader = vtk.vtkPolyDataReader() 
rp_org_reader.SetFileName( filePath_RP )
rp_org_reader.Update()
rp_org = rp_org_reader.GetOutput()


# Rearrange Data to 2D
lc_projected = vtk.vtkPolyData()
lc_projected.DeepCopy( lc_org )

rc_projected = vtk.vtkPolyData()
rc_projected.DeepCopy( rc_org )

lp_projected = vtk.vtkPolyData()
lp_projected.DeepCopy( lp_org )

rp_projected = vtk.vtkPolyData()
rp_projected.DeepCopy( rp_org )

# Project LC
for i in range( lc_org.GetNumberOfPoints() ):
	pt_org_i = lc_org.GetPoint( i )
	lc_projected.GetPoints().SetPoint( i, [ pt_org_i[ 0 ], 0, pt_org_i[ 2 ] ] )

for i in range( rc_org.GetNumberOfPoints() ):
	pt_org_i = rc_org.GetPoint( i )
	rc_projected.GetPoints().SetPoint( i, [ pt_org_i[ 0 ], 0, pt_org_i[ 2 ] ] )

for i in range( lp_org.GetNumberOfPoints() ):
	pt_org_i = lp_org.GetPoint( i )
	lp_projected.GetPoints().SetPoint( i, [ pt_org_i[ 0 ], 0, pt_org_i[ 2 ] ] )

for i in range( rp_org.GetNumberOfPoints() ):
	pt_org_i = rp_org.GetPoint( i )
	rp_projected.GetPoints().SetPoint( i, [ pt_org_i[ 0 ], 0, pt_org_i[ 2 ] ] )

lc_projected.Modified()
rc_projected.Modified()
lp_projected.Modified()
rp_projected.Modified()

outFolderPath = "/media/shong/IntHard1/4DAnalysis/IPMI2018/GeodesicRegressionResults/CMRep_Shape_Complex_CTRL_Projected/"
filePath_LC = outFolderPath + "CMRep_Regression_CTRL_Comlex_left_caudate_PR.vtk"
filePath_RC = outFolderPath + "CMRep_Regression_CTRL_Comlex_right_caudate_PR.vtk"
filePath_LP = outFolderPath + "CMRep_Regression_CTRL_Comlex_left_putamen_PR.vtk"
filePath_RP = outFolderPath + "CMRep_Regression_CTRL_Comlex_right_putamen_PR.vtk"

writer_lc = vtk.vtkPolyDataWriter()
writer_lc.SetFileName( filePath_LC )
writer_lc.SetInputData( lc_projected )
writer_lc.Update()
writer_lc.Write()

writer_rc = vtk.vtkPolyDataWriter()
writer_rc.SetFileName( filePath_RC )
writer_rc.SetInputData( rc_projected )
writer_rc.Update()
writer_rc.Write()

writer_lp = vtk.vtkPolyDataWriter()
writer_lp.SetFileName( filePath_LP )
writer_lp.SetInputData( lp_projected )
writer_lp.Update()
writer_lp.Write()

writer_rp = vtk.vtkPolyDataWriter()
writer_rp.SetFileName( filePath_RP )
writer_rp.SetInputData( rp_projected )
writer_rp.Update()
writer_rp.Write()
