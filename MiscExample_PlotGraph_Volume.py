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

ageList_cont = [] 

VolList_high_LC = []
VolList_high_RC = []
VolList_high_LP = []
VolList_high_RP = [] 

ageList_high = [] 

VolList_cont_LC = []
VolList_cont_RC = []
VolList_cont_LP = []
VolList_cont_RP = [] 

regression_subject_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2018/GeodesicRegressionResults/CMRep_Subject_Complex_Linearized/'

# vtkPolyData for Intrinsic Mean
meanPolyDataList = []

for d in range( len( anatomy_list ) ):
	meanPolyData_d = vtk.vtkPolyData()
	meanPolyDataList.append( meanPolyData_d )	

# For all subjects
cnt = 0

cnt_volInc = 0
cnt_volDec = 0

volInc_subj_list = []

for i in range( len( dataInfoList )  ):
	subj_dataFolder = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID
	if not os.path.isdir( subj_dataFolder ):
		print( 'PHD-AS1-' + dataInfoList[i].ID + "does not exist" )
		continue

	# Skip if there is only one shape in the list 
	if len( dataInfoList[i].AgeList ) < 2:
		print( dataInfoList[i].ID + "has less than 2 data" )
		continue

	regression_output_folder_subject = regression_subject_folder_path + "/" + dataInfoList[i].ID + "_" + dataInfoList[i].CAPGroupList[ 0 ] + "/"
	cmrep_greg_base_path = regression_output_folder_subject + "CMRep_LinearizedGeodesicRegression_Complex_Subject_base.rpt"

	if not os.path.isfile( cmrep_greg_base_path ):
		print( cmrep_greg_base_path )				
		print( "Data Missing" )
		continue

	subj_cnt = 0
	riskGroupList_subj = []
	ageList_subj = [] 
	SubjectList_subj = []
	CAPList_subj = [] 

	VolList_LC_subj = []
	VolList_RC_subj = []
	VolList_LP_subj = []
	VolList_RP_subj = [] 	

	for j in range( len( dataInfoList[i].LabelList ) ):
		# if j > 0:
		# 	break

		# if not dataInfoList[i].CAPGroupList[ j ] == 'cont':
		# 	continue
			
		subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

		IsAllAnatomy = True

		vol_LC = 0
		vol_RC = 0
		vol_LP = 0
		vol_RP = 0

		if dataInfoList[i].ID == "51095" or dataInfoList[i].ID == "52749" or dataInfoList[i].ID == "51451" :
			continue


		for a in range( len( anatomy_list ) ):
			anatomy = anatomy_list[ a ] 

			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3_target.vtk"

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				IsAllAnatomy = False
				break
			print( anatomy_cmrep_surface_path )				

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( anatomy_cmrep_surface_path )
			reader.Update()
			polyData = reader.GetOutput()

			massProp = vtk.vtkMassProperties()
			massProp.SetInputData( polyData )
			massProp.Update()

			vol = massProp.GetVolume() 

			if anatomy == "left_caudate":
				vol_LC = vol
			elif anatomy == "right_caudate":
				vol_RC = vol
			elif anatomy == "left_putamen":
				vol_LP = vol
			elif anatomy == "right_putamen":
				vol_RP = vol

		if not IsAllAnatomy:
			continue

		riskGroupList_subj.append( dataInfoList[i].CAPGroupList[ j ] )
		ageList_subj.append( dataInfoList[i].AgeList[ j ] )
		SubjectList_subj.append( dataInfoList[i].ID )
		CAPList_subj.append( dataInfoList[i].CAPList[j] )
		VolList_LC_subj.append( vol_LC )
		VolList_RC_subj.append( vol_RC )
		VolList_LP_subj.append( vol_LP )
		VolList_RP_subj.append( vol_RP )

		subj_cnt += 1 

	if subj_cnt < 2:
		continue
	else:
		if riskGroupList_subj[ 0 ] == "high":
			isVolIncrease = False
			for sb in range( subj_cnt ):
				riskGroupList.append( riskGroupList_subj[ sb ] )
				ageList_high.append( ageList_subj[ sb ] )
				SubjectList.append( SubjectList_subj[ sb ] )
				CAPList.append( CAPList_subj[ sb ] )

				if sb >= 1:
					volDiff = VolList_RP_subj[ sb ] - VolList_RP_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

					volDiff = VolList_LC_subj[ sb ] - VolList_LC_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

					volDiff = VolList_RC_subj[ sb ] - VolList_RC_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

					volDiff = VolList_LP_subj[ sb ] - VolList_LP_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

				VolList_high_LC.append( VolList_LC_subj[ sb ] )
				VolList_high_RC.append( VolList_RC_subj[ sb ] )
				VolList_high_LP.append( VolList_LP_subj[ sb ] )
				VolList_high_RP.append( VolList_RP_subj[ sb ] )

			if isVolIncrease:
				print( dataInfoList[i].ID )
				volInc_subj_list.append( dataInfoList[ i ].ID )				
				cnt_volInc += 1 
			else:
				cnt_volDec += 1 

			cnt += subj_cnt

		if riskGroupList_subj[ 0 ] == "cont":
			isVolIncrease = False
			for sb in range( subj_cnt ):
				riskGroupList.append( riskGroupList_subj[ sb ] )
				ageList_cont.append( ageList_subj[ sb ] )
				SubjectList.append( SubjectList_subj[ sb ] )
				CAPList.append( CAPList_subj[ sb ] )

				if sb >= 1:
					volDiff = VolList_RP_subj[ sb ] - VolList_RP_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

					volDiff = VolList_LC_subj[ sb ] - VolList_LC_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

					volDiff = VolList_RC_subj[ sb ] - VolList_RC_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

					volDiff = VolList_LP_subj[ sb ] - VolList_LP_subj[ sb - 1 ] 
					if volDiff > 0:
						isVolIncrease = True

				VolList_cont_LC.append( VolList_LC_subj[ sb ] )
				VolList_cont_RC.append( VolList_RC_subj[ sb ] )
				VolList_cont_LP.append( VolList_LP_subj[ sb ] )
				VolList_cont_RP.append( VolList_RP_subj[ sb ] )

			# if isVolIncrease:
			# 	# print( dataInfoList[i].ID )
			# 	# volInc_subj_list.append( dataInfoList[ i ].ID )				
			# 	# cnt_volInc += 1 
			# else:
			# 	rr = 0
			# 	# cnt_volDec += 1 

			cnt += subj_cnt



print( "=========================================" )
print( volInc_subj_list )

print( "# Volume Increase" )
print( cnt_volInc )
print( "# Volume Decrease" )
print( cnt_volDec )


########################################
#####        Visualization        ######   
########################################
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)

plt.figure( figsize=(6,6 ) )
colors = [ [ 0.2, 0.8, 0.2 ], [ 1.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 1.0, 0, 1.0 ], [ 0, 0, 1.0 ] ] 
est_colors =[ [ 0, 1, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  
est_lReg_colors =[ [ 1, 0, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  

plt.scatter( ageList_cont, VolList_cont_RP, c=colors[ 0 ], alpha=0.5, label="Control Data" )
plt.scatter( ageList_high, VolList_high_RP, c=colors[ 4 ], alpha=0.5, label="High Data" )

plt.xlabel('Age') 
plt.ylabel('Volume' )
plt.legend()
plt.tight_layout()
plt.show()

