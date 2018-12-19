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
		self.GenderList = []	


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
			Gender_str = row[ 'Gender' ]

			if CAPG_str == '':
				CAPG_str = 'cont'
			if CAP_str == '':				
				CAP_str = '-1'

			dataInfo_k.LabelList.append( label_str )
			dataInfo_k.AgeList.append( float( age_str ) )
			dataInfo_k.CAPGroupList.append( CAPG_str )
			dataInfo_k.CAPList.append( float( CAP_str ) )			
			dataInfo_k.GenderList.append( Gender_str )			


			bIsInList = 1
			break

	if bIsInList == 0:
		label_str = row[ 'Label' ]
		age_str = row[ 'Scan_Age' ]
		CAPG_str = row[ 'CAP Group' ]
		CAP_str = row[ 'CAP' ] 
		Gender_str = row[ 'Gender' ]

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
		dataInfo_new.GenderList.append( Gender_str )

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
VolList_LC = []
VolList_RC = []
VolList_LP = []
VolList_RP = [] 

regression_subject_folder_path = '/media/shong/IntHard1/4DAnalysis/IPMI2018/GeodesicRegressionResults/CMRep_Subject_Complex_Linearized/'

# vtkPolyData for Intrinsic Mean
meanPolyDataList = []

for d in range( len( anatomy_list ) ):
	meanPolyData_d = vtk.vtkPolyData()
	meanPolyDataList.append( meanPolyData_d )	

# For all subjects
excluded_list = ['52956', '50963', '51968', '50607', '50591', '52601', '52119', '52498', '50463', '51706', '51034', '52639', '50850', '51543', '52033', '50384', '52104', '50452', '50077', '50867', '52894', '52454', '50073', '52355', '52494', '51618', '50693', '51433', '50114', '52754', '51389', '50217', '50181', '51781', '52340', '51243', '50186', '51476', '50640', '51788', '50609']
# excluded_list = ['52956', '51968', '50607', '52119', '52639', '52894', '51433', '52754', '50217', '50640', '51788']


cnt = 0
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

	if dataInfoList[i].ID in excluded_list:
		continue

	# if dataInfoList[i].GenderList[ 0 ] == 'f':
	# 	continue


	subj_cnt = 0
	CMRepDataList_subj = []
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

		nAtoms = 0
		cmrep_ij = manifolds.cmrep( 0 )

		IsAllAnatomy = True

		vol_LC = 0
		vol_RC = 0
		vol_LP = 0
		vol_RP = 0


		if dataInfoList[i].ID == "51095" or dataInfoList[i].ID == "52749" or dataInfoList[i].ID == "51451" :
			continue

		for a in range( len( anatomy_list ) ):
			anatomy = anatomy_list[ a ] 

			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				IsAllAnatomy = False
				break

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
			else:
				vol_RP = vol

			if cnt == 0:
				meanPolyDataList[ a ].DeepCopy( polyData )	
				# print( polyData )

			nAtoms_a = polyData.GetNumberOfPoints() 
			nAtoms += nAtoms_a

			for k in range( nAtoms_a ):
				pos = polyData.GetPoint( k )
				rad = polyData.GetPointData().GetArray( "Radius Function" ).GetValue( k )

				cmrep_ij_pos_a_k = manifolds.euclidean( 3 )
				cmrep_ij_rad_a_k = manifolds.pos_real( 1 )

				cmrep_ij_pos_a_k.SetPoint( pos )
				cmrep_ij_rad_a_k.SetPoint( rad )

				cmrep_ij.AppendAtom( [ cmrep_ij_pos_a_k, cmrep_ij_rad_a_k ] )

			# cmrep_ij.UpdateMeanRadius()

		if not IsAllAnatomy:
			continue


		CMRepDataList_subj.append( cmrep_ij )
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
		if riskGroupList_subj[ 0 ] == "high" or riskGroupList_subj[ 0 ] == "cont":
			for sb in range( subj_cnt ):
				CMRepDataList.append( CMRepDataList_subj[ sb ] )
				riskGroupList.append( riskGroupList_subj[ sb ] )
				ageList.append( ageList_subj[ sb ] )
				SubjectList.append( SubjectList_subj[ sb ] )
				CAPList.append( CAPList_subj[ sb ] )
				VolList_LC.append( VolList_LC_subj[ sb ] )
				VolList_RC.append( VolList_RC_subj[ sb ] )
				VolList_LP.append( VolList_LP_subj[ sb ] )
				VolList_RP.append( VolList_RP_subj[ sb ] )
			cnt += subj_cnt

nDataList = []
nDimStartlist = [ 0 ]

for d in range( len( anatomy_list ) ):
	meanPolyData_d = meanPolyDataList[ d ]
	nDataList.append( meanPolyData_d.GetNumberOfPoints() )

	if d > 0:
		nDimStartlist.append( nDataList[ d - 1 ] + nDimStartlist[ d - 1 ] )

# Manifold Dimension
nManDim = CMRepDataList[0].nDim
nData = len( CMRepDataList )

base = manifolds.cmrep( nManDim )
base.Read( "CMRep_LinearizedGeodesicRegression_Complex_CTRL_base.rpt" )

tangent = manifolds.cmrep_tVec( nManDim ) 
tangent.Read(  "CMRep_LinearizedGeodesicRegression_Complex_CTRL_tangent.tVec"  )

t0 = 20 
tN = 80
nTimePt = 30
a = 3 

point_idx_list = [ 343, 342, 851, 820, 760, 341, 202, 809, 174, 821, 372, 371, 808, 756, 218, 370, 197, 757, 758, 765, 317, 301, 198, 356, 178, 302, 303, 811, 354, 355, 805, 220, 849, 766, 529, 527, 528, 804, 167 ]
# point_idx_list = np.arange( 0, 853 )
pt_list_cont = []
t_list_cont = []

pt_list_high = []
t_list_high = []

for n in range( nData ):
	if riskGroupList[ n ] == "cont":
		rad_pt_list = []

		for p_i in range( len( point_idx_list ) ):
			pt_idx = point_idx_list[ p_i ]
			k_cmrep = nDimStartlist[ a ] + pt_idx

			rad_pt_list.append( CMRepDataList[ n ].pt[ k_cmrep ][ 1 ] )

		rad_pt = rsm.FrechetMean( rad_pt_list )

		pt_list_cont.append( rad_pt.pt )
		t_list_cont.append( ageList[ n ] )

	if riskGroupList[ n ] == "high":
		rad_pt_list = []
		for p_i in range( len( point_idx_list ) ):
			pt_idx = point_idx_list[ p_i ]
			k_cmrep = nDimStartlist[ a ] + pt_idx

			rad_pt_list.append( CMRepDataList[ n ].pt[ k_cmrep ][ 1 ] )

		rad_pt = rsm.FrechetMean( rad_pt_list )

		pt_list_high.append( rad_pt.pt )
		t_list_high.append( ageList[ n ] )


est_pt_list = [] 
est_t_list = []

base_rad_pt_list = []
tangent_rad_tVec_list = []

for p_i in range( len( point_idx_list ) ):
	pt_idx = point_idx_list[ p_i ]
	k_cmrep = nDimStartlist[ a ] + pt_idx

	base_rad_pt_list.append( base.pt[ k_cmrep ][ 1 ] )
	tangent_rad_tVec_list.append( tangent.tVector[ k_cmrep ][ 1 ].tVector[ 0 ] )

base_rad_pt = rsm.FrechetMean( base_rad_pt_list ) 
tangent_pt = np.average( tangent_rad_tVec_list )
tangent_rad_pt = manifolds.pos_real_tVec( 1 )
tangent_rad_pt.tVector[ 0 ] = tangent_pt

for t in range( nTimePt ):
	time_pt = ( tN - t0 ) * t / ( nTimePt - 1 ) + t0

	v_t = manifolds.pos_real_tVec( 1 )
	v_t.SetTangentVector( np.multiply( tangent_rad_pt.tVector, time_pt ).tolist() )
	mean = base_rad_pt.ExponentialMap( v_t )

	est_pt_list.append( mean.pt ) 
	est_t_list.append( time_pt ) 


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

plt.scatter( t_list_cont, pt_list_cont, c=colors[ 0 ], alpha=0.5, label="Control Data" )
plt.scatter( t_list_high, pt_list_high, c=colors[ 4 ], alpha=0.5, label="High Data" )
plt.plot( est_t_list, est_pt_list, c=est_colors[ 0 ], linestyle="-", linewidth=4, label="Aging Atlas" )

# Subject specific plots

nManDim = CMRepDataList[0].nDim
nData = len( CMRepDataList )

cont_cnt = 0
low_cnt = 0
med_cnt = 0
high_cnt = 0

print( "===========================================" )
print( "Positive Tangent Vector" ) 
print( "===========================================" )

nHighPositiveTVec = 0
nHighNegativeTVec = 0

for i in range( len( dataInfoList )  ):
	if not np.mod( i, 5 ) == 0:
		continue

	subj_dataFolder = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID
	if not os.path.isdir( subj_dataFolder ):
		# print( 'PHD-AS1-' + dataInfoList[i].ID + "does not exist" )
		continue

	# Skip if there is only one shape in the list 
	if len( dataInfoList[i].AgeList ) < 2:
		# print( dataInfoList[i].ID + "has less than 2 data" )
		continue


	if dataInfoList[i].ID == "51095" or dataInfoList[i].ID == "52749" or dataInfoList[i].ID == "51451" :
		continue

	if dataInfoList[i].ID in excluded_list:
		continue

	# if dataInfoList[i].GenderList[ 0 ] == 'f':
	# 	continue

	regression_output_folder_subject = regression_subject_folder_path + "/" + dataInfoList[i].ID + "_" + dataInfoList[i].CAPGroupList[ 0 ] + "/"
	cmrep_greg_base_path = regression_output_folder_subject + "CMRep_LinearizedGeodesicRegression_Complex_Subject_base.rpt"

	if not os.path.isfile( cmrep_greg_base_path ):
		# print( cmrep_greg_base_path )				
		# print( "Data Missing" )
		continue

	base_i = manifolds.cmrep( nManDim )
	base_i.Read( regression_output_folder_subject + "CMRep_LinearizedGeodesicRegression_Complex_Subject_base.rpt" )
	
	tangent_i = manifolds.cmrep_tVec( nManDim ) 
	tangent_i.Read( regression_output_folder_subject + "CMRep_LinearizedGeodesicRegression_Complex_Subject_tangent.tVec" )

	t0_i = dataInfoList[i].AgeList[ 0 ]  
	tN_i = dataInfoList[i].AgeList[ -1 ]

	nTimePt = 30
	est_time_list_i = []
	est_rad_pt_list_i = []

	IsTangentPositive = False

	base_rad_pt_list = []
	tangent_rad_tVec_list = []

	for p_i in range( len( point_idx_list ) ):
		pt_idx = point_idx_list[ p_i ]
		k_cmrep = nDimStartlist[ a ] + pt_idx

		base_rad_pt_list.append( base_i.pt[ k_cmrep ][ 1 ] )
		tangent_rad_tVec_list.append( tangent_i.tVector[ k_cmrep ][ 1 ].tVector[ 0 ] )

	base_rad_pt = rsm.FrechetMean( base_rad_pt_list ) 
	tangent_pt = np.average( tangent_rad_tVec_list )
	tangent_rad_pt = manifolds.pos_real_tVec( 1 )
	tangent_rad_pt.tVector[ 0 ] = tangent_pt


	# base_rad_pt = base_i.pt[ k_cmrep ][ 1 ]	
	# tangent_rad_pt = tangent_i.tVector[ k_cmrep ][ 1 ]

	if tangent_rad_pt.tVector[ 0 ] > 0:
		IsTangentPositive = True

	for t in range( nTimePt ):
		time_pt = ( tN_i - t0_i ) * t / ( nTimePt - 1 ) + t0_i

		v_t = manifolds.pos_real_tVec( 1 )
		v_t.SetTangentVector( np.multiply( tangent_rad_pt.tVector, time_pt ).tolist() )
		mean = base_rad_pt.ExponentialMap( v_t )

		est_rad_pt_list_i.append( mean.pt[ 0 ] ) 
		est_time_list_i.append( time_pt ) 

	if dataInfoList[i].CAPGroupList[ 0 ] == 'cont':
		if cont_cnt == 0:			
			plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 1 ], linestyle="-", linewidth=2, label="cont" )
			cont_cnt = 1 
		else:
			plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 1 ], linestyle="-", linewidth=2 )
	# elif dataInfoList[i].CAPGroupList[ 0 ] == 'low':
	# 	if low_cnt == 0:			
	# 		plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 2 ], linestyle="-", linewidth=2, label="low" )
	# 		low_cnt = 1 
	# 	else:
	# 		plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 2 ], linestyle="-", linewidth=2 )
	# elif dataInfoList[i].CAPGroupList[ 0 ] == 'med':
	# 	if med_cnt == 0:			
	# 		plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 3 ], linestyle="-", linewidth=2, label="med" )
	# 		med_cnt = 1
	# 	else:
	# 		plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 3 ], linestyle="-", linewidth=2 )
	elif dataInfoList[i].CAPGroupList[ 0 ] == 'high':
		if high_cnt == 0:
			if IsTangentPositive:
				plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 4 ], linestyle="-", linewidth=2, label="high" )
			else:
				plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 3 ], linestyle="-", linewidth=2, label="high" )

			high_cnt = 1
		else:
			if IsTangentPositive:
				plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 4 ], linestyle="-", linewidth=2 )
			else:
				plt.plot( est_time_list_i, est_rad_pt_list_i, c=est_colors[ 3 ], linestyle="-", linewidth=2 )
	else:
		rrr = 0
		# print( "... ?" )

	if IsTangentPositive and dataInfoList[ i ].CAPGroupList[ 0 ] == 'high':
		print( dataInfoList[i].ID )
		nHighPositiveTVec += 1 
	elif dataInfoList[ i ].CAPGroupList[ 0 ] == 'high':
		nHighNegativeTVec += 1


print( "Number of Pos" )
print( nHighPositiveTVec )
print( "# of Neg")
print( nHighNegativeTVec )
print( "===========================================" )


plt.xlabel('Age') 
plt.ylabel('Rad' )
plt.legend()
plt.tight_layout()
plt.show()

