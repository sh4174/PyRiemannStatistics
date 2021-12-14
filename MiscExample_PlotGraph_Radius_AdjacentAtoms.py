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

# vtkPolyData for Intrinsic Mean
meanPolyDataList = []

for d in range( len( anatomy_list ) ):
	meanPolyData_d = vtk.vtkPolyData()
	meanPolyDataList.append( meanPolyData_d )	

# For all subjects
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

	for j in range( len( dataInfoList[i].LabelList ) ):
		if j > 0:
			break

		if not dataInfoList[i].CAPGroupList[ j ] == 'cont':
			continue
			
		subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/decimated_aligned/"

		nAtoms = 0
		cmrep_ij = manifolds.cmrep( 0 )

		for a in range( len( anatomy_list ) ):
			anatomy = anatomy_list[ a ] 

			anatomy_cmrep_surface_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3.med.vtk"

			if not os.path.isfile( anatomy_cmrep_surface_path ):
				print( anatomy_cmrep_surface_path )				
				print( "File doesn't exist" )
				break

			reader = vtk.vtkPolyDataReader()
			reader.SetFileName( anatomy_cmrep_surface_path )
			reader.Update()
			polyData = reader.GetOutput()

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

		CMRepDataList.append( cmrep_ij )
		riskGroupList.append( dataInfoList[i].CAPGroupList[ j ] )
		ageList.append( dataInfoList[i].AgeList[ j ] )
		SubjectList.append( dataInfoList[i].ID )
		CAPList.append( dataInfoList[i].CAPList[j] )
		cnt +=1 

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
est_time_list_i = []
est_rad_pt_list_i = []

a = 3 
k_cmrep = nDimStartlist[ a ] + 757

pt_list_a = []
t_list_a = []

for n in range( nData ):
	pt_list_a.append( CMRepDataList[ n ].pt[ k_cmrep ][ 1 ].pt )
	t_list_a.append( ageList[ n ] )

est_pt_list = [] 
est_t_list = []

for t in range( nTimePt ):
	time_pt = ( tN - t0 ) * t / ( nTimePt - 1 ) + t0

	base_rad_pt = base.pt[ k_cmrep ][ 1 ]
	tangent_rad_pt = tangent.tVector[ k_cmrep ][ 1 ]


	v_t = manifolds.pos_real_tVec( 1 )
	v_t.SetTangentVector( np.multiply( tangent_rad_pt.tVector, time_pt ).tolist() )
	mean = base_rad_pt.ExponentialMap( v_t )

	est_pt_list.append( mean.pt ) 
	est_t_list.append( time_pt ) 

k_cmrep2 = nDimStartlist[ a ] + 756

pt_list_a2 = []
t_list_a2 = []

for n in range( nData ):
	pt_list_a2.append( CMRepDataList[ n ].pt[ k_cmrep2 ][ 1 ].pt )
	t_list_a2.append( ageList[ n ] )

est_pt_list2 = [] 
est_t_list2 = []

for t in range( nTimePt ):
	time_pt = ( tN - t0 ) * t / ( nTimePt - 1 ) + t0

	base_rad_pt = base.pt[ k_cmrep2 ][ 1 ]
	tangent_rad_pt = tangent.tVector[ k_cmrep2 ][ 1 ]


	v_t = manifolds.pos_real_tVec( 1 )
	v_t.SetTangentVector( np.multiply( tangent_rad_pt.tVector, time_pt ).tolist() )
	mean = base_rad_pt.ExponentialMap( v_t )

	est_pt_list2.append( mean.pt ) 
	est_t_list2.append( time_pt ) 



########################################
#####        Visualization        ######   
########################################
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)

plt.figure( figsize=(6,6 ) )
colors = [ [ 0, 0, 1 ], [ 1.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0, 0.5, 1.0 ], [ 0, 0, 1.0 ] ] 
est_colors =[ [ 0, 1, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  
est_lReg_colors =[ [ 1, 0, 0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 0.5, 0.0 ], [ 0.5, 0, 0.5 ], [ 0, 0, 0.5 ] ]  

plt.scatter( t_list_a2, pt_list_a2, c=colors[ 4 ], alpha=0.5, label="756th Atom Data" )
plt.scatter( t_list_a, pt_list_a, c=colors[ 3 ], alpha=0.5, label="757th Atom Data" )
plt.plot( est_t_list2, est_pt_list2, c=est_colors[ 1 ], linestyle="-", linewidth=4, label="756th Atom \nEGReg" )
plt.plot( est_t_list, est_pt_list, c=est_colors[ 0 ], linestyle="-", linewidth=4, label="757th Atom \nEGReg" )
plt.xlabel('Age(year)') 
plt.ylabel('Rad(mm)' )
plt.legend()
plt.tight_layout()
plt.show()

