import os
import sys
import csv
import vtk

import numpy as np
import pandas as pd

import math 

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

addLibraryPath = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/media/shong/IntHard1/Installer/MeshValmet-3.0-Linux64/"

meshValmetPath = "/media/shong/IntHard1/Installer/MeshValmet-3.0-Linux64/MeshValmet -type 2 -absolute "

MSDArr = [] 
MADArr = [] 
DICEArr = [] 

# For all subjects
cnt = 0
for i in range( len( dataInfoList )  ):
# for i in range( 10 ):
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

		print( "ID : " + dataInfoList[i].ID + ", Label : " + dataInfoList[i].LabelList[j ] )

		MSDArr_a = [] 
		MADArr_a = []
		DICEArr_a = []

		for anatomy in anatomy_list:
			bnd_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3_bnd.byu"

			if dataInfoList[i].ID == "51095" or dataInfoList[i].ID == "51451" or dataInfoList[i].ID == "52050":
				bnd_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def2_bnd.byu"

			target_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/def3_target.byu"

			if not os.path.isfile( bnd_path ):
				print( bnd_path )				
				print( "File doesn't exist" )
				continue

			output_path = subj_i_label_j_folderPath + "cmrep_" + anatomy + "/mesh/bnd_target_diff" 

			commandPath = meshValmetPath + "-in1 " + bnd_path + " -in2 " + target_path + " -o " + output_path
			print( commandPath )

			os.system( addLibraryPath + " && " + commandPath )

			statFilePath = output_path + ".stat"
			diffStat = pd.read_csv( statFilePath, sep='\t')
			
			if type( diffStat[ 'MAD' ][0] ) == list:
				break;
			if type( diffStat[ 'MSD' ][0] ) == list:
				break;
			if type( diffStat[ 'Dice' ][0] ) == list:
				break;

			if float( diffStat[ 'MSD' ][0] ) < 100 and float( diffStat[ 'MSD' ][0] ) > 0:
				MSDArr_a.append( float( diffStat[ 'MSD' ][0] ) )
			else:
				MSDArr_a.append( -1 )

			if float( diffStat[ 'MAD' ][0] ) < 100 and float( diffStat[ 'MAD' ][0] ) > 0:
				MADArr_a.append( float( diffStat[ 'MAD' ][0] ) )
			else:
				MADArr_a.append( -1 )

			if float( diffStat[ 'Dice' ][0] ) < 100 and float( diffStat[ 'Dice' ][0] ) > 0:
				DICEArr_a.append( float( diffStat[ 'Dice' ][0] ) )
			else:
				DICEArr_a.append( -1 )

		if not len( MSDArr_a ) == 4:
			continue

		MSDArr.append( MSDArr_a )
		MADArr.append( MADArr_a )
		DICEArr.append( DICEArr_a )

print( len( MSDArr ) ) 
print( len( MSDArr[ 0 ] ) ) 

MSDMat = np.array( MSDArr )
MADMat = np.array( MADArr )
DICEMat = np.array( DICEArr )

print( MSDMat )
print( MADMat )
print( DICEMat )

np.savetxt( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_Fitting/MSD_allstruct.csv", MSDMat, delimiter=",", fmt='%s' )
np.savetxt( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_Fitting/MAD_allstruct.csv", MADMat, delimiter=",", fmt='%s' )
np.savetxt( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_Fitting/DICE_allstruct.csv", DICEMat, delimiter=",", fmt='%s' )


# MSDMat = np.loadtxt( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_Fitting/MSD_allstruct.csv", delimiter="," )
# MADMat = np.loadtxt( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_Fitting/MAD_allstruct.csv", delimiter="," )
# DICEMat = np.loadtxt( "/media/shong/IntHard1/Projects/4DShapeAnalysis/Data/CMRep_Fitting/DICE_allstruct.csv", delimiter="," )

# print( "==============================")
# print( "Left Caudate" ) 

# MSD_LC = MSDMat[ :, 0 ]

# print( MSD_LC.shape ) 
# print( np.average( MSD_LC ) ) 
# print( np.std( MSD_LC ) ) 
# print( np.max( MSD_LC ) )
# print( np.min( MSD_LC ) ) 

# DICE_LC = DICEMat[ :, 0 ]
# print( DICE_LC.shape ) 
# print( np.average( DICE_LC ) ) 
# print( np.std( DICE_LC ) ) 
# print( np.max( DICE_LC ) ) 
# print( np.min( DICE_LC ) )
# print( "==============================")
# print( "==============================")
# print( "Right Caudate" ) 

# MSD_RC = MSDMat[ :, 1 ]

# print( MSD_RC.shape ) 
# print( np.average( MSD_RC ) ) 
# print( np.std( MSD_RC ) ) 
# print( np.max( MSD_RC ) )
# print( np.min( MSD_RC ) ) 

# DICE_RC = DICEMat[ :, 1 ]
# print( DICE_RC.shape ) 
# print( np.average( DICE_RC ) ) 
# print( np.std( DICE_RC ) ) 
# print( np.max( DICE_RC ) ) 
# print( np.min( DICE_RC ) )
# print( "==============================")
# print( "==============================")
# print( "Left Putamen" ) 

# MSD_LP = MSDMat[ :, 2 ]

# print( MSD_LP.shape ) 
# print( np.average( MSD_LP ) ) 
# print( np.std( MSD_LP ) ) 
# print( np.max( MSD_LP ) )
# print( np.min( MSD_LP ) ) 

# DICE_LP = DICEMat[ :, 2 ]
# print( DICE_LP.shape ) 
# print( np.average( DICE_LP ) ) 
# print( np.std( DICE_LP ) ) 
# print( np.max( DICE_LP ) ) 
# print( np.min( DICE_LP ) )

# print( "==============================")
# print( "==============================")
# print( "Right Putamen" ) 

# MSD_RP = MSDMat[ :, 3 ]

# print( MSD_RP.shape ) 
# print( np.average( MSD_RP ) ) 
# print( np.std( MSD_RP ) ) 
# print( np.max( MSD_RP ) )
# print( np.min( MSD_RP ) ) 

# DICE_RP = DICEMat[ :, 3 ]
# print( DICE_RP.shape ) 
# print( np.average( DICE_RP ) ) 
# print( np.std( DICE_RP ) ) 
# print( np.max( DICE_RP ) ) 
# print( np.min( DICE_RP ) )
# print( "==============================")
