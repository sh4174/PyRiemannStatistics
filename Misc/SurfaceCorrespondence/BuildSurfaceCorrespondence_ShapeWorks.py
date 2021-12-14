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
# anatomy_list = [ 'left_caudate', 'left_putamen', 'right_caudate', 'right_putamen' ]
anatomy_list = [ 'left_caudate' ]

ShapeWorksFolderPath = "/media/shong/IntHard1/Installer/ShapeWorks_Win/ShapeWorks_Linux_Command/"

MSDArr = [] 
MADArr = [] 
DICEArr = [] 


# For all subjects
for anatomy in anatomy_list:
	# Left Caudate
	input_arr = [] 

	pre1_output_arr = []

	pre2_output_arr = [] 

	corr_output_arr = [] 

	# for i in range( len( dataInfoList )  ):
	for i in range( 10 ):
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

			input_path = subj_i_label_j_folderPath + anatomy + ".mha"

			if not os.path.isfile( input_path ):
				continue

			pre1_output_path = subj_i_label_j_folderPath + anatomy + "_DT.mha"

			pre2_output_path = subj_i_label_j_folderPath + anatomy + "_DT2.mha"

			corr_output_path = subj_i_label_j_folderPath + anatomy + ".wpts"


			input_arr.append( input_path )
			pre1_output_arr.append( pre1_output_path )
			pre2_output_arr.append( pre2_output_path )
			corr_output_arr.append( corr_output_path )


	# Preprocessing 1 XML file
	param_pre1_path = anatomy + '_preprocessing1.xml'
	param_pre1_file = open( param_pre1_path, 'w' )

	param_pre1_file.write( '<!-- ' + anatomy + '_preprocessing1.xml -->\n' )
	param_pre1_file.write( '<!-- Preprocessing Parameters 1 for ' + anatomy + ' -->\n' )
	param_pre1_file.write( '\n' )
	param_pre1_file.write( '<!-- Value of background pixels in the image -->\n' )
	param_pre1_file.write( '<background> 0.0 </background>\n' )
	param_pre1_file.write( '\n' )
	param_pre1_file.write( '<!-- Value of foreground pixels in the image -->\n' )
	param_pre1_file.write( '<foreground> 1.0 </foreground>\n' )
	param_pre1_file.write( '\n' )
	param_pre1_file.write( '<!-- Number of background pixels to pad the edges of the cropped volume -->\n' )
	param_pre1_file.write( '<pad> 10 </pad>\n' )
	param_pre1_file.write( ' \n' )
	param_pre1_file.write( '<!-- filename to store transforms generated during preprocessing -->\n' )
	param_pre1_file.write( '<transform_file> ' + anatomy + '.translations </transform_file>\n' )
	param_pre1_file.write( ' \n' )
	param_pre1_file.write( ' \n' )
	param_pre1_file.write( '<!-- Output progress information -->\n' )
	param_pre1_file.write( '<verbose> 1 </verbose>\n' )
	param_pre1_file.write( ' \n' )
	param_pre1_file.write( '<!-- Set of input files to process -->\n' )
	param_pre1_file.write( '<inputs>\n' )

	for k in  range( len( input_arr ) ):
		param_pre1_file.write( input_arr[ k ] + '\n' )

	param_pre1_file.write('</inputs>\n')
	param_pre1_file.write( '\n' )
	param_pre1_file.write( '<!-- Output filenames to use -->\n' )
	param_pre1_file.write( '<outputs>\n' )

	for k in  range( len( pre1_output_arr ) ):
		param_pre1_file.write( pre1_output_arr[ k ] + '\n' )

	param_pre1_file.write('</outputs>\n')
	param_pre1_file.close()

	groom_preprocessing1_command = ShapeWorksFolderPath + 'ShapeWorksGroom ' + param_pre1_path + ' isolate hole_fill center auto_crop'
	os.system( groom_preprocessing1_command )

	# PreProcessing 2 XML file
	param_pre2_path = anatomy + '_preprocessing2.xml'
	param_pre2_file = open( param_pre2_path, 'w' )

	param_pre2_file.write( '<!-- ' + anatomy + '_preprocessing2.xml -->\n' )
	param_pre2_file.write( '<!-- Preprocessing Parameters 2 for ' + anatomy + ' -->\n' )
	param_pre2_file.write( '\n' )
	param_pre2_file.write( '<!-- Value of background pixels in the image -->\n' )
	param_pre2_file.write( '<background> 0.0 </background>\n' )
	param_pre2_file.write( '\n' )
	param_pre2_file.write( '<!-- Value of foreground pixels in the image -->\n' )
	param_pre2_file.write( '<foreground> 1.0 </foreground>\n' )
	param_pre2_file.write( '\n' )
	param_pre2_file.write( '<!-- Number of anti-aliasing iterations -->\n' )
	param_pre2_file.write( '<antialias_iterations> 20 </antialias_iterations>\n' )
	param_pre2_file.write( ' \n' )
	param_pre2_file.write( '<!-- Size of Gaussian blurring kernel for smoothing -->\n' )
	param_pre2_file.write( '<blur_sigma> 2.0 </blur_sigma>\n' )
	param_pre2_file.write( ' \n' )
	param_pre2_file.write( ' \n' )
	param_pre2_file.write( '<!-- Pixel value associated with shape surfaces -->\n' )
	param_pre2_file.write( '<fastmarching_isovalue> 0.0 </fastmarching_isovalue>\n' )
	param_pre2_file.write( ' \n' )
	param_pre2_file.write( '<!-- Set of input files to process -->\n' )
	param_pre2_file.write( '<inputs>\n' )

	for k in  range( len( pre1_output_arr ) ):
		param_pre2_file.write( pre1_output_arr[ k ] + '\n' )

	param_pre2_file.write('</inputs>\n')
	param_pre2_file.write( '\n' )
	param_pre2_file.write( '<!-- Output filenames to use -->\n' )
	param_pre2_file.write( '<outputs>\n' )

	for k in  range( len( pre2_output_arr ) ):
		param_pre2_file.write( pre2_output_arr[ k ] + '\n' )

	param_pre2_file.write('</outputs>\n')
	param_pre2_file.close()

	groom_preprocessing2_command = ShapeWorksFolderPath + 'ShapeWorksGroom ' + param_pre2_path + ' antialias fastmarching blur'
	os.system( groom_preprocessing2_command )

	# Correspondence
	param_corr_path = anatomy + '_correspondence.xml'
	param_corr_file = open( param_corr_path, 'w' )

	param_corr_file.write( '<!-- ' + anatomy + '_correspondence.xml -->\n' )
	param_corr_file.write( '<!-- Correspondence Parameters 2 for ' + anatomy + ' -->\n' )
	param_corr_file.write( '\n' )

	param_corr_file.write( '<!-- Set of input files to process -->\n' )
	param_corr_file.write( '<inputs>\n' )

	for k in  range( len( pre2_output_arr ) ):
		param_corr_file.write( pre2_output_arr[ k ] + '\n' )

	param_corr_file.write('</inputs>\n')
	param_corr_file.write( '\n' )

	param_corr_file.write( '<number_of_particles> 256 </number_of_particles>\n' )
	param_corr_file.write( '\n' )
	param_corr_file.write( '<!-- Iterations between splitting during initialization phase. -->\n' )
	param_corr_file.write( '<iterations_per_split> 200 </iterations_per_split>\n' )
	param_corr_file.write( '\n' )
	param_corr_file.write( '<!-- Starting regularization for the entropy-based correspondence optimization. -->\n' )
	param_corr_file.write( '<starting_regularization> 10.0 </starting_regularization>\n' )
	param_corr_file.write( ' \n' )
	param_corr_file.write( '<!-- Final regularization for the entropy-based correspondence. -->\n' )
	param_corr_file.write( '<ending_regularization> 0.1 </ending_regularization>\n' )
	param_corr_file.write( ' \n' )
	param_corr_file.write( '<!-- Number of iterations for the entropy-based correspondence. -->\n' )
	param_corr_file.write( '<optimization_iterations> 200 </optimization_iterations>\n' )
	param_corr_file.write( ' \n' )

	param_corr_file.write( '<!-- Number of iterations between checkpoints (iterations at which results are saved) -->\n' )
	param_corr_file.write( '<checkpointing_interval> 20 </checkpointing_interval>\n' )
	param_corr_file.write( ' \n' )

	param_corr_file.write( '<!-- Output filenames to use -->\n' )
	param_corr_file.write( '<outputs>\n' )

	for k in  range( len( corr_output_arr ) ):
		param_corr_file.write( corr_output_arr[ k ] + '\n' )

	param_corr_file.write('</outputs>\n')
	param_corr_file.close()

	corr_command = ShapeWorksFolderPath + 'ShapeWorksRun ' + param_corr_path
	os.system( corr_command )
