import numpy as np
import os
import vtk
import csv 


import torch

print( torch.__version__ )

# regExePath = '/home/shong/anaconda3/envs/deformetrica/bin/deformetrica'

regExePath = '/home/shong/anaconda3/envs/deformetrica/bin/deformetrica estimate -v DEBUG'

regExePath = 'deformetrica estimate -v DEBUG'


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
# anatomy_list = [ 'left_caudate' ]

# Regression Command 
folderPath = '/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/Surface_Complex_CTRL_Deformetrica/'
outFolderPath = '/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/Surface_Complex_CTRL_Deformetrica/output_atlas'
baselineFolderPath = '/media/shong/IntHard1/4DAnalysis/IPMI2019/GeodesicRegressionResults/Surface_Complex_CTRL_Deformetrica/Init_Baseline/' 

paramModelPath = folderPath + 'model_atlas.xml'
paramDataSetPath = folderPath + 'data_set_atlas.xml'
paramOptPath = folderPath + 'optimization_parameters_atlas.xml' 

# Model Parameter XML
model_f = open( paramModelPath, "w" )
model_f.write( "<?xml version=\"1.0\"?>\n" )
model_f.write( "<model>\n" )

model_f.write( "  <model-type>DeterministicAtlas</model-type>\n" )
model_f.write( "  <dimension>3</dimension>\n")
# model_f.write( "  <initial-cp-spacing>8</initial-cp-spacing>\n" )

model_f.write( "  <template>\n" )
denseModeStr = "        <dense-mode>Off</dense-mode>\n"
deformObjTypeStr = "           <deformable-object-type>SurfaceMesh</deformable-object-type>\n"
attachmentTypeStr = "           <attachment-type>Varifold</attachment-type>\n"
kernelTypeStr = "           <kernel-type>keops</kernel-type>\n"
kernelWidthStr = "           <kernel-width>6.0</kernel-width>\n"
noiseStdStr = "           <noise-std>0.5</noise-std>\n"
model_f.write( denseModeStr )

for p in range( len( anatomy_list ) ):
	objIdStr = "        <object id=\"" + anatomy_list[ p ] + "\">\n"
	model_f.write( objIdStr )
	model_f.write( deformObjTypeStr )
	model_f.write( attachmentTypeStr )
	model_f.write( kernelTypeStr )
	model_f.write( kernelWidthStr )

	initTemplateAnatomyPath = baselineFolderPath + 'init_baseline_' + anatomy_list[ p ] + '.vtk'
	fileNameStr = "           <filename>" + initTemplateAnatomyPath + "</filename>\n"
	model_f.write( fileNameStr )
	model_f.write( noiseStdStr )
	model_f.write( "        </object>\n")

	# Change VTK Legacy Format to VTK 7.0 standard format
	temp_reader = vtk.vtkPolyDataReader()
	temp_reader.SetFileName( initTemplateAnatomyPath )
	temp_reader.Update()

	temp_writer = vtk.vtkPolyDataWriter()
	temp_writer.SetFileName( initTemplateAnatomyPath )
	temp_writer.SetInputData( temp_reader.GetOutput() )
	temp_writer.Update()
	temp_writer.Write()

model_f.write( "  </template>\n")
model_f.write( "  <deformation-parameters>\n")
model_f.write( "    <kernel-type>keops</kernel-type>\n" )
model_f.write( "    <kernel-width>12.0</kernel-width>\n" )
# model_f.write( "    <noise-std>1.0</noise-std>\n" )

# nTimePt = int( float( ages[-1] ) - float( ages[0] ) + 0.5 )
# print nTimePt
# model_f.write( "    <concentration-of-timepoints> 20 </concentration-of-timepoints>\n")
# model_f.write( "    <t0>" + str( 25 ) + "</t0>\n" )
# model_f.write( "    <tN>" + str( 80 ) + "</tN>\n" )
model_f.write( "  </deformation-parameters>\n")

model_f.write( "</model>\n" )
model_f.close()

# Optimization Parameters
opt_f = open( paramOptPath, "w" )
opt_f.write( "<?xml version=\"1.0\"?>\n" )
opt_f.write( "<optimization-parameters>\n" )
opt_f.write( "    <optimization-method-type>GradientAscent</optimization-method-type>\n" )
opt_f.write( "    <max-iterations>500</max-iterations>\n" )
opt_f.write( "    <initial-step-size>1e-10</initial-step-size>\n" )
opt_f.write( "    <scale-initial-step-size>Off</scale-initial-step-size>\n" )
opt_f.write( "    <scale-initial-step-size>Off</scale-initial-step-size>\n" )
opt_f.write( "    <convergence-tolerance>1e-10</convergence-tolerance>\n" )
opt_f.write( "    <freeze-template>Off</freeze-template>\n" )
opt_f.write( "    <freeze-control-points>Off</freeze-control-points>\n" )
opt_f.write( "    <use-cuda>On</use-cuda>\n" )
opt_f.write( "</optimization-parameters>\n" )
opt_f.close()




# Data Set Parameter XML
dataSet_f = open( paramDataSetPath, "w" )
dataSet_f.write( "<?xml version=\"1.0\"?>\n" )
dataSet_f.write( "<data-set>\n" )

ageList = []

for i in range( len( dataInfoList )  ):
# for i in range( 10 ):
	subj_dataFolder = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID
	if not os.path.isdir( subj_dataFolder ):
		# print( 'PHD-AS1-' + dataInfoList[i].ID + "does not exist" )
		continue

	# Skip if there is only one shape in the list 
	if len( dataInfoList[i].AgeList ) < 2:
		# print( dataInfoList[i].ID + "has less than 2 data" )
		continue

	for j in range( len( dataInfoList[i].LabelList ) ):
		if j > 0:
			break

		if not dataInfoList[i].CAPGroupList[ j ] == 'cont':
			continue

		isAllAnatomy = True

		for anatomy in anatomy_list:
			subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/full_res_aligned/"

			# print( "ID : " + dataInfoList[i].ID + ", Label : " + dataInfoList[i].LabelList[j ] )

			input_path = subj_i_label_j_folderPath + anatomy + ".vtk"

			if not os.path.isfile( input_path ):
				isAllAnatomy = False
				break

		if not isAllAnatomy:
			continue

		dataSet_f.write( "  <subject id=\"" + dataInfoList[i].ID + "\">\n" )

		dataSet_f.write( "    <visit id=\"Initial\">\n" )
		# dataSet_f.write( "      <age>" + str( dataInfoList[i].AgeList[ j ] ) + "</age>\n" )

		ageList.append( dataInfoList[i].AgeList[ j ] )

		for anatomy in anatomy_list:
			subj_i_label_j_folderPath = dataFolderPath + 'PHD-AS1-' + dataInfoList[i].ID + "/" + dataInfoList[i].LabelList[j ] + "/surfaces/full_res_aligned/"
			input_path = subj_i_label_j_folderPath + anatomy + ".vtk"

			fileNameStr = "      <filename object_id=\"" + anatomy + "\">" + input_path + "</filename>\n"
			dataSet_f.write( fileNameStr )

		dataSet_f.write( "    </visit>\n" ) 

		dataSet_f.write( "  </subject>\n" )	
		
dataSet_f.write( "</data-set>\n" )
dataSet_f.close()

RegCommand = regExePath + " --output " + outFolderPath + " " + paramModelPath + " " + paramDataSetPath + " --p " + paramOptPath  
print( RegCommand )
os.system( RegCommand )

print( np.min( ageList )) 
print( np.max( ageList )) 