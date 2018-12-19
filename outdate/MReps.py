import atom

class MRep( object ):
	def __init__(self):
		self.nAtoms = 0
		self.atom_list = []
		self.mean_radius = 0

	def AppendAtom( self, ind_atom= atom.mrep_atom() ):
		self.atom_list.append( ind_atom )
		self.mean_radius = ( self.mean_radius * self.nAtoms + ind_atom.rad ) / ( self.nAtoms + 1 )
		self.nAtoms += 1 

	def SetAtom( self, idx, ind_atom= atom.mrep_atom() ):
		self.atom_list[ idx ] = ind_atom
		self.UpdateMeanRadius()

	def UpdateMeanRadius(self):
		sumRad = 0
		for i in range( self.nAtoms ):
			sumRad += self.atom_list[ i ].rad

		self.mean_radius = ( sumRad ) / self.nAtoms

	# def norm( self, another_mrep=MRep() ):
	# 	if another_mrep.nAtoms == 0:
	
	# def normSquared( self, another_mrep=MRep() ):
	# 	if another_mrep.nAtoms == 0:
	
	# def InnerProduct( self, another_mrep=MRep() ):
	# 	if another_mrep.nAtoms == 0:

class CMRep( object ):
	def __init__(self):
		self.nAtoms = 0 
		self.atom_list = [] 
		self.mean_radius = 0

	def AppendAtom( self, ind_atom= atom.cmrep_atom() ):
		self.atom_list.append( ind_atom )
		self.mean_radius = ( self.mean_radius * self.nAtoms + ind_atom.rad ) / ( self.nAtoms + 1 )
		self.nAtoms += 1 

	def SetAtom( self, idx, ind_atom= atom.mrep_atom() ):
		self.atom_list[ idx ] = ind_atom
		self.UpdateMeanRadius()
		
	def UpdateMeanRadius(self):
		sumRad = 0
		for i in range( self.nAtoms ):
			sumRad += self.atom_list[ i ].rad

		self.mean_radius = ( sumRad ) / self.nAtoms

	# def norm( self, another_mrep=MRep() )
	# 	if another_mrep.nAtoms == 0:
	
	# def normSquared( self, another_mrep=MRep() )
	# 	if another_mrep.nAtoms == 0:
	
	# def InnerProduct( self, another_mrep=MRep() )
	# 	if another_mrep.nAtoms == 0:
