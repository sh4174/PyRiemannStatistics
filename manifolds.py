import numbers
import numpy as np

import pickle


class euclidean_tVec( object ):
	# def __init__( self ):
	# 	self.Type = "Euclidean_Tangent"
	# 	self.nDim = 3
	# 	self.tVector = [ 0, 0, 0 ]

	def __init__( self, nDim ):
		self.Type = "Euclidean_Tangent"
		self.nDim = nDim
		self.tVector = np.zeros( nDim )

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector(self, tVec):
		if not len( tVec ) == self.nDim:
 			print( "Error : Dimensions does not match" )
 			return

		self.tVector = tVec

	def InnerProduct( self, tVec1 ):
		result = 0
		for i in range( self.nDim ):
			result += self.tVector[ i ] * tVec1.tVector[ i ]
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )		

	def ScalarMultiply( self, t ):
		tVector_t = euclidean_tVec( self.nDim )

		for i in range( self.nDim ):
			tVector_t.tVector[ i ] = self.tVector[ i ] * t

		return tVector_t

	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.tVector, False ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.tVector = infoList[ 2 ] 



class euclidean( object ):
	# def __init__( self ):
	# 	self.Type = "Euclidean"
	# 	self.nDim = 3
	# 	self.pt = [ 0.0, 0.0, 0.0 ] 

	def __init__( self, nDim ):
		self.Type = "Euclidean"
		self.nDim = nDim
		self.pt = np.zeros( nDim )

	def SetPoint( self, pt ):
		if not len( pt ) == self.nDim:
			print( "Error : Dimensions does not match" )
			return
		self.pt = pt

	def GetPoint( self ):
		return self.pt

	def InnerProduct( self, ptA ):
		result = 0
		for i in range( self.nDim ):
			result += self.pt[ i ] * ptA.pt[ i ]
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )		

	def ExponentialMap( self, tVec ):
		exp_pt = euclidean( self.nDim )

		newPt_mat = np.zeros( self.nDim )

		for i in range( self.nDim ):
			newPt_mat[ i ] = self.pt[ i ] + tVec.tVector[ i ]

		exp_pt.SetPoint( newPt_mat.tolist() )
		return exp_pt

	def LogMap( self, ptA ):
		tVec = euclidean_tVec( self.nDim )

		tVec_list = []

		for i in range( self.nDim ):
			tVec_i = ptA.pt[ i ] - self.pt[ i ]

			tVec_list.append( tVec_i )

		tVec.SetTangentVector( tVec_list ) 
		return tVec

	def RiemannianDistanceToA( self, ptA ):
		tVec_toA = self.LogMap( ptA )
		distSq = tVec_toA.norm()
		dist = np.sqrt( distSq )
		return dist			

	def ProjectTangent( self, pt, tVec ):
		vProjected = tVec
		return vProjected

	def ParallelTranslate( self, v, w ):
		return w

	def ParallelTranslateToA( self, ptA, w ):
		return w

	def ParallelTranslateAtoB( self, ptA, ptB, w ):
		return w

	def AdjointGradientJacobi( self, v, j, dj ):
		# Function Output
		jOutput = j
		jOutputDash = dj

		jOutputDash_list = []

		for i in range( self.nDim ):
			jOutputDash_i = j.tVector[i] + dj.tVector[i]
			jOutputDash_list.append( jOutputDash_i )

		jOutputDash.SetTangentVector( jOutputDash_list ) 
		return jOutput, jOutputDash

	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.pt, False ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.pt = infoList[ 2 ] 


class sphere_tVec( object ):
	# def __init__( self ):
	# 	self.Type = "Sphere_Tangent"
	# 	self.nDim = 3
	# 	self.tVector = [ 0, 0, 0 ]

	def __init__( self, nDim ):
		self.Type = "Sphere_Tangent"
		self.nDim = nDim
		self.tVector = np.zeros( nDim )

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector(self, tVec):
		if not len( tVec ) == self.nDim:
			print( "Error : Dimensions does not match" )
			return

		self.tVector = tVec

	def InnerProduct( self, tVec1 ):
		result = 0
		for i in range( self.nDim ):
			result += self.tVector[ i ] * tVec1.tVector[ i ]
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )	

	def ScalarMultiply( self, t ):
		tVector_t = sphere_tVec( self.nDim )

		for i in range( self.nDim ):
			tVector_t.tVector[ i ] = self.tVector[ i ] * t

		return tVector_t

	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.tVector, False ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.tVector = infoList[ 2 ] 


class sphere( object ):
	# def __init__( self ):
	# 	self.Type = "Sphere"
	# 	self.nDim = 3
	# 	self.pt = [ 1.0, 0.0, 0.0 ] # Base point in S^2

	def __init__( self, nDim ):
		self.Type = "Sphere"
		self.nDim = nDim
		pt_base = np.zeros( nDim )
		pt_base[ 0 ] = 1 
		self.pt = pt_base

	def SetPoint( self, pt ):
		if not len( pt ) == self.nDim:
			print( "Error : Dimensions does not match" )
			return

		if not np.linalg.norm( pt ) == 1:
			# print( "Warning : The point is not on a sphere")
			self.pt = pt
			return

		self.pt = pt

	def GetPoint( self ):
		return self.pt

	def InnerProduct( self, ptA ):
		result = 0
		for i in range( self.nDim ): 
			result += self.pt[ i ] * ptA.pt[ i ]
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )		

	def ProjectTangent( self, pt, tVec ):
		inner_prod_pt_tVec = 0
		for i in range( self.nDim ):
			inner_prod_pt_tVec += pt.pt[i] * tVec.tVector[i]

		vProjected = sphere_tVec( self.nDim )

		for i in range( self.nDim ):
			vProjected.tVector[i] = tVec.tVector[i] - inner_prod_pt_tVec * pt.pt[i] 

		return vProjected

	def ParallelTranslate( self, v, w ):
		vNorm = v.norm()

		if( vNorm < 1.0e-12 ):
			return w

		innerProd = v.InnerProduct( w )
		scaleFactor = innerProd / ( vNorm * vNorm )

		# Component of w orthogonal to v
		orth = sphere_tVec( self.nDim )
		for i in range( self.nDim ):
			orth.tVector[ i ] = w.tVector[ i ] - v.tVector[ i ] * scaleFactor

		# Compute parallel translated v
		vParallel = sphere_tVec( self.nDim )

		for i in range( self.nDim ):
			vParallel.tVector[ i ] = self.pt[ i ]* ( -np.sin( vNorm ) * vNorm ) + v.tVector[ i ] * np.cos( vNorm )

		# Parallel Translated w
		wParallelTranslated = sphere_tVec( self.nDim )

		for i in range( self.nDim ):
			wParallelTranslated.tVector[ i ] = vParallel.tVector[ i ] * scaleFactor + orth.tVector[ i ]

		return wParallelTranslated

	def ParallelTranslateToA( self, ptA, w ):
		v = self.LogMap( ptA )
		vNorm = v.norm()

		if( vNorm < 1.0e-12 ):
			return w

		innerProd = v.InnerProduct( w )
		scaleFactor = innerProd / ( vNorm * vNorm )

		# Component of w orthogonal to v
		orth = sphere_tVec( self.nDim )
		for i in range( self.nDim ):
			orth.tVector[ i ] = w.tVector[ i ] - v.tVector[ i ] * scaleFactor

		# Compute parallel translated v
		vParallel = sphere_tVec( self.nDim )

		for i in range( self.nDim ):
			vParallel.tVector[ i ] = self.pt[ i ]* ( -np.sin( vNorm ) * vNorm ) + v.tVector[ i ] * np.cos( vNorm )

		# Parallel Translated w
		wParallelTranslated = sphere_tVec( self.nDim )

		for i in range( self.nDim ):
			wParallelTranslated.tVector[ i ] = vParallel.tVector[ i ] * scaleFactor + orth.tVector[ i ]

		return wParallelTranslated

	def AdjointGradientJacobi( self, v, j, dj ):
		e_base = self

		vNorm = v.norm() 

		if vNorm < 1.0e-12:
			for i in range( self.nDim ):
				dj.tVector[ i ] = j.tVector[ i ] + dj.tVector[ i ]

			# Function Output
			jOutput = j
			jOutputDash = dj
		else:
			innerProdVJ = v.InnerProduct( j )
			innerProdVJPrime = v.InnerProduct( dj )

			scaleFactorJ = innerProdVJ / ( vNorm * vNorm )
			scaleFactorJPrime = innerProdVJPrime / ( vNorm * vNorm )

			jTang = sphere_tVec( self.nDim )
			djTang = sphere_tVec( self.nDim )
			jOrth = sphere_tVec( self.nDim )
			djOrth = sphere_tVec( self.nDim )

			for i in range( self.nDim ):
				jTang.tVector[ i ] = v.tVector[ i ] * scaleFactorJ
				djTang.tVector[ i ] = v.tVector[ i ] * scaleFactorJPrime

				jOrth.tVector[ i ] = j.tVector[ i ] - jTang.tVector[ i ]
				djOrth.tVector[ i ] = dj.tVector[ i ] - djTang.tVector[ i ]

				j.tVector[ i ] = jTang.tVector[ i ] + ( np.cos( vNorm ) * jOrth.tVector[ i ] ) - ( ( vNorm * np.sin( vNorm ) ) * djOrth.tVector[ i ] )

			j = e_base.ParallelTranslate( v, j )

			for i in range( self.nDim ):
				dj.tVector[ i ] = jTang.tVector[ i ] + djTang.tVector[ i ] + ( np.sin(vNorm) / vNorm ) * jOrth.tVector[ i ] + np.cos( vNorm ) * djOrth.tVector[ i ]

			dj = e_base.ParallelTranslate( v, dj )

			# Function Output
			jOutput = j
			jOutputDash = dj

		return jOutput, jOutputDash

	def ParallelTranslateAtoB( self, a, b, w ):
		v = a.LogMap( b )
		return a.ParallelTranslate( v, w )

	def ExponentialMap( self, tVec ):
		theta = np.linalg.norm( tVec.tVector )

		if theta < 1e-12:
			exp_pt = sphere( self.nDim )
			exp_pt.pt = self.pt			
			return exp_pt

		if theta > np.pi * 2:
			theta = np.mod( theta, np.pi * 2 )

		exp_pt = sphere( self.nDim )

		lhs = np.multiply( np.cos( theta ), self.pt )
		rhs = np.multiply( np.sin( theta ) / theta, tVec.tVector )

		exp_pt.pt = lhs + rhs
		exp_pt.pt = np.divide( exp_pt.pt, exp_pt.norm() ) 

		return exp_pt

	def LogMap( self, another_pt ):
		cosTheta = self.InnerProduct( another_pt )
		tVec = sphere_tVec( self.nDim )

		for i in range( self.nDim ):
			tVec.tVector[ i ] = another_pt.pt[ i ] - cosTheta * self.pt[ i ]

		length = tVec.norm()
		if length < 1e-12 or cosTheta >= 1.0 or cosTheta <= -1.0:
			tVec = sphere_tVec( self.nDim ) 
			return tVec

		for i in range( self.nDim ):
			tVec.tVector[ i ] = tVec.tVector[ i ] * np.arccos( cosTheta ) / length

		return tVec

	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.pt, False ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.pt = infoList[ 2 ] 


class pos_real_tVec( object ):
	def __init__( self, nDim ):
		self.Type = "PositiveReal_Tangent"
		self.nDim = nDim
		self.tVector = np.zeros( nDim ).tolist()

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector(self, tVec):
		if type( tVec ) == list:
			if not len( tVec ) == self.nDim:
				print( "Error : Dimensions does not match" )
				return 
			else:
				self.tVector = tVec
		else:
			if not self.nDim == 1:
				print( "Error : Dimensions does not match" )
				return
			else:
				self.tVector[ 0 ] = tVec

	def InnerProduct( self, tVec1 ):
		result = 0
		for i in range( self.nDim ): 
			result += self.tVector[ i ] * tVec1.tVector[ i ]
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )	

	def ScalarMultiply( self, t ):
		tVector_t = pos_real_tVec( self.nDim )

		for i in range( self.nDim ):
			tVector_t.tVector[ i ] = self.tVector[ i ] * t

		return tVector_t

	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.tVector, False ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.tVector = infoList[ 2 ] 


class pos_real( object ):
	def __init__( self, nDim ):
		self.Type = "PositiveReal"
		self.nDim = nDim
		self.pt = np.ones( nDim ).tolist()	

	def SetPoint( self, pt ):
		if type( pt ) == list:
			if not len( pt ) == self.nDim:
				print( "Error : Dimensions does not match" )
				return 
			else:
				self.pt = pt
		else:
			if not self.nDim == 1:
				print( "Error : Dimensions does not match" )
				return
			else:
				self.pt[ 0 ] = pt


	def GetPoint( self ):
		return self.pt

	def InnerProduct( self, ptA ):
		result = 0
		for i in range( self.nDim ): 
			result += self.pt[ i ] * ptA.pt[ i ]
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )		

	def ExponentialMap( self, tVec ):
		exp_pt = pos_real( self.nDim )
		# print( "Tangent Vector" )
		# print( tVec.tVector )
		exp_pt.SetPoint( np.multiply( self.pt, np.exp( tVec.tVector ) ).tolist() ) 
		return exp_pt

	def LogMap( self, another_pt ):
		tVec = pos_real_tVec( self.nDim )
		
		tVec.SetTangentVector( np.log( np.divide( another_pt.pt, self.pt ) ).tolist() )
		return tVec

	def ProjectTangent( self, pt, tVec ):
		vProjected = tVec
		return vProjected

	def ParallelTranslate( self, v, w ):
		return w

	def ParallelTranslateToA( self, ptA, w ):
		return w

	def ParallelTranslateAtoB( self, a, b, w ):
		return w

	def AdjointGradientJacobi( self, v, j, dj, ):
		jOutput = j
		jOutputDash = dj

		for i in range( self.nDim ):
			jOutputDash.tVector[ i ] = j.tVector[ i ] + dj.tVector[ i ] 

		return jOutput, jOutputDash


	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.pt, False ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.pt = infoList[ 2 ] 


class cmrep_tVec( object ):
	def __init__( self, nDim ):
		self.Type = "CMRep_Tangent"
		self.nDim = nDim
		self.tVector = []

		for i in range( self.nDim ):
			self.tVector.append( [ euclidean_tVec( 3 ), pos_real_tVec( 1 ) ] )

		self.meanRadius = 1

	def SetPositionTangentVector( self, idx, pos_tVec ):
		self.tVector[ idx ][ 0 ].SetTangentVector( pos_tVec )

	def SetRadiusTangentVector( self, idx, rad_tVec ):
		self.tVector[ idx ][ 1 ].SetTangentVector( rad_tVec )

	def SetMeanRadius( self, meanRadius ):
		self.meanRadius = meanRadius

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector( self, tVec ):
		if not len( tVec ) == self.nDim:
			print( "Error : Dimension Mismatch" )
			return 

		self.tVector = tVec

	def InnerProduct( self, tVec1 ):
		result = 0
		for i in range( self.nDim ): 
			result += self.tVector[ i ][ 0 ].InnerProduct( tVec1.tVector[ i ][ 0 ] )
			result += self.meanRadius * tVec1.meanRadius * self.tVector[ i ][ 1 ].InnerProduct( tVec1.tVector[ i ][ 1 ] )
			
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )	

	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.tVector, self.meanRadius ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.tVector = infoList[ 2 ] 
		self.meanRadius = infoList[ 3 ] 

class cmrep( object ):
	def __init__( self, nDim ):
		self.Type = "CMRep"
		self.nDim = nDim
		self.pt = []
		self.pos = []
		self.rad = []

		for i in range( nDim ):
			self.pt.append( [ euclidean( 3 ), pos_real( 1 ) ] )
			self.pos.append( self.pt[ i ][ 0 ] )
			self.rad.append( self.pt[ i ][ 1 ] )
		self.meanRadius = 1

	def SetPoint( self, pt ):
		if not len( pt ) == self.nDim:
			print( "Error : Dimensions does not match" )
			return 
		self.pt = pt
		self.pos = [] 
		self.rad = [] 

		for i in range( self.nDim ):
			self.pos.append( self.pt[ i ][ 0 ] )
			self.rad.append( self.pt[ i ][ 1 ] )

	def UpdateMeanRadius( self ):
		meanRad = 0
		for i in range( self.nDim ):
			meanRad += ( float( self.rad[ i ].pt[0] ) / float( self.nDim ) )

		self.meanRadius = meanRad
		return

	def AppendAtom( self, pt = [ euclidean( 3 ), pos_real( 1 ) ] ):
		self.nDim = self.nDim + 1
		self.pt.append( pt ) 
		self.pos.append( self.pt[ self.nDim - 1 ][ 0 ] )
		self.rad.append( self.pt[ self.nDim - 1 ][ 1 ] )

		self.meanRadius = ( self.meanRadius * ( self.nDim - 1 ) + pt[ 1 ].pt[ 0 ] ) / self.nDim
		

	def SetPosition( self, idx, position=[0.0,0.0,0.0] ):
		self.pos[ idx ].SetPoint( position )
		self.pt[ idx ][ 0 ].SetPoint( position )

	def SetRadius( self, idx, rad=1.0 ):
		self.rad[ idx ].SetPoint( rad )
		self.pt[ idx ][ 1 ].SetPoint( rad )		

	def ExponentialMap( self, tVec ):
		if not tVec.Type == "CMRep_Tangent":
			print( "Tangent Vector Type Mismatched" )
			return
		exp_pt = cmrep( self.nDim )		
		for i in range( self.nDim ):
			exp_pt.pt[ i ][ 0 ] = self.pt[ i ][ 0 ].ExponentialMap( tVec.tVector[ i ][ 0 ]  )
			exp_pt.pt[ i ][ 1 ] = self.pt[ i ][ 1 ].ExponentialMap( tVec.tVector[ i ][ 1 ]  )

		return exp_pt

	def LogMap( self, another_pt ):
		if not another_pt.Type == "CMRep":
			print( "Error: Component Type Mismatched" )
			return

		tVec = cmrep_tVec( self.nDim )

		for i in range( self.nDim ):
			tVec.tVector[ i ][ 0 ] = self.pt[ i ][ 0 ].LogMap( another_pt.pt[ i ][ 0 ] )
			tVec.tVector[ i ][ 1 ] = self.pt[ i ][ 1 ].LogMap( another_pt.pt[ i ][ 1 ] )

		return tVec		

	def GetPoistion( self ):
		return self.pos

	def GetRadius( self ):
		return self.rad

	def InnerProduct( self, ptA ):
		result = 0
		for i in range( self.nDim ): 
			result += self.pt[ i ][ 0 ].InnerProduct( ptA.pt[ i ][ 0 ] )
			result += self.meanRadius * ptA.meanRadius * self.pt[ i ][ 1 ].InnerProduct( ptA.pt[ i ][ 1 ] )
			
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )	

	def ProjectTangent( self, pt, tVec ):
		vProjected = cmrep_tVec( self.nDim )

		for i in range( self.nDim ):
			vProjected.tVector[ i ][ 0 ] = self.pt[ i ][ 0 ].ProjectTangent( pt.pt[ i ][ 0 ], tVec.tVector[ i ][ 0 ] )
			vProjected.tVector[ i ][ 1 ] = self.pt[ i ][ 1 ].ProjectTangent( pt.pt[ i ][ 1 ], tVec.tVector[ i ][ 1 ] )

		return vProjected

	def ParallelTranslate( self, v, w ):
		wPrallelTranslated = cmrep_tVec( self.nDim )

		for i in range( self.nDim ):
			wPrallelTranslated.tVector[ i ][ 0 ] = self.pt[ i ][ 0 ].ParallelTranslate( v.tVector[ i ][ 0 ], w.tVector[ i ][ 0 ] )
			wPrallelTranslated.tVector[ i ][ 1 ] = self.pt[ i ][ 1 ].ParallelTranslate( v.tVector[ i ][ 1 ], w.tVector[ i ][ 1 ] )

		return wParallelTranslated

	def ParallelTranslateToA( self, ptA, w ):
		v = self.LogMap( ptA )
		return ParallelTranslate( v, w )

	def ParallelTranslateAtoB( self, a, b, w ):
		v = a.LogMap( b )
		return a.ParallelTranslate( v, w )

	def AdjointGradientJacobi( self, v, j, dj ):
		e_base = self
		vNorm = v.norm() 
		jOutput = cmrep_tVec( self.nDim )
		jOutputDash = cmrep_tVec( self.nDim )

		for i in range( self.nDim ):
			jOutput.tVector[ i ][ 0 ], jOutputDash.tVector[ i ][ 0 ] = self.pt[ i ][ 0 ].AdjointGradientJacobi( v.tVector[ i ][ 0 ], j.tVector[ i ][ 0 ], dj.tVector[ i ][ 0 ] )
			jOutput.tVector[ i ][ 1 ], jOutputDash.tVector[ i ][ 1 ] = self.pt[ i ][ 1 ].AdjointGradientJacobi( v.tVector[ i ][ 1 ], j.tVector[ i ][ 1 ], dj.tVector[ i ][ 1 ] )

		return jOutput, jOutputDash

	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.pt, self.meanRadius ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.pt = infoList[ 2 ] 
		self.meanRadius = infoList[ 3 ] 


class cmrep_abstract( object ):
	def __init__( self, nDim ):
		self.Type = "CMRep_Abstract"
		self.nDim = nDim
		# pt : center, scale, abstract position, radius
		self.pt = [ euclidean(3), pos_real(1), sphere( 3 * ( nDim - 1 ) ), pos_real( nDim ) ]
		self.center = self.pt[ 0 ]
		self.scale = self.pt[ 1 ] 
		self.pos = self.pt[ 2 ]		
		self.rad = self.pt[ 3 ]
		self.meanRadius = 1

	def SetPoint( self, pt ):
		if not ( len( pt ) == 4 and pt[ 0 ].nDim == 3 and pt[ 1 ].nDim == 1 and pt[ 2 ].nDim == 3 * ( self.nDim - 1 ) and pt[ 3 ].nDim == self.nDim ):
			print( "cmrep_abstract.SetPoint")
			print( "Error : Dimensions does not match" )
			return 
		self.pt = pt
		self.center = pt[ 0 ]
		self.scale = pt[ 1 ] 
		self.pos = pt[ 2 ]
		self.rad = pt[ 3 ]

		self.UpdateMeanRadius()

	def UpdateMeanRadius( self ):
		meanRad = 0
		for i in range( self.nDim ):
			meanRad += ( float( self.rad.pt[i] ) / float( self.nDim ) )

		self.meanRadius = meanRad
		return

	def ExponentialMap( self, tVec ):
		if not tVec.Type == "CMRep_Abstract_Tangent":
			print( "Tangent Vector Type Mismatched" )
			return

		exp_pt = cmrep_abstract( self.nDim )
		exp_pt_arr = []

		for i in range( 4 ):
			exp_pt_arr.append( self.pt[ i ].ExponentialMap( tVec.tVector[ i ] ) )

		exp_pt.SetPoint( exp_pt_arr )

		return exp_pt

	def LogMap( self, another_pt ):
		if not another_pt.Type == "CMRep_Abstract":
			print( "Error: Component Type Mismatched" )
			return

		tVec = cmrep_abstract_tVec( self.nDim )

		for i in range( 4 ):
			tVec.tVector[ i ] = self.pt[ i ].LogMap( another_pt.pt[ i ] )

		return tVec		

	def GetScale( self ):
		return self.scale

	def GetCenter( self ):
		return self.center

	def GetPoistion( self ):
		return self.pos

	def GetRadius( self ):
		return self.rad

	def InnerProduct( self, ptA ):
		result = 0

		# Center
		result += self.pt[ 0 ].InnerProduct( ptA.pt[ 0 ] )

		# Scale
		result += self.meanRadius * ptA.meanRadius * self.pt[ 1 ].InnerProduct( ptA.pt[ 1 ] )

		# Abstract Position
		result += self.meanRadius * ptA.meanRadius * self.pt[ 2 ].InnerProduct( ptA.pt[ 2 ] )

		# Radius
		result += self.meanRadius * ptA.meanRadius * self.pt[ 3 ].InnerProduct( ptA.pt[ 3 ] )
		
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )	

	def ProjectTangent( self, pt, tVec ):
		vProjected = cmrep_abstract_tVec( self.nDim )

		for i in range( 4 ):
			vProjected_tVector[ i ] = self.pt[ i ].ProjectTangent( pt.pt[ i ], tVec.tVector[ i ] )

		return vProjected

	def ParallelTranslate( self, v, w ):
		wPrallelTranslated = cmrep_tVec( self.nDim )

		for i in range( 4 ):
			wParallelTranslated.tVector[ i ] = self.pt[ i ].ParallelTranslate( v.tVector[ i ], w.tVector[ i ] )

		return wParallelTranslated

	def ParallelTranslateToA( self, ptA, w ):
		v = self.LogMap( ptA )
		return ParallelTranslate( v, w )

	def ParallelTranslateAtoB( self, a, b, w ):
		v = a.LogMap( b )
		return a.ParallelTranslate( v, w )

	def AdjointGradientJacobi( self, v, j, dj ):
		e_base = self
		vNorm = v.norm() 
		jOutput = cmrep_abstract_tVec( self.nDim )
		jOutputDash = cmrep_abstract_tVec( self.nDim )

		for i in range( 4 ):
			jOutput.tVector[ i ], jOutputDash.tVector[ i ] = self.pt[ i ].AdjointGradientJacobi( v.tVector[ i ], j.tVector[ i ], dj.tVector[ i ] )

		return jOutput, jOutputDash

	def GetEuclideanLocations( self ):
		H_sub = HelmertSubmatrix( self.nDim )
		H_sub_T = H_sub.T

		# Relative Positions on a 3(n-1)-1 sphere
		pos_abstr_sphere_matrix = np.array( self.pt[ 2 ].pt ).reshape( -1, 3 )

		# Relative Positions on Euclidean
		pos_abstr_euclidean_matrix = np.dot( H_sub_T, pos_abstr_sphere_matrix )

		# Multiply Scale 
		# print( self.scale.pt[ 0 ] )
		pos_scale_eucldiean_matrix = np.multiply( pos_abstr_euclidean_matrix, self.scale.pt[ 0 ] )

		# Add Center of Mass 
		pos_world_coord_euclidean_matrix = np.zeros( [ self.nDim, 3 ] )

		for i in range( self.nDim ):
			pos_world_coord_euclidean_matrix[ i, : ] = np.add( pos_scale_eucldiean_matrix[ i, : ], self.center.pt )

		return pos_world_coord_euclidean_matrix

	def GetAbstractEuclideanLocations( self ):
		H_sub = HelmertSubmatrix( self.nDim )
		H_sub_T = H_sub.T

		# Relative Positions on a 3(n-1)-1 sphere
		pos_abstr_sphere_matrix = np.array( self.pt[ 2 ].pt ).reshape( -1, 3 )

		# Relative Positions on Euclidean
		pos_abstr_euclidean_matrix = np.dot( H_sub_T, pos_abstr_sphere_matrix )

		# # Multiply Scale 
		# # print( self.scale.pt[ 0 ] )
		# pos_scale_eucldiean_matrix = np.multiply( pos_abstr_euclidean_matrix, self.scale.pt[ 0 ] )

		# # Add Center of Mass 
		# pos_world_coord_euclidean_matrix = np.zeros( [ self.nDim, 3 ] )

		# for i in range( self.nDim ):
		# 	pos_world_coord_euclidean_matrix[ i, : ] = np.add( pos_scale_eucldiean_matrix[ i, : ], self.center.pt )

		return pos_abstr_euclidean_matrix


	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.pt, self.meanRadius ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.pt = infoList[ 2 ] 
		self.center = self.pt[ 0 ]
		self.scale = self.pt[ 1 ]
		self.pos = self.pt[ 2 ] 
		self.rad = self.pt[ 3 ]
		self.meanRadius = infoList[ 3 ] 


class cmrep_abstract_tVec( object ):
	def __init__( self, nDim ):
		self.Type = "CMRep_Abstract_Tangent"
		self.nDim = nDim
		# tVector : center, scale, abstract position, radius
		self.tVector = [ euclidean_tVec( 3 ), pos_real_tVec( 1 ), sphere_tVec( 3 * ( nDim - 1 ) ), pos_real_tVec( nDim ) ]
		self.meanRadius = 1

	# def SetPositionTangentVector( self, idx, pos_tVec ):
	# 	self.tVector[ idx ][ 0 ].SetTangentVector( pos_tVec )

	# def SetRadiusTangentVector( self, idx, rad_tVec ):
	# 	self.tVector[ idx ][ 1 ].SetTangentVector( rad_tVec )

	def SetMeanRadius( self, meanRadius ):
		self.meanRadius = meanRadius

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector( self, tVec ):
		if not ( len( tVec ) == 4 and tVec[ 0 ].nDim == 3 and tVec[ 1 ].nDim == 1 and tVec[ 2 ].nDim == ( 3 * ( self.nDim - 1 ) ) and tVec[ 3 ].nDim == self.nDim ):
			print( "Error : Dimension Mismatch" )
			return 

		self.tVector = tVec

	def InnerProduct( self, tVec1 ):
		result = 0

		# Center 
		result += self.tVector[ 0 ].InnerProduct( tVec1.tVector[ 0 ] )

		# Scale
		result += self.meanRadius * tVec1.meanRadius * self.tVector[ 1 ].InnerProduct( tVec1.tVector[ 1 ] )

		# Abstract Position
		result += self.meanRadius * tVec1.meanRadius * self.tVector[ 2 ].InnerProduct( tVec1.tVector[ 2 ] )

		# Radius
		result += self.meanRadius * tVec1.meanRadius * self.tVector[ 3 ].InnerProduct( tVec1.tVector[ 3 ] )
			
		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )	

	def ScalarMultiply( self, t ):
		tVector_t = cmrep_abstract_tVec( self.nDim )

		for i in range( 4 ):
			tVector_t.tVector[ i ] = self.tVector[ i ].ScalarMultiply( t )

		return tVector_t


	def Write( self, filePath ):
		infoList = [ self.Type, self.nDim, self.tVector, self.meanRadius ]
		
		with open( filePath, 'wb' ) as fp:
			pickle.dump( infoList, fp )

	def Read( self, filePath ):
		with open( filePath, 'rb' ) as fp:
			infoList = pickle.load( fp )

		self.Type = infoList[ 0 ]
		self.nDim = infoList[ 1 ]
		self.tVector = infoList[ 2 ] 
		self.meanRadius = infoList[ 3 ] 


###############################################################
#####					 Miscelleneous		 			  #####
###############################################################
def HelmertSubmatrix( nAtoms ):
	# Create a Helmert submatrix - similarity-invariant
	H = np.zeros( [ nAtoms - 1, nAtoms ] )

	for k in range( nAtoms - 1 ):
		h_k = -np.divide( 1.0, np.sqrt( ( k + 1 ) * ( k + 2 ) ) )
		neg_kh_k = np.multiply( h_k, -( k + 1 ) )
		for h in range( k + 1 ):
			H[ k, h ] = h_k
		H[ k, k + 1 ] = neg_kh_k 

	return H


def HelmertMatrix( nAtoms ):
	# Create a Helmert matrix - similiarity-invariant : First row - Center of Gravity (mass) (uniform mass of points)
	H_full = np.zeors( [ nAtoms, nAtoms ] )

	for h in range( nAtoms ):
		H_full[ 0, h ] = np.divide( 1, np.sqrt( nAtoms ) )

	for k in range( 1, nAtoms, 1 ):
		h_k = -np.divide( 1.0, np.sqrt( ( k ) * ( k + 1 ) ) )
		neg_kh_k = np.multiply( h_k, -k )
		for h in range( k ):
			H_full[ k, h ] = h_k
		H_full[ k, k ] = neg_kh_k 

	return H_full

'''
class mrep_tVec( object ):
	def __init__( self ):
		self.Type = "MRep_Tangent"
		self.nDim = 1
		self.tVector = [ euclidean_tVec(), pos_real_tVec(), sphere_tVec(), sphere_tVec() ]
		self.meanRadius = 1

	def __init__( self, nDim ):
		self.Type = "MRep_Tangent"
		self.nDim = nDim
		self.tVector = []
		self.meanRadius = 1

		for i in range( nDim ):
			self.tVector.append( [ euclidean_tVec(), pos_real_tVec(), sphere_tVec(), sphere_tVec() ] )

	def __init__( self, nDim, meanRadius ):
		self.Type = "MRep_Tangent"
		self.nDim = nDim
		self.tVector = []
		self.meanRadius = meanRadius

		for i in range( nDim ):
			self.tVector.append( [ euclidean_tVec(), pos_real_tVec(), sphere_tVec(), sphere_tVec() ] )

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector( self, tVec ):
		if not type( tVec[0] ) == list:
			if not nDim == 1:
				print( "Error : Dimensions does not match" )
				return
			else:
				self.tVector = tVec
		else:
			if not len( tVec ) == nDim:
				print( "Error : Dimensions odes not match" )
				return
			else:
				self.tVector = tVec

	def InnerProduct( self, tVec1 ):
		result = 0
		for i in range( self.nDim ): 
			result += self.tVector[ i ][ 0 ].InnerProduct( tVec1.tVector[ i ][ 0 ] )
			result += self.meanRadius * tVec1.meanRadius * self.tVector[ i ][ 1 ].InnerProduct( tVec1.tVector[ i ][ 1 ] )
			result += self.meanRadius * tVec1.meanRadius * self.tVector[ i ][ 2 ].InnerProduct( tVec1.tVector[ i ][ 2 ] )
			result += self.meanRadius * tVec1.meanRadius * self.tVector[ i ][ 3 ].InnerProduct( tVec1.tVector[ i ][ 3 ] )

		return result

	def normSquared( self ):
		return self.InnerProduct( self )

	def norm( self ):
		return np.sqrt( self.normSquared() )	

class mrep( object ):
	def __init__( self ):
		self.Type = "MRep"
		self.nDim = 1
		self.pt = [ euclidean(), pos_real(), sphere(), sphere() ] 
		self.pos = pt[ 0 ]
		self.rad = pt[ 1 ] 
		self.sphere_comp1 = pt[ 2 ] # Base point in S^2
		self.sphere_comp2 = pt[ 3 ] # Base point in S^2
		self.meanRadius = self.rad

	def __init__( self, nDim ):
		self.Type = "MRep"
		self.nDim = nDim
		self.pt = []
		self.pos = []
		self.rad = []
		self.sphere_comp1 = []
		self.sphere_comp2 = []
		self.meanRadius = 0

		for i in range( nDim ):
			self.pt.append( [ euclidean(), pos_real(), sphere(), sphere() ] )
			self.pos.append( pt[ i ][ 0 ] )
			self.rad.append( pt[ i ][ 1 ] )
			self.sphere_comp1.append( pt[ i ][ 2 ] )
			self.sphere_comp2.append( pt[ i ][ 3 ] )
			self.meanRadius += ( pt[ i ][ 1 ] / float( nDim ) )


	def SetComponents( self, position=[0.0,0.0,0.0], rad=1.0, sphere_pt1=[0.0,0.0,1.0], sphere_pt2=[0.0, 0.0, 1.0] ):
		if not ( len( position ) == 3 and len( sphere_pt1 ) == 3 and len( sphere_pt2 ) == 3 and isinstance( rad, numbers.Number ) ):
			print( "Error: Component dimension mismatch" )
			return

		self.pos = position
		self.rad = rad
		self.sphere_comp1.sphere_pt = sphere_pt1
		self.sphere_comp2.sphere_pt = sphere_pt2

		print( self )

	def SetPosition( self, position=[0.0,0.0,0.0] ):
		self.pos = position 

	def SetRadius( self, rad=1.0 ):
		self.rad = rad

	def SetSphereComps( self, sphere_pt1=[ 0.0, 0.0, 1.0 ], sphere_pt2=[0.0,0.0,1.0] ):
		self.sphere_comp1.sphere_pt = sphere_pt1
		self.sphere_comp2.sphere_pt = sphere_pt2

	def SetSphereComp1( self, sphere_pt1=[ 0.0, 0.0, 1.0 ] ):
		self.sphere_comp1.sphere_pt = sphere_pt1

	def SetSphereComp2( self, sphere_pt2=[ 0.0, 0.0, 1.0 ] ):
		self.sphere_comp2.sphere_pt = sphere_pt2

	def GetPoistion( self ):
		return self.pos

	def GetRadius( self ):
		return self.rad

	def GetSphereComp1( self ):
		return self.sphere_comp1

	def GetSphereComp2( self ):
		return self.sphere_comp2		

	def ExponentialMap( self, tVec ):
		if not tVec.tVecType == "MRep_Tangent":
			print( "Tangent Vector Type Mismatched" )
			return

		exp_pt = mrep_atom()
		exp_pt.pos = [ self.pos[0] + tVec.tVector[ 0 ][0], self.pos[1] + tVec.tVector[ 0 ][1], self.pos[2] + tVec.tVector[ 0 ][2] ] 
		exp_pt.rad = self.rad * np.exp( tVec.tVector[ 1 ] )
		exp_pt.sphere_comp1 = self.sphere_comp1.ExponentialMap( tVec.tVector[ 2 ] )
		exp_pt.sphere_comp2 = self.sphere_comp2.ExponentialMap( tVec.tVector[ 3 ] )
		return exp_pt

	def LogMap( self, another_pt ):
		if not another_pt.AtomType == "MRep":
			print( "Error: Component Type Mismatched" )
			return

		tVec = mrep_tVec()
		tVec.tVector[ 0 ] = [ another_pt.pos[0] - self.pos[0], another_pt.pos[1] - self.pos[1], another_pt.pos[2] - self.pos[2] ]
		tVec.tVector[ 1 ] = np.log( another_pt.rad / self.rad )
		tVec.tVector[ 2 ] = self.sphere_comp1.LogMap( another_pt.sphere_comp1 )
		tVec.tVector[ 3 ] = self.sphere_comp2.LogMap( another_pt.sphere_comp2 )
		return tVec		
'''

