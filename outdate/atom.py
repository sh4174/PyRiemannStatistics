from abc import ABC, abstractmethod
import numbers
import numpy as np

class atom( ABC ):
	def __init__(self):
		for cls in reversed( self.__class__.mro() ):
			if hasattr( cls, 'init' ):
				cls.init( self )

	def init( self ):
		self.Type = "Atom"
		self.nComponent = 0
		self.nComponent_dim = []
		self.nTangentComponent_dim = []

	@abstractmethod
	def SetComponents( self ):
		print( "atom" )   

	@abstractmethod
	def ExponentialMap( self ):
		pass

	@abstractmethod
	def LogMap( self ):
		pass

class org_atom( atom ):
	def init( self ):
		super().init()

	def SetComponents( self ):
		super().SetComponents()

	def ExponentialMap( self ):
		super().ExponentialMap()

	def LogMap( self ):
		super().LogMap()

class euclidean_tVec( object ):
	def __init__( self ):
		self.Type = "Euclidean_Tangent"
		self.nDim = 3
		self.tVector = [ 0, 0, 0 ]

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector(self, tVec):
		self.tVector = tVec

class euclidean_atom( atom ):
	def __init__( self ):
		self.Type = "Euclidean"
		self.nComp = 1 
		self.nComp_dim = [ 3 ]
		self.nTangentComp_dim = [ 3 ]
		self.pt = [ 0.0, 0.0, 1.0 ] # Base point in S^2

	def SetComponents( self, pt=[ 0.0, 0.0, 0.0 ] ):
		self.pt = pt

	def SetPoint( self, pt = [ 0.0, 0.0, 0.0 ] ):
		self.pt = pt

	def GetPoint( self ):
		return self.pt

	def InnerProduct( self, pt1, pt2 ):
		if pt1.Type == "Euclidean":
			result = pt1.pt[ 0 ] * pt2.pt[0] + pt1.pt[1] * pt2.pt[1] + pt1.pt[2] * pt2.pt[2]  
			return result
		elif pt1.Type == "Euclidean_Tangent":
			result = pt1.tVector[ 0 ] * pt2.tVector[ 0 ] + pt1.tVector[ 1 ] * pt2.tVector[ 1 ] + pt1.tVector[ 2 ] * pt2.tVector[ 2 ] 
			return result

	def normSquared( self, pt1 ):
		return self.InnerProduct( pt1, pt1 )

	def norm( self, pt1 ):
		return np.sqrt( self.normSquared( pt1 ) )		

	def ExponentialMap( self, tVec ):
		exp_pt = euclidean_atom()
		exp_pt.SetPoint( [ self.pt[ 0 ] + tVec.tVector[0], self.pt[1] + tVec.tVector[1], self.pt[2] + tVec.tVector[2] ] )
		return exp_pt

	def LogMap( self, another_pt ):
		tVec = euclidean_tVec()
		tVec.SetTangentVector( [ another_pt.pt[0] - self.pt[0], another_pt.pt[1] - self.pt[1], another_pt.pt[2] - self.pt[2] ] ) 
		return tVec

	def ProjectTangent( self, pt, tVec ):
		vProjected = tVec
		return vProjected

	def ParallelTranslate( self, v, w ):
		return w

	def ParallelTranslateAtoB( self, a, b, w ):
		return w

	def AdjointGradientJacobi( self, v, j, dj, ):
		# Function Output
		jOutput = j
		jOutputDash = dj
		jOutputDash.SetTangentVector( [ j.tVector[0] + dj.tVector[0], j.tVector[1] + dj.tVector[1], j.tVector[2] + dj.tVector[2] ] ) 
		return jOutput, jOutputDash

class pos_real_tVec( object ):
	def __init__( self ):
		self.Type = "PositiveReal_Tangent"
		self.nDim = 1
		self.tVector = 1

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector(self, tVec):
		self.tVector = tVec


class pos_real_atom( atom ):
	def __init__( self ):
		self.Type = "PositiveReal"
		self.nComp = 1 
		self.nComp_dim = [ 1 ]
		self.nTangentComp_dim = [ 1 ]
		self.pt = 1.0 # Base point in R+

	def SetComponents( self, pt=1 ):
		self.pt = pt

	def SetPoint( self, pt = 1 ):
		self.pt = pt

	def GetPoint( self ):
		return self.pt

	def InnerProduct( self, pt1, pt2 ):
		if pt1.Type == "PositiveReal":
			result = pt1.pt * pt2.pt
			return result
		elif pt1.Type == "PositiveReal_Tangent":
			result = pt1.tVector * pt2.tVector
			return result

	def normSquared( self, pt1 ):
		return self.InnerProduct( pt1, pt1 )

	def norm( self, pt1 ):
		return np.sqrt( self.normSquared( pt1 ) )		

	def ExponentialMap( self, tVec ):
		exp_pt = pos_real_atom()
		exp_pt.SetPoint( self.pt * np.exp( tVec.tVector ) )
		return exp_pt

	def LogMap( self, another_pt ):
		tVec = pos_real_tVec()
		tVec.SetTangentVector( np.log( another_pt.pt / self.pt ) ) 
		return tVec

	def ProjectTangent( self, pt, tVec ):
		vProjected = tVec
		return vProjected

	def ParallelTranslate( self, v, w ):
		return w

	def ParallelTranslateAtoB( self, a, b, w ):
		return w

	def AdjointGradientJacobi( self, v, j, dj, ):
		# Function Output
		jOutput = j
		jOutputDash = dj
		jOutputDash.SetTangentVector( j.tVector + dj.tVector ) 
		return jOutput, jOutputDash

class sphere_tVec( object ):
	def __init__( self ):
		self.Type = "Sphere_Tangent"
		self.nDim = 3
		self.tVector = [ 0, 0, 0 ]

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector(self, tVec):
		self.tVector = tVec

class sphere_atom( atom ):
	def __init__( self ):
		self.Type = "Sphere"
		self.nComp = 1 # [ position, radius, angle1, angle2 ] : R^3 x R^+ x S^2 x S^2 
		self.nComp_dim = [ 3 ]
		self.nTangentComp_dim = [ 3 ]
		self.sphere_pt = [ 0.0, 0.0, 1.0 ] # Base point in S^2

	def SetComponents( self, sphere_pt=[ 0.0, 0.0, 1.0 ] ):
		self.sphere_pt = sphere_pt

	def SetSpherePt( self, sphere_pt = [ 0.0, 0.0, 1.0 ] ):
		self.sphere_pt = sphere_pt

	def GetSpherePt( self ):
		return self.sphere_pt

	def InnerProduct( self, pt1, pt2 ):
		if pt1.Type == "Sphere_Tangent":
			result = pt1.tVector[ 0 ] * pt2.tVector[ 0 ] + pt1.tVector[ 1 ] * pt2.tVector[ 1 ] + pt1.tVector[ 2 ] * pt2.tVector[ 2 ] 
			return result
		elif pt1.Type == "Sphere":
			result = pt1.sphere_pt[ 0 ] * pt2.sphere_pt[0] + pt1.sphere_pt[1] * pt2.sphere_pt[1] + pt1.sphere_pt[2] * pt2.sphere_pt[2]  
			return result

	def ProjectTangent( self, pt, tVec ):
		inner_prod_pt_tVec = pt.sphere_pt[0] * tVec.tVector[0] + pt.sphere_pt[1] * tVec.tVector[1] +  pt.sphere_pt[2] * tVec.tVector[2]  
		vProjected = sphere_tVec()
		vProjected.tVector[0] = tVec.tVector[0] - inner_prod_pt_tVec * pt.sphere_pt[0] 
		vProjected.tVector[1] = tVec.tVector[1] - inner_prod_pt_tVec * pt.sphere_pt[1] 
		vProjected.tVector[2] = tVec.tVector[2] - inner_prod_pt_tVec * pt.sphere_pt[2] 
		return vProjected

	def normSquared( self, pt1 ):
		return self.InnerProduct( pt1, pt1 )

	def norm( self, pt1 ):
		return np.sqrt( self.normSquared( pt1 ) )		

	def ParallelTranslate( self, v, w ):
		vNorm = self.norm( v )

		if( vNorm < 1.0e-12 ):
			return w

		innerProd = self.InnerProduct( v, w )
		scaleFactor = innerProd / ( vNorm * vNorm )

		# Component of w orthogonal to v
		orth = sphere_tVec()
		orth.SetTangentVector( [ w.tVector[0] - v.tVector[0] * scaleFactor, w.tVector[1] - v.tVector[1] * scaleFactor, w.tVector[2] - v.tVector[2] * scaleFactor ] )

		# Compute parallel translated v
		vParallel = sphere_tVec()
		vParallel.SetTangentVector( [ self.sphere_pt[ 0 ]* ( -np.sin( vNorm ) * vNorm ) + v.tVector[ 0 ] * np.cos( vNorm ), self.sphere_pt[1] * ( -np.sin( vNorm ) * vNorm ) + v.tVector[1] * np.cos( vNorm ), self.sphere_pt[2] * ( -np.sin( vNorm ) * vNorm ) + v.tVector[2] * np.cos( vNorm ) ] )

		wParallelTranslated = sphere_tVec()
		wParallelTranslated.SetTangentVector( [ vParallel.tVector[0] * scaleFactor + orth.tVector[0], vParallel.tVector[1] * scaleFactor + orth.tVector[1], vParallel.tVector[2] * scaleFactor + orth.tVector[2] ] )

		return wParallelTranslated

	def AdjointGradientJacobi( self, v, j, dj, ):
		e_base = self

		vNorm = e_base.norm( v ) 

		if vNorm < 1.0e-12:
			dj.SetTangentVector( [ j.tVector[ 0 ] + dj.tVector[ 0 ], j.tVector[ 1 ] + dj.tVector[ 1 ], j.tVector[ 2 ] + dj.tVector[ 2 ] ] )

			# Function Output
			jOutput = j
			jOutputDash = dj
		else:
			innerProdVJ = e_base.InnerProduct( v, j )
			innerProdVJPrime = e_base.InnerProduct( v, dj )

			scaleFactorJ = innerProdVJ / ( vNorm * vNorm )
			scaleFactorJPrime = innerProdVJPrime / ( vNorm * vNorm )

			jTang = sphere_tVec()
			jTang.SetTangentVector( [ v.tVector[0] * scaleFactorJ, v.tVector[1] * scaleFactorJ, v.tVector[2] * scaleFactorJ ] )

			djTang = sphere_tVec()
			djTang.SetTangentVector( [ v.tVector[0] * scaleFactorJPrime, v.tVector[1] * scaleFactorJPrime, v.tVector[2] * scaleFactorJPrime ] )

			jOrth = sphere_tVec()
			jOrth.SetTangentVector( [ j.tVector[0] - jTang.tVector[0], j.tVector[1] - jTang.tVector[1], j.tVector[2] - jTang.tVector[2] ] )

			djOrth = sphere_tVec()
			djOrth.SetTangentVector( [ dj.tVector[0] - djTang.tVector[0], dj.tVector[1] - djTang.tVector[1], dj.tVector[2] - djTang.tVector[2] ] )

			j.SetTangentVector( [ jTang.tVector[0] + ( np.cos( vNorm ) * jOrth.tVector[0] ) - ( ( vNorm * np.sin( vNorm ) ) * djOrth.tVector[0] ), jTang.tVector[1] + np.cos( vNorm ) * jOrth.tVector[1] - ( vNorm * np.sin( vNorm ) ) * djOrth.tVector[1], jTang.tVector[2] + np.cos( vNorm ) * jOrth.tVector[2] - ( vNorm * np.sin( vNorm ) ) * djOrth.tVector[2] ] ) 

			j = e_base.ParallelTranslate( v, j )

			dj.SetTangentVector( [ jTang.tVector[0] + djTang.tVector[0] + ( np.sin(vNorm) / vNorm ) * jOrth.tVector[ 0 ] + np.cos( vNorm ) * djOrth.tVector[ 0 ], jTang.tVector[1] + djTang.tVector[1] + ( np.sin(vNorm) / vNorm ) * jOrth.tVector[1] + np.cos( vNorm ) * djOrth.tVector[1], jTang.tVector[2] + djTang.tVector[2] + ( np.sin(vNorm) / vNorm ) * jOrth.tVector[2] + np.cos( vNorm ) * djOrth.tVector[2] ] ) 

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
			exp_pt = sphere_atom()
			exp_pt.sphere_pt = self.sphere_pt			
			return exp_pt

		exp_pt = sphere_atom()
		lhs = np.multiply( np.cos( theta ), self.sphere_pt )
		rhs = np.multiply( np.sin( theta ) / theta, tVec.tVector )
		exp_pt.sphere_pt = [ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2] ]		
		exp_pt.sphere_pt = np.divide( exp_pt.sphere_pt, exp_pt.norm( exp_pt ) ) 
		return exp_pt

	def LogMap( self, another_pt ):
		cosTheta = self.InnerProduct( self, another_pt )
		tVec = sphere_tVec() 
		tVec.tVector = [ another_pt.sphere_pt[0] - cosTheta * self.sphere_pt[0], another_pt.sphere_pt[1] - cosTheta * self.sphere_pt[1], another_pt.sphere_pt[2] - cosTheta * self.sphere_pt[2] ]
		length = self.norm( tVec )

		if length < 1e-12 or cosTheta >= 1.0 or cosTheta <= -1.0:
			tVec = sphere_tVec()
			return tVec

		tVec.tVector = [ tVec.tVector[ 0 ] * np.arccos( cosTheta ) / length, tVec.tVector[ 1 ] * np.arccos( cosTheta ) / length, tVec.tVector[ 2 ] * np.arccos( cosTheta ) / length ]		
		return tVec

class mrep_tVec( object ):
	def __init__( self ):
		self.tVecType = "MRep_Tangent"
		self.nComp = 4 
		self.nComp_dim = [ 3, 1, 3, 3 ]
		self.nTangentComp_dim = [ 3, 1, 3, 3 ]
		self.tVector = [ [ 0.0, 0.0, 0.0 ], 0.0, sphere_tVec(), sphere_tVec() ]

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector( self, position=[ 0.0, 0.0, 0.0 ], rho=0.0, sphere_tVec1=sphere_tVec(), sphere_tVec2 =sphere_tVec() ):
		self.tVector[ 0 ] = position
		self.tVector[ 1 ] = rho
		self.tVector[ 2 ] = sphere_tVec1
		self.tVector[ 3 ] = sphere_tVec2

class mrep_atom( atom ):
	def __init__( self ):
		self.AtomType = "MRep"
		self.nComp = 4 # [ position, radius, angle1, angle2 ] : R^3 x R^+ x S^2 x S^2 
		self.nComp_dim = [ 3, 1, 3, 3 ]
		self.nTangentComp_dim = [ 3, 1, 3, 3 ]
		self.pos = [ 0.0, 0.0, 0.0 ] 
		self.rad = 1
		self.sphere_comp1 = sphere_atom() # Base point in S^2
		self.sphere_comp2 = sphere_atom() # Base point in S^2

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

class cmrep_tVec( object ):
	def __init__( self ):
		self.tVecType = "CMRep_Tangent"
		self.nComp = 2
		self.nComp_dim = [ 3, 1 ]
		self.nTangentComp_dim = [ 3, 1 ]
		self.tVector = [ [ 0.0, 0.0, 0.0 ], 0.0 ]

	def GetTangentVector(self):
		return self.tVector

	def SetTangentVector( self, position=[ 0.0, 0.0, 0.0 ], rho=0.0 ):
		self.tVector[ 0 ] = position
		self.tVector[ 1 ] = rho

class cmrep_atom( atom ):
	def __init__( self ):
		self.AtomType = "CMRep"
		self.nComp = 2 # [ Position, radius ] 
		self.nComp_dim = [ 3, 1 ]
		self.nTangentComp_dim = [ 3, 1 ]
		self.pos = [ 0, 0, 0 ]
		self.rad = 1

	def SetComponents( self, position=[0.0,0.0,0.0], rad=0.0 ):
		if not ( len( position ) == 3 and isinstance( rad, numbers.Number ) ):
			print( "component dimension mismatch" )
			return
		self.pos = position
		self.rad = rad

		print( self )

	def SetPosition( self, position=[0.0,0.0,0.0] ):
		self.pos = position 

	def SetRadius( self, rad=0.0 ):
		self.rad = rad

	def ExponentialMap( self, tVec ):
		if not tVec.tVecType == "CMRep_Tangent":
			print( "Tangent Vector Type Mismatched" )
			return

		exp_pt = cmrep_atom()
		exp_pt.pos = [ self.pos[0] + tVec.tVector[ 0 ][0], self.pos[1] + tVec.tVector[ 0 ][1], self.pos[2] + tVec.tVector[ 0 ][2] ] 
		exp_pt.rad = self.rad * np.exp( tVec.tVector[ 1 ] )
		return exp_pt

	def LogMap( self, another_pt ):
		if not another_pt.AtomType == "CMRep":
			print( "Error: Component Type Mismatched" )
			return

		tVec = cmrep_tVec()
		tVec.tVector[ 0 ] = [ another_pt.pos[0] - self.pos[0], another_pt.pos[1] - self.pos[1], another_pt.pos[2] - self.pos[2] ]
		tVec.tVector[ 1 ] = np.log( another_pt.rad / self.rad )
		return tVec		

	def GetPoistion( self ):
		return self.pos

	def GetRadius( self ):
		return self.rad

# class sphere2D_tVec( object ):
# 	def __init__( self ):
# 		self.Type = "Sphere2D_Tangent"
# 		self.nDim = 3
# 		self.tVector = [ 0, 0, 0 ]

# 	def GetTangentVector(self):
# 		return self.tVector

# 	def SetTangetVector(self, tVec):
# 		self.tVector = tVec

# class sphere2d_atom( atom ):
# 	def __init__( self ):
# 		self.AtomType = "Sphere2D"
# 		self.nComp = 1 # [ Position, radius ] 
# 		self.nComp_dim = [ 3 ]
# 		self.nTangentComp_dim = [ 3 ]
# 		self.pt = [ 1, 0, 0 ]

# 	def SetPoint( self, sphere_pt=[ 0, 0, 0 ] ):
# 		self.pt[ 0 ] = sphere_pt[ 0 ]
# 		self.pt[ 1 ] = sphere_pt[ 1 ]
# 		self.pt[ 2 ] = sphere_pt[ 2 ]

# 	def InnerProduct( self, pt1, pt2 ):
# 		if( pt1.AtomType == "Sphere2D" ):
# 			return ( pt1.pt[0] * pt2.pt[0] + pt1.pt[1] * pt2.pt[1] + pt1.pt[2] * pt2.pt[2] )
# 		else if( pt1.Type == "Sphere2D_Tangent" ):
# 			return ( pt1.tVector[0] * pt2.tVector[0] + pt1.tVector[1] * pt2.tVector[1] + pt1.tVector[2] * pt2.tVector[2] )
	
# 	def NormSquared( self, pt1 ):
# 		return( self.InnerProduct( pt1, pt1 ) )

# 	def Norm( self, pt1 ):
# 		return( np.sqrt( self.normSquared( pt1 ) ) )		

# 	def GetDistance( self, pt1, pt2 ):
# 		v = pt1.LogMap( pt2 )
# 		return( self.Norm( self, v ) )

# 	def GetDistanceSquared( self, pt1, pt2 ):
# 		v = pt1.LogMap( pt2 )
# 		return( self.NormSquared( self, v ) )

# 	def ExponentialMap( self, tVec ):
# 		