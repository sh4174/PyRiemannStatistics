import manifolds

# Euclidean 
eucl_tVec = manifolds.euclidean_tVec( 3 )
eucl_tVec.SetTangentVector( [ 1, 3, 1 ] ) 

eucl_tVec.Write( "eucl_tVec.tvec" )

eucl_tVec_r = manifolds.euclidean_tVec( 3 )
eucl_tVec_r.Read( "eucl_tVec.tvec" )

print( eucl_tVec_r.tVector )
print( eucl_tVec_r.nDim )
print( eucl_tVec_r.Type )

eucl_pt = manifolds.euclidean( 3 )
eucl_pt.SetPoint( [ 3, 1, 5 ] )
eucl_pt.Write( "eucl.rpt" )

eucl_pt_r = manifolds.euclidean( 3 )
eucl_pt_r.Read( "eucl.rpt" )

print( eucl_pt_r.pt )
print( eucl_pt_r.nDim ) 
print( eucl_pt_r.Type )

# Sphere
sphere_tVec = manifolds.sphere_tVec( 3 )
sphere_tVec.SetTangentVector( [ 0, 0.8192, 0 ] ) 

sphere_tVec.Write( "sphere_tVec.tvec" )

sphere_tVec_r = manifolds.sphere_tVec( 3 )
sphere_tVec_r.Read( "sphere_tVec.tvec" )

print( sphere_tVec_r.tVector )
print( sphere_tVec_r.nDim )
print( sphere_tVec_r.Type )

sphere_pt = manifolds.sphere( 3 )
sphere_pt.SetPoint( [ 0.0, 1.0, 0.0 ] )
sphere_pt.Write( "sphere.rpt" )

sphere_pt_r = manifolds.sphere( 3 )
sphere_pt_r.Read( "sphere.rpt" )

print( sphere_pt_r.pt )
print( sphere_pt_r.nDim ) 
print( sphere_pt_r.Type )

# Pos Real 
pos_real_tVec = manifolds.pos_real_tVec( 1 )
pos_real_tVec.SetTangentVector( 2 ) 

pos_real_tVec.Write( "pos_real_tVec.tvec" )

pos_real_tVec_r = manifolds.pos_real_tVec( 1 )
pos_real_tVec_r.Read( "pos_real_tVec.tvec" )

print( pos_real_tVec_r.tVector )
print( pos_real_tVec_r.nDim )
print( pos_real_tVec_r.Type )

pos_real_pt = manifolds.pos_real( 1 )
pos_real_pt.SetPoint( 4.0 )
pos_real_pt.Write( "pos_real.rpt" )

pos_real_pt_r = manifolds.pos_real( 1 )
pos_real_pt_r.Read( "pos_real.rpt" )

print( pos_real_pt_r.pt )
print( pos_real_pt_r.nDim ) 
print( pos_real_pt_r.Type )

# CMRep
cmrep_tVec = manifolds.cmrep_tVec( 2 )
cmrep_tVec.SetTangentVector( [ [ eucl_tVec, pos_real_tVec ], [ eucl_tVec, pos_real_tVec ] ] ) 

cmrep_pt = manifolds.cmrep( 2 )
cmrep_pt.SetPoint( [ [ eucl_pt, pos_real_pt ], [ eucl_pt, pos_real_pt ] ] )
cmrep_pt.UpdateMeanRadius()

cmrep_tVec.SetMeanRadius( cmrep_pt.meanRadius )


cmrep_tVec.Write( "cmrep_tVec.tvec" )
cmrep_pt.Write( "cmrep.rpt" )

cmrep_tVec_r = manifolds.cmrep_tVec( 2 )
cmrep_tVec_r.Read( "cmrep_tVec.tvec" )

cmrep_pt_r = manifolds.cmrep( 2 )
cmrep_pt_r.Read( "cmrep.rpt" )

print( cmrep_tVec_r.tVector[ 0 ][ 0 ].tVector )
print( cmrep_tVec_r.nDim )
print( cmrep_tVec_r.meanRadius )
print( cmrep_tVec_r.Type )

print( cmrep_pt_r.pt[ 0 ][ 0 ].pt )
print( cmrep_pt_r.nDim ) 
print( cmrep_pt_r.meanRadius )
print( cmrep_pt_r.Type )
