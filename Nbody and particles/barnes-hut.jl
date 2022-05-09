mutable struct point

  x
  y
  z

  u
  v
  w

  acc_x
  acc_y
  acc_z

  m

end



p = point(0,0,0,0,0,0,0,0,0,1)
p1 = point(1,1,1,0,0,0,0,0,0,1)

println(p.m)

#=
! = not
&& = and
|| = or
=#
