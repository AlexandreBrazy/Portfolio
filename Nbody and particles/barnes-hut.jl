using GLMakie
using Node
using Tree
include("Kernel.jl")
using .Kernel


mutable struct particle

    h::Float64 # kernel smoothing length
    x::Float64
    y::Float64
    z::Float64
    u::Float64
    v::Float64
    w::Float64
    m::Float64
    acc_x::Float64
    acc_y::Float64
    acc_z::Float64
    P::Float64 # pressure
    density::Float64
    #count::Float64
    #def __init__(self, h, pos, vel = [0, 0], m = 1, P = 0, density = 0, acc = [0, 0]):
    # constructor
end

function kin_en(list_part)
    ek=0
    @simd for p in list_part
        ek+= .5*p.m*(p.u*p.u + p.v+p.v + p.w+p.w)
    end
    return ek
end

function pot_en(list_part)
    ep = 0
    @simd for p in list_part
        for part in list_part
            if p!=part
                dist = sqrt((part.x-p.x)^2 + (part.y-p.y)^2 + (part.z-p.z)^2)
                ep += -G*p.m * part.m /dist
            end
        end
    end
    return ep
end

function calc_dens_press!(part::particle, searchtree::tree)
    part.density = 0.
    part.P = 0.

    research_zone = node(part.x, part.y, part.z, part.h, part.h, part.h)
    jnk = []

    result = query(searchtree, research_zone, jnk ) # node : center and width and height of the search rectangle or position of a point and search radius

    for p in result #list_part
        part.density +=  p.m * Kernel.kernel(part, p)
    end
        #" polytropic fluid ie PV^gamma = cste or P = k* density^gamma"
    #part.P = k * ( part.density ^4) # 1.8

    rho_0 = 1000 
    cs = .07
    gamma = 7
    k = 5
    #part.P = rho_0*cs*cs/gamma*((part.density/rho_0)^gamma-1)
    #part.P = k*(part.density - rho_0)
    # ideal gas law at 10K
    part.P = part.density * 8.314 / 1. * 10

end

function get_acc!(part::particle, searchtree::tree)
  # sph acc
    
    research_zone = node(part.x, part.y, part.z, part.h, part.h, part.h)
    jnk = []

    result = query(searchtree, research_zone, jnk ) # node : center and width and height of the search rectangle or position of a point and search radius

    for p in result    #(j,p) in enumerate(list_part)
        if p != part
            r_norm = sqrt((p.x-part.x)^2 + (p.y-part.y)^2 + (p.z-part.z)^2)

            cst = (part.P/part.density^2 + p.P/p.density^2) * Kernel.grad_kernel(part, p)
            part.acc_x += - p.m * cst*(p.x - part.x)/r_norm
            part.acc_y += - p.m * cst*(p.y - part.y)/r_norm
            part.acc_z += - p.m * cst*(p.z - part.z)/r_norm
        end
    end
end

function update_tree!(_tree, list_part)
    
    _tree.divided = false
    _tree.m = 0
    _tree.center_of_mass_x = 0
    _tree.center_of_mass_y = 0
    _tree.center_of_mass_z = 0
    _tree.une = nothing
    _tree.unw = nothing
    _tree.use = nothing
    _tree.usw = nothing

    _tree.dne = nothing
    _tree.dnw = nothing
    _tree.dse = nothing
    _tree.dsw = nothing

    _tree.points = []
    _tree.point = []

    for p in list_part
        Tree.insert!(p,_tree)
    end

end

function verlet!(list_part)
    """
    1- kick/drift verlet
    2- get acc
    3- kick verlet
    4- recompute the tree

    """
    tmp = list_part

    for part in list_part
        #"velocity verlet / leapfrog kick drift "

        part.u += part.acc_x * dt/2
        part.v += part.acc_y * dt/2
        part.w += part.acc_z * dt/2
    
        part.x += part.u * dt
        part.y += part.v * dt
        part.z += part.w * dt

        # reset acc
        part.acc_x = 0
        part.acc_y = 0
        part.acc_z = 0
    end

    _tree = tree(1, _range)
    for p in tmp
      Tree.insert!(p, _tree)
    end

    # gravity
    Threads.@threads for part in list_part
        query_acc!(_tree, part, 0.5, soft) #softening about 1 ly, theta, soft
        calc_dens_press!(part, _tree)
    end

    # pressure ; euler eq
    Threads.@threads for p in  list_part     
        get_acc!(p, _tree)
    end

    for (i, part) in enumerate(list_part)

        #"velocity verlet / leapfrog kick"
    
        part.u += part.acc_x * dt/2
        part.v += part.acc_y * dt/2
        part.w += part.acc_z * dt/2
        points[i,1], points[i,2], points[i,3] = part.x, part.y, part.z

    end
    #push!(en, kin_en(_tree.points)+pot_en(_tree.points))
    return _tree
end


const G = 6.67e-11
const mw_radius = 50_000e16
const m_sol = 2e30
const m_cloud = 1e12 *m_sol#50. * m_sol
const dt = 86400. * 365.25 * 200e3 # 25_000
const sim_range = 5 * mw_radius # light year in meter 1e12 solar system 2.5e11 2.5e11# 
const plot_range = 2.5 * mw_radius # light year in meter 1e12 solar system 2.5e11 2.5e11# 
const n_part = 25000
const n_iter = 50
const soft = 1000e16

points = zeros(n_part, 3)

_range = node(0, 0, 0, sim_range, sim_range, sim_range)
octree = tree(1, _range)
#" 1 - generate random particule "
list_part=Vector{particle}(undef,n_part)


for i in 1:n_part
    
    # sphere with no init vel
    # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
    #=
    lambda = rand()
    u = 2. * rand() - 1. # full circle
    phi = 2. * pi * rand() # +/- elevation

    x = sim_range * lambda^(1/3) * sqrt(1. - u*u) * cos(phi)
    y = sim_range * lambda^(1/3) * sqrt(1. - u*u) * sin(phi)
    z = sim_range * lambda^(1/3) * u

    u, v, w = 0, 0, 0
    =#

     
    # rand = unifrom distribution in range (0,1)
    # so to get in range (a,b) => a + (b-a) * rand(npart)
    #radius = mw_radius # milky way radius
    r = (0.01 + (1-0.01)* rand())*mw_radius#*radius
    
    th = rand() *2. * pi
    
    x = r * cos(th)
    
    y = r * sin(th)

    z = -2000e16 + 4000e16 * rand()
    
    jnk = (r/(24_000e16)) ^(3/2) * 250e6*365.25*86400. #86400. * 365.25 * 300e6 #time for one rotation # de l'ordre de 2.5e8 pour le soleil
    dth = 2. * pi / jnk #jnk # rad/s

    #lettre sqrt((center mass+m)/r**3)*vel mult
    dens = m_cloud/(pi*mw_radius^2)
    m = dens * pi * r^2
    α = sqrt(G*(m)/((r)))#*(1e-12*r+1e9)
    u = - α *sin(th)# r * dth * -sin(th)    
    v = α *cos(th) #r * dth * cos(th)

    """
        r=exp10.(range(20, stop=21, length=50)) # log range, renvoie les puissances de 10
        u = - dth *sin(th) .* r # r * dth * sin(th)
        
        v = dth *cos(th) .* r #r * dth * cos(th)
        jnk = sqrt.(u .* u .+ v .* v)
        scatterlines!(r,jnk)
    """ 

    z = 0
    w = 0

    h = 10e16 # smoothing length = softening = 10e16 = 10 ly

    p = particle(soft,x,y,z,u,v,w,m_cloud/n_part,0,0,0,0,0)
    list_part[i] = p
    points[i, :] = [x,y,z]
end

ox = Observable(points[:,1])
oy = Observable(points[:,2])
oz = Observable(points[:,3])

fig = Figure()
ax1 = Axis3(fig[1, 1], elevation = .5 *pi, azimuth = 1. *pi)
ax1.title = "BH"
display(fig)


meshscatter!(ox,oy,oz, markersize = 4e18)#sim_range/100)#9e18) # 1% of sim_range seems fine 1e19, 3e18 for 100k, 25k 4e18

xlims!(ax1,-plot_range,plot_range)
ylims!(ax1,-plot_range,plot_range)
zlims!(ax1,-plot_range,plot_range)


step = 5.
frames = 1:trunc(n_iter/step)

en = Float64[]
enb = Float64[]

record(fig,"barnes hut pressure.mp4",frames; framerate=30) do i # i = frame
    for j in 1:trunc(step) # do 2 steps frames #
        
        verlet!(list_part)
        
    ox[]= points[:,1]
    oy[]= points[:,2]
    oz[]= points[:,3]
    
end

end
