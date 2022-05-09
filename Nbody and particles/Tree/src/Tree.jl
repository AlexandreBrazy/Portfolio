module Tree

#using GLMakie
#include("Node.jl")
using Node

export tree, insert!, subdivide!, query, query_acc!, brute

mutable struct tree
    
    capacity::Int64
    node::node
    point::Array # point currently in node
    points::Array # list all the points in the node including in children
    divided::Bool

    # for barnes-hut
    m::Float64
    center_of_mass_x::Float64
    center_of_mass_y::Float64
    center_of_mass_z::Float64

    # children
    unw::Union{tree, Nothing}
    une::Union{tree, Nothing}
    usw::Union{tree, Nothing}
    use::Union{tree, Nothing}

    dnw::Union{tree, Nothing}
    dne::Union{tree, Nothing}
    dsw::Union{tree, Nothing}
    dse::Union{tree, Nothing}

    # inner constructor for incomplete Initialization bc of recursion

    # see https://docs.julialang.org/en/v1/manual/constructors/#Incomplete-Initialization-1
    # as of dec 2021 my understanding is that new allow to call the type with less field than
    # what its specified in the type definition
    # here it allow to not specified all the children when creating the quadtree but later if needed
    tree(capacity, node) = new(capacity, node, [], [], false, 0,0,0,0)

end

function insert!(part, _tree)

    # if particle is not in the current node return, no need to check the reset
    if ! Node.contains(_tree.node, part)
        return
    
    else
        # add the particle to the list of all part in the node
        push!(_tree.points, part)


        # update mass/center of mass of the octrant !

        _tree.center_of_mass_x = (_tree.m * _tree.center_of_mass_x + part.m * part.x) # center of mass of the node
        _tree.center_of_mass_y = (_tree.m * _tree.center_of_mass_y + part.m * part.y) # center of mass of the node
        _tree.center_of_mass_z = (_tree.m * _tree.center_of_mass_z + part.m * part.z) # center of mass of the node

        _tree.m = _tree.m + part.m # new total mass in the node

        _tree.center_of_mass_x = _tree.center_of_mass_x / _tree.m # center of mass of the node
        _tree.center_of_mass_y = _tree.center_of_mass_y / _tree.m # center of mass of the node
        _tree.center_of_mass_z = _tree.center_of_mass_z / _tree.m # center of mass of the node

        # if the capacity is not full and node is not divided add the point
        # without the divided condition, we could readd a point on a node that has been previously divided, as the capacity is clear when dividing it
        if (length(_tree.point) < _tree.capacity) && (! _tree.divided)  
            push!(_tree.point, part)



        else # if the capacity is full or its divided
            if ! _tree.divided # and not already divided,
                subdivide!(_tree) # divide !
            end

            #add to the child recursively
            insert!(part, _tree.une)
            insert!(part, _tree.unw)
            insert!(part, _tree.use)
            insert!(part, _tree.usw)

            insert!(part, _tree.dne)
            insert!(part, _tree.dnw)
            insert!(part, _tree.dse)
            insert!(part, _tree.dsw)

        end

    end
end

function subdivide!(_tree)
    
    x = _tree.node.x
    y = _tree.node.y
    z = _tree.node.z
    h = _tree.node.h
    w = _tree.node.w
    d = _tree.node.d

    # une = up north east and so on, dsw = down south west

    une = node(x + w/2, y + h/2, z + d/2, w/2, h/2, d/2)  # create a new Node/rectangle
    _tree.une = tree(_tree.capacity, une) # create a quad_tree from the Node created

    unw = node(x - w/2, y + h/2, z + d/2, w/2, h/2, d/2)
    _tree.unw = tree(_tree.capacity, unw)

    use = node(x + w/2, y - h/2, z + d/2, w/2, h/2, d/2)
    _tree.use = tree(_tree.capacity, use)

    usw = node(x - w/2, y - h/2, z + d/2, w/2, h/2, d/2)
    _tree.usw = tree(_tree.capacity, usw)

    dne = node(x + w/2, y + h/2, z - d/2, w/2, h/2, d/2)  # create a new Node/rectangle
    _tree.dne = tree(_tree.capacity, dne) # create a quad_tree from the Node created

    dnw = node(x - w/2, y + h/2, z - d/2, w/2, h/2, d/2)
    _tree.dnw = tree(_tree.capacity, dnw)

    dse = node(x + w/2, y - h/2, z - d/2, w/2, h/2, d/2)
    _tree.dse = tree(_tree.capacity, dse)

    dsw = node(x - w/2, y - h/2, z - d/2, w/2, h/2, d/2)
    _tree.dsw = tree(_tree.capacity, dsw)

    _tree.divided = true

    for part in _tree.point
        insert!(part, _tree.une)
        insert!(part, _tree.unw)
        insert!(part, _tree.use)
        insert!(part, _tree.usw)

        insert!(part, _tree.dne)
        insert!(part, _tree.dnw)
        insert!(part, _tree.dse)
        insert!(part, _tree.dsw)

    end

    _tree.point = []#[] # when subdividing we move the point in the new node, so a clear the previous list
end

function show(_tree,ax)

    #f = Figure()
    #ax = Axis3(f[1, 1])
    
    # vector of shapes
    x = _tree.node.x - _tree.node.w
    y = _tree.node.y - _tree.node.h
    poly!( ax,  Rect(x,y, 2*_tree.node.w, 2*_tree.node.h), color=:transparent   )
    # draw the quad
    # rectangle take the bottom left point coordinate and the full height and width
    #axes.add_patch( mplp.Rectangle((self.node.x - self.node.w, self.node.y - self.node.h), self.node.w*2, self.node.h*2, fill = False, ec = 'k') )
    if _tree.divided
        _tree.ne.show(_tree, ax)
        _tree.nw.show(_tree, ax)
        _tree.se.show(_tree, ax)
        _tree.sw.show(_tree, ax)
    end

    for p in tree.point
        scatter!(ax,p.x, p.y)
    end

end


function query(_tree, test_node, result ) # node : center and width and height of the search rectangle or position of a point and search radius
    #global count

    #="""
    simple query return the list of point that is in range of the test node

    """ =#

    if ! Node.intersects(_tree.node, test_node)
        return result # if the test quadrant doesnt intersect the test node (rectangle) then return

    else
        if ! _tree.divided # if hte current quadrant is not divided ie bottom node
            for p in _tree.point
                if Node.contains(test_node, p) # and the points is in the test range
                    push!(result, p) # then append
                    #count += 1
                end
            end

        else # else look at its children recursively
            query(_tree.une, test_node, result)
            query(_tree.unw, test_node, result)
            query(_tree.use, test_node, result)
            query(_tree.usw, test_node, result)

            query(_tree.dne, test_node, result)
            query(_tree.dnw, test_node, result)
            query(_tree.dse, test_node, result)
            query(_tree.dsw, test_node, result)

            return result
        end
    end
    return result
end

function query_acc!(quadtree, test_point, theta = 0.8, softening = 1, G = 6.67e-11) # node : center and width and height of the search rectangle or position of a point and search radius
    #=
    query return the acceleration that the test node is subjected to
    basically if close it is computed as particle-particle interacction
    if far enough its an approximation between the center of mass of several particle and the current particle

    and dont forget plummer's softening !
    =#

    # if node is empty ie m ==0, return
    # checking the mass is faster than getting the length of the stored point ?

    if quadtree.m == 0
        return 
    end

    # cant use the mass a a node can have mass but no point, bc point are in bottom node
    if length(quadtree.point) != 0  #if there one point in the octrant
        if quadtree.point[1] == test_point # and if the point is the current one, only valid when capacity ==1
            # print('quadtree octrant')
            return    # we are looking at the current particle
        end
    end


    #comptute dist
    dx = quadtree.center_of_mass_x - test_point.x
    dy = quadtree.center_of_mass_y - test_point.y
    dz = quadtree.center_of_mass_z - test_point.z

    dist = sqrt(dx*dx + dy*dy + dz*dz)
    
    if dist<0
        print("error")
    end


    if (quadtree.node.w/dist < theta ) || !(quadtree.divided)

        #="""
        if not divided ie bootom node or
        if s/d < theta get acc using mass and center of mass of the current node

        """=#
        # print('something is here', quadtree.m)
        test_point.acc_x += G * quadtree.m / (dist + softening)^(3) * dx
        test_point.acc_y += G * quadtree.m / (dist + softening)^(3) * dy
        test_point.acc_z += G * quadtree.m / (dist + softening)^(3) * dz


        # print(acc_x,acc_y)
        return 

    else
        #= """
        important note:
            we need to affect the result (acc_x, ... = quadtree.une.queru...) before passing it to the next call
            if not it will return 0
            not entirely sure why

        """=#
        query_acc!(quadtree.une, test_point, theta, softening)
        query_acc!(quadtree.unw, test_point, theta, softening)
        query_acc!(quadtree.use, test_point, theta, softening)
        query_acc!(quadtree.usw, test_point, theta, softening)

        query_acc!(quadtree.dne, test_point, theta, softening)
        query_acc!(quadtree.dnw, test_point, theta, softening)
        query_acc!(quadtree.dse, test_point, theta, softening)
        query_acc!(quadtree.dsw, test_point, theta, softening)

        #return acc_x, acc_y, acc_z # if the test quadrant doesnt intersect the rectangle then return

    end
    return 
end


function brute(list_part, p, G=6.67e-11)
        for part in list_part
        if p!=part
            dx = part.x - p.x
            dy = part.y - p.y
            dz = part.z - p.z
        
            dist = sqrt(dx*dx + dy*dy + dz*dz)
            println(dist)
        
            p.acc_x += G * part.m/((dist+0)^3) * dx
            p.acc_y += G * part.m/((dist+0)^3) * dy
            p.acc_z += G * part.m/((dist+0)^3) * dz

        end
    end

end

end # module