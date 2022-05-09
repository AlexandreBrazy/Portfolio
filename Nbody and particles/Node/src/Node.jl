module Node

export node, contains, intersect

mutable struct node

    # x,y,z center of the node
    x::Float64
    y::Float64
    z::Float64

    # h, w, d dimension of the node form the center ie h is total height / 2 etc...
    h::Float64 # height
    w::Float64 # width
    d::Float64 # depth

    # mass contains in the node
    m::Float64 

    # constructor to alleviate the writing/reading when initializing the node. mass is always 0 when creating a node
    # inner constructor to initialize m at 0 (without having to specify it)
    node(x,y,z, h,w,d) = new(x,y,z, h,w,d, 0)

end

function contains(node, part)
    # return true if a particle is in the node
    # particle is in the node if node.left_side < part.x < node.right_side etc...
    # && = and operator || = or operator

    ( ( part.x >= (node.x - node.w) ) && # part is on the right of the left border ; >= in case of the particle is rigth on the border
    ( part.x < (node.x + node.w) ) && # part in on the left of the right border
    ( part.y >= (node.y - node.h) ) && # part is up of the bottom border ; >= in case of the particle is rigth on the border
    ( part.y < (node.y + node.h) ) && # part in down of the top border
    ( part.z >= (node.z - node.d) ) && # part is on the right of the left border ; >= in case of the particle is rigth on the border
    ( part.z < (node.z + node.d) ) ) # part in on the left of the right border

end


function intersects(node, rect)
    # return true if rect intersect node
    # if the expression (after the not) is true, it doesnt intersect so we need a not in front of the expression"
    # check if 2 nodes intersect
    # use for neighbours search as we search the neighbours as all the particle
    # within a node center on the particle we looking at with a dimension of the search radius
    # here is the 'rectangular' search

    # node intersect rect if left side of node is at the rigth of the rigth side of rect or rigth side of node is at the left of left side of rect etc... 
    ! ( ( (rect.x + rect.w) < (node.x - node.w) ) || # rigth side of rect is on the left of left side of node
        ( (rect.x - rect.w) > (node.x + node.w) ) || # left side of rect is on the right of the right side of node
        ( (rect.y + rect.h) < (node.y - node.h) ) || #
        ( (rect.y - rect.h) > (node.y + node.h) ) || # 
        ( (rect.z + rect.d) < (node.z - node.d) ) || # 
        ( (rect.z - rect.d) > (node.z + node.d) ) ) # 

end


end # module