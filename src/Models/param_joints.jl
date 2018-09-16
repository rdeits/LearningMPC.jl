const rbd = RigidBodyDynamics

struct ParameterJoint{T} <: RigidBodyDynamics.JointType{T}
end

Base.show(io::IO, jt::ParameterJoint) = print(io, "ParameterJoint joint")
Base.rand(::Type{ParameterJoint{T}}) where {T} = ParameterJoint{T}(randn())
rbd.flip_direction(jt::ParameterJoint) = deepcopy(jt)

rbd.num_positions(::Type{<:ParameterJoint}) = 1
rbd.num_velocities(::Type{<:ParameterJoint}) = 0
rbd.has_fixed_subspaces(jt::ParameterJoint) = true
rbd.isfloating(::Type{<:ParameterJoint}) = false

function rbd.joint_transform(jt::ParameterJoint{T}, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector{X}) where {T, X}
    S = promote_type(T, X)
    eye(Transform3D{S}, frame_after, frame_before)
end

function rbd.joint_twist(jt::ParameterJoint{T}, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector{X}, v::AbstractVector{X}) where {T, X}
    S = promote_type(T, X)
    zero(Twist{S}, frame_after, frame_before, frame_after)
end

function rbd.joint_spatial_acceleration(jt::ParameterJoint{T}, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector{X}, v::AbstractVector{X}, vd::AbstractVector{XD}) where {T, X, XD}
    S = promote_type(T, X, XD)
    zero(SpatialAcceleration{S}, frame_after, frame_before, frame_after)
end

function rbd.motion_subspace(jt::ParameterJoint{T}, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector{X}) where {T, X}
    S = promote_type(T, X)
    GeometricJacobian(frame_after, frame_before, frame_after, zeros(SMatrix{3, 0, S}), zeros(SMatrix{3, 0, S}))
end

function rbd.constraint_wrench_subspace(jt::ParameterJoint{T}, joint_transform::Transform3D{X}) where {T, X}
    S = promote_type(T, X)
    angular = hcat(eye(SMatrix{3, 3, S}), zeros(SMatrix{3, 3, S}))
    linear = hcat(zeros(SMatrix{3, 3, S}), eye(SMatrix{3, 3, S}))
    WrenchMatrix(joint_transform.from, angular, linear)
end

function rbd.zero_configuration!(q::AbstractVector, ::ParameterJoint)
    q .= 0
end
function rbd.rand_configuration!(q::AbstractVector, ::ParameterJoint)
    q .= randn()
end

function rbd.bias_acceleration(jt::ParameterJoint{T}, frame_after::CartesianFrame3D, frame_before::CartesianFrame3D,
        q::AbstractVector{X}, v::AbstractVector{X}) where {T, X}
    S = promote_type(T, X)
    zero(SpatialAcceleration{S}, frame_after, frame_before, frame_after)
end

rbd.configuration_derivative_to_velocity!(v::AbstractVector, ::ParameterJoint, q::AbstractVector, q̇::AbstractVector) = nothing
rbd.velocity_to_configuration_derivative!(q̇::AbstractVector, ::ParameterJoint, q::AbstractVector, v::AbstractVector) = nothing
rbd.joint_torque!(τ::AbstractVector, jt::ParameterJoint, q::AbstractVector, joint_wrench::Wrench) = nothing

function rbd.velocity_to_configuration_derivative_jacobian(::ParameterJoint{T}, ::AbstractVector) where T
    SMatrix{1, 0, T}()
end

function rbd.configuration_derivative_to_velocity_jacobian(::ParameterJoint{T}, ::AbstractVector) where T
    SMatrix{0, 1, T}()
end