using IKMimic: IKMimicWorkspace, ik_mimic!
using QPControl
using RigidBodyDynamics
using RigidBodyDynamics.Graphs: target, source, TreePath
using RigidBodyDynamics.PDControl

struct MappedController{C1 <: QPControl.MPCController, C2 <: MomentumBasedController}
    reduced_model_controller::C1
    full_model_controller::C2
    task_map::Vector{Tuple{RigidBody{Float64}, QPControl.AbstractMotionTask}} # TODO: abstract
    orientation_control_tasks::Vector{AngularAccelerationTask}
    joint_tasks::Vector{Tuple{JointID, Float64, JointAccelerationTask}}
    ik_workspace::IKMimicWorkspace{Float64}
end

# function MappedController(reduced_model::Mechanism, full_model::Mechanism, matching_body_names::AbstractVector{<:AbstractString})
#     ik_workspace = IKMimicWorkspace(full_model, reduced_model, matching_body_names)
#     MappedController(reduced_model, full_model, ik_workspace)
# end

function ensure_initialized!(c::MappedController)
    c.reduced_model_controller.initialized[] || QPControl.initialize!(c.reduced_model_controller)
    c.full_model_controller.initialized || QPControl.initialize!(c.full_model_controller)
end

function point_acceleration(state::MechanismState, J::PointJacobian, path::TreePath, point::Point3D, v̇::AbstractVector)
    frame = J.frame
    J̇v = transform(state,
        -bias_acceleration(state, source(path)) + bias_acceleration(state, target(path)),
        frame)
    T = transform(state,
        relative_twist(state, target(path), source(path)),
        frame)
    ω = angular(T)
    p = transform(state, point, frame)
    ṗ = point_velocity(T, p)

    @framecheck p.frame frame
    @framecheck ṗ.frame frame

    FreeVector3D(frame, ω × ṗ.v + J.J * v̇ + (angular(J̇v) × p.v + linear(J̇v)))
end

function (c::MappedController)(τ::AbstractVector, t::Real, x::Union{AbstractVector, MechanismState})
    ensure_initialized!(c)
    reduced_state = c.reduced_model_controller.state
    full_state = c.full_model_controller.state

    # Set the state of the reduced model based on the state of the full model
    copyto!(full_state, x)
    # if t < 0.01
        for i in 1:2
            ik_mimic!(reduced_state, full_state, c.ik_workspace)
        end
    # end

    # Run the reduced model controller
    SimpleQP.solve!(c.reduced_model_controller.qpmodel)

    # Extract the state at the end of the current time step from the reduced model controller
    set_configuration!(reduced_state, SimpleQP.value.(c.reduced_model_controller.qpmodel, c.reduced_model_controller.stages[1].q))
    set_velocity!(reduced_state, SimpleQP.value.(c.reduced_model_controller.qpmodel, c.reduced_model_controller.stages[1].v))

    # Set desired linear accelerations for the matching bodies in the full model
    for (reduced_model_body, full_model_task) in c.task_map
        # The target point is the origin of the body on both models
        point = Point3D(default_frame(reduced_model_body), 0., 0, 0)
        H_desired = transform_to_root(reduced_state, reduced_model_body)
        T_desired = twist_wrt_world(reduced_state, reduced_model_body)
        pref = H_desired * point
        ṗref = point_velocity(T_desired, pref)

        reduced_mechanism = reduced_state.mechanism
        path_to_reduced_body = path(reduced_mechanism, root_body(reduced_mechanism), reduced_model_body)
        p̈ref = point_acceleration(reduced_state,
            point_jacobian(reduced_state, path_to_reduced_body, pref),
            path_to_reduced_body,
            pref,
            SimpleQP.value.(c.reduced_model_controller.qpmodel, c.reduced_model_controller.stages[1].v̇))
        @framecheck p̈ref.frame root_frame(reduced_mechanism)
        @show p̈ref

        full_model_body = target(full_model_task.path)
        point = Point3D(default_frame(full_model_body), 0., 0, 0)
        H_current = transform_to_root(full_state, full_model_body)
        T_current = twist_wrt_world(full_state, full_model_body)
        pcurrent = H_current * point
        ṗcurrent = point_velocity(T_current, pcurrent)

        kp = 1.0
        gains = PDGains(kp, 2 * sqrt(kp))
        desired = FreeVector3D(root_frame(full_state.mechanism),
            pd(gains, pcurrent.v, pref.v, ṗcurrent.v, ṗref.v) + 1e-1 * p̈ref.v)
        setdesired!(full_model_task, desired)
    end

    # Update additional tasks to stabilize full model body orientations
    for task in c.orientation_control_tasks
        body = target(task.path)
        H_current = transform_to_root(full_state, body)
        T_current = transform(twist_wrt_world(full_state, body), inv(H_current))
        kp = 0.1
        gains = PDGains(kp, 2 * sqrt(kp))
        α_desired = FreeVector3D(T_current.frame, pd(gains, rotation(H_current), angular(T_current)))
        setdesired!(task, α_desired)
    end

    for (jointid, ref, task) in c.joint_tasks
        kp = 0.1
        gains = PDGains(kp, 2 * sqrt(kp))
        v̇desired = pd(gains, configuration(full_state, jointid)[1], ref, velocity(full_state, jointid)[1], 0.0)
        setdesired!(task, v̇desired)
    end

    c.full_model_controller(τ, t, x)
end

