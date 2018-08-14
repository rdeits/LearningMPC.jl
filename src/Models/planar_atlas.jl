planar_atlas_urdf = joinpath(@__DIR__, "urdf", "planar_atlas_with_walls.urdf")

struct PlanarAtlas{T} <: AbstractModel{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
    floating_base::Joint{T}
end

mechanism(b::PlanarAtlas) = b.mechanism
environment(b::PlanarAtlas) = b.environment
urdf(b::PlanarAtlas) = planar_atlas_urdf

function PlanarAtlas(variant::Symbol)
    mechanism = parse_urdf(Float64, planar_atlas_urdf)
    remove_fixed_tree_joints!(mechanism)
    floating_base = findjoint(mechanism, "floating_base")
    replace_joint!(mechanism, floating_base,
        Joint("floating_base",
              frame_before(floating_base), frame_after(floating_base),
              Planar(SVector(0., 1, 0), SVector(0., 0, 1))))
    floating_base = findjoint(mechanism, "floating_base")
    floating_base.effort_bounds .= RigidBodyDynamics.Bounds(0., 0)

    # Add the body definitions to match box atlas
    modifications = [
        ("r_hand_mount", "r_hand", RotZYX(π, -π/2, 0), SVector(0, -0.195, 0)),
        ("l_hand_mount", "l_hand", RotZYX(π, -π/2, 0), SVector(0, -0.195, 0)),
        ("r_foot_sole", "r_foot", RotZYX(0., 0., 0.), SVector(0.0426, 0, -0.07645)),
        ("l_foot_sole", "l_foot", RotZYX(0., 0., 0.), SVector(0.0426, 0, -0.07645)),
    ]

    for (bodyname, basename, rot, trans) in modifications
        base = findbody(mechanism, basename)
        frame = CartesianFrame3D(bodyname)
        inertia = SpatialInertia(frame, SDiagonal(0., 0, 0), SVector(0., 0, 0), 0.0)
        body = RigidBody(inertia)
        joint = Joint("$(basename)_to_$(bodyname)", Fixed{Float64}())
        before_joint = Transform3D(frame_before(joint), default_frame(base), rot, trans)
        attach!(mechanism, base, body, joint, joint_pose=before_joint)
    end

    # Increase joint velocity limits because there's no restoring force
    # to ensure they aren't violated
    for joint in joints(mechanism)
        joint.velocity_bounds .= RigidBodyDynamics.Bounds(-1000, 1000)
    end

    floating_base.position_bounds .= RigidBodyDynamics.Bounds(-10, 10)
    floating_base.velocity_bounds .= RigidBodyDynamics.Bounds(-1000, 1000)

    if variant == :control

        feet = Dict(:left => findbody(mechanism, "l_foot_sole"),
                    :right => findbody(mechanism, "r_foot_sole"))
        hands = Dict(:left => findbody(mechanism, "l_hand_mount"),
                     :right => findbody(mechanism, "r_hand_mount"))
        floor = findbody(mechanism, "floor")
        wall = findbody(mechanism, "wall")

        # Load the obstacles from URDF but create our own contact points
        urdf_env = LCPSim.parse_contacts(mechanism, planar_atlas_urdf, 1.0, :xyz)
        obstacles = unique([c[3] for c in urdf_env.contacts])
        env = LCPSim.Environment{Float64}([
            (body, Point3D(default_frame(body), 0., 0., 0.), obstacle)
            for body in Iterators.flatten((values(feet), values(hands)))
            for obstacle in obstacles])

        LCPSim.filter_contacts!(env, mechanism,
            Dict(hands[:right] => [],
                 hands[:left] => [wall],
                 feet[:right] => [floor],
                 feet[:left] => [floor, wall]))
        return PlanarAtlas(mechanism, env, floating_base)
    elseif variant == :simulation
        urdf_env = LCPSim.parse_contacts(mechanism, planar_atlas_urdf, 1.0, :xyz)
        obstacles = unique([c[3] for c in urdf_env.contacts])
        state = MechanismState(mechanism)
        for obstacle in obstacles
            face = obstacle.contact_face
            point_in_world = transform(state, face.point, root_frame(mechanism))
            normal_in_world = transform(state, face.outward_normal, root_frame(mechanism))
            add_environment_primitive!(mechanism, HalfSpace3D(point_in_world, normal_in_world))
        end
        contactmodel = SoftContactModel(hunt_crossley_hertz(k = 500e3), ViscoelasticCoulombModel(0.8, 20e3, 100.))
        for side in (:left, :right)
            foot = findbody(mechanism, "$(first(string(side)))_foot")
            frame = default_frame(foot)
            z = -0.07645

            # heel
            add_contact_point!(foot, ContactPoint(Point3D(frame, -0.0876, AtlasRobot.flipsign_if_right(0.066, side), z), contactmodel))
            add_contact_point!(foot, ContactPoint(Point3D(frame, -0.0876, AtlasRobot.flipsign_if_right(-0.0626, side), z), contactmodel))

            # toe:
            add_contact_point!(foot, ContactPoint(Point3D(frame, 0.1728, AtlasRobot.flipsign_if_right(0.066, side), z), contactmodel))
            add_contact_point!(foot, ContactPoint(Point3D(frame, 0.1728, AtlasRobot.flipsign_if_right(-0.0626, side), z), contactmodel))
        end
        return PlanarAtlas(mechanism, LCPSim.Environment{Float64}([]), floating_base)
    end
end

function nominal_state(robot::PlanarAtlas)
    m = mechanism(robot)
    xstar = MechanismState{Float64}(m)
    kneebend = 1.1
    hipbendextra = 0.1
    for sideprefix in ('l', 'r')
        knee = findjoint(m, "$(sideprefix)_leg_kny")
        hippitch = findjoint(m, "$(sideprefix)_leg_hpy")
        anklepitch = findjoint(m, "$(sideprefix)_leg_aky")
        shoulderroll = findjoint(m, "$(sideprefix)_arm_shx")
        set_configuration!(xstar, knee, [kneebend])
        set_configuration!(xstar, hippitch, [-kneebend / 2 + hipbendextra])
        set_configuration!(xstar, anklepitch, [-kneebend / 2 - hipbendextra])
        if sideprefix == 'r'
            set_configuration!(xstar, shoulderroll, 1)
        else
            set_configuration!(xstar, shoulderroll, -1)
        end
    end
    set_configuration!(xstar, robot.floating_base, [0, 0.84, 0])
    xstar
end

function default_costs(robot::PlanarAtlas, r=1e-6)
    m = mechanism(robot)
    nq = num_positions(m)
    nv = num_velocities(m)
    qq = fill(1.0, nq)
    qq[1] = 10
    qq[2] = 100
    qq[3] = 1000
    qv = fill(0.01, nv)
    Q = diagm(vcat(qq, qv))
    R = diagm(fill(r, nv))
    Q, R
end

function LearningMPC.MPCParams(robot::PlanarAtlas)
    mpc_params = LearningMPC.MPCParams(
        Δt=0.05,
        horizon=10,
        mip_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0,
            TimeLimit=10,
            MIPGap=1e-2,
            FeasibilityTol=1e-3),
        lcp_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0))
end

function LearningMPC.LQRSolution(robot::PlanarAtlas, params::MPCParams=MPCParams(robot), zero_base_x=false)
    xstar = nominal_state(robot)
    Q, R = default_costs(robot)
    lqrsol = LearningMPC.LQRSolution(xstar, Q, R, params.Δt,
        [Point3D(default_frame(robot.feet[:left]), 0., 0., 0.),
         Point3D(default_frame(robot.feet[:right]), 0., 0., 0.)])
    if zero_base_x
        lqrsol.S[1,:] .= 0
        lqrsol.S[:,1] .= 0
        lqrsol.K[:,1] .= 0
    end
    lqrsol
end

function MechanismGeometries.URDFVisuals(robot::PlanarAtlas)
    @show AtlasRobot.packagepath()
    URDFVisuals(urdf(robot); package_path=[AtlasRobot.packagepath()])
end
