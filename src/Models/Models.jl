module Models

using RigidBodyDynamics
using RigidBodyDynamics.Contact
using LCPSim
using LearningMPC
using MeshCat
using MeshCatMechanisms
using Gurobi
using MechanismGeometries
using AtlasRobot
using Rotations
using StaticArrays

export Hopper,
       CartPole,
       BoxAtlas,
       Slider,
       AbstractModel,
       mechanism,
       environment,
       nominal_state,
       default_costs,
       urdf


abstract type AbstractModel{T} end

function nominal_state end
function default_costs end
function mechanism end
function environment end
function urdf end


MeshCatMechanisms.MechanismVisualizer(h::AbstractModel, v::Visualizer=Visualizer()) = MechanismVisualizer(mechanism(h), URDFVisuals(h), v)

MechanismGeometries.URDFVisuals(h::AbstractModel) = URDFVisuals(urdf(h))

include("cartpole.jl")
include("boxatlas.jl")
include("hopper.jl")
include("slider.jl")
include("planar_atlas.jl")

end