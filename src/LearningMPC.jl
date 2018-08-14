__precompile__()

module LearningMPC

using LCPSim
using LCPSim: LCPUpdate, contact_force, _getvalue
using RigidBodyDynamics
using MeshCatMechanisms
using Parameters: @with_kw
using MathProgBase.SolverInterface: AbstractMathProgSolver
using JuMP
using CoordinateTransformations: AffineMap
using Flux
using FluxExtensions
import ConditionalJuMP
using Base.Iterators: product
using DataFrames: DataFrame
using ProgressMeter
using Gurobi
using ForwardDiff
using RigidBodySim
using DiffEqCallbacks
using Compat

export playback,
       MPCParams,
       LQRSolution,
       MPCController


const StateLike = Union{MechanismState, LCPSim.StateRecord}

struct LQRSolution{T} <: Function
    Q::Matrix{T}
    R::Matrix{T}
    K::Matrix{T}
    S::Matrix{T}
    x0::Vector{T}
    u0::Vector{T}
    Δt::T
end

qv(x::Union{MechanismState, LCPSim.StateRecord}) = vcat(Vector(configuration(x)), Vector(velocity(x)))

function LQRSolution(x0::MechanismState{T}, Q, R, Δt, contacts::AbstractVector{<:Point3D}=Point3D[]) where T
    u0 = nominal_input(x0, contacts)
    v0 = copy(velocity(x0))
    velocity(x0) .= 0
    RigidBodyDynamics.setdirty!(x0)
    A, B, c, K, S = LCPSim.ContactLQR.contact_dlqr(x0, u0, Q, R, Δt, contacts)
    set_velocity!(x0, v0)
    LQRSolution{T}(Q, R, K, S, qv(x0), copy(u0), Δt)
end

function (c::LQRSolution)(τ, t, x)
    τ .= -c.K * (qv(x) .- c.x0) .+ c.u0
end

@with_kw mutable struct MPCParams{S1 <: AbstractMathProgSolver, S2 <: AbstractMathProgSolver}
    Δt::Float64 = 0.05
    horizon::Int = 15
    mip_solver::S1
    lcp_solver::S2
end


include("Models/Models.jl")
using .Models

include("mpc.jl")
include("learning.jl")
include("controlutils.jl")
include("simpleqp.jl")
include("mapped_control.jl")

function MeshCatMechanisms.setanimation!(vis::MechanismVisualizer,
    results::AbstractVector{<:LCPUpdate}, args...; kw...)
    ts = cumsum([r.Δt for r in results])
    qs = [configuration(r.state) for r in results]
    MeshCatMechanisms.setanimation!(vis, ts, qs, args...; kw...)
end

end
