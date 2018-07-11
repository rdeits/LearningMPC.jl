abstract type MPCSink <: Function end

struct MPCSampleSink{T} <: MPCSink
    samples::Vector{Sample{T}}
    keep_nulls::Bool

    MPCSampleSink{T}(keep_nulls=false) where {T} = new{T}([], keep_nulls)
end

Base.empty!(s::MPCSampleSink) = empty!(s.samples)

function (s::MPCSampleSink)(x::StateLike, results::MPCResults)
    if s.keep_nulls || !isnull(results.lcp_updates)
        push!(s.samples, LearningMPC.Sample(x, results))
    end
end

mutable struct PlaybackSink{T} <: MPCSink
    vis::MechanismVisualizer
    last_trajectory::Vector{LCPSim.LCPUpdate{T, T, T, T}}

    PlaybackSink{T}(vis::MechanismVisualizer) where {T} = new{T}(vis, [])
end

function (p::PlaybackSink)(x::StateLike, results::MPCResults)
    if !isnull(results.lcp_updates)
        p.last_trajectory = get(results.lcp_updates)
        setanimation!(p.vis, p.last_trajectory)
        sleep(0.1)
    end
end

function call_each(f::Function...)
    f1 = first(f)
    f2 = Base.tail(f)
    (args...) -> begin
        result = f1(args...)
        (x -> x(args...)).(f2)
        return result
    end
end

# import Base: &
# (&)(s::MPCSink...) = (x, results) -> begin
#     for sink in s
#         sink(x, results)
#     end
# end

function live_viewer(vis::MechanismVisualizer)
    x -> begin
        set_configuration!(vis, configuration(x))
    end
end

function call_with_probability(args::Tuple{Function, Float64}...)
    p_total = sum(last.(args))
    p = last.(args) ./ p_total
    cdf = cumsum(p)
    @assert cdf[end] ≈ 1
    (x...) -> begin
        i = searchsortedfirst(cdf, rand())
        if i > length(cdf)
            i = length(cdf)
        end
        first(args[i])(x...)
    end
end

function dagger_controller(mpc_controller, net_controller, p_mpc)
    x ->  begin
        if rand() < p_mpc
            return mpc_controller(x)
        else
            return net_controller(x)
        end
    end
end

function randomize!(x::MechanismState, xstar::MechanismState, σ_q = 0.1, σ_v = 0.5)
    set_configuration!(x, configuration(xstar) .+ σ_q .* randn(num_positions(xstar)))
    set_velocity!(x, velocity(xstar) .+ σ_v .* randn(num_velocities(xstar)))
end

struct Dataset{T}
    lqrsol::LQRSolution{T}
    training_data::Vector{Sample{T}}
    validation_data::Vector{Sample{T}}
    testing_data::Vector{Sample{T}}

    Dataset(lqrsol::LQRSolution{T}) where {T} = new{T}(lqrsol, [], [], [])
end

function interval_net(widths, activation=Flux.elu)
    net = Chain([Dense(widths[i-1], widths[i], i==length(widths) ? identity : activation) for i in 2:length(widths)]...)
    loss = (x, lb, ub) -> begin
        y = net(x)
        sum(ifelse.(y .< lb, lb .- y, ifelse.(y .> ub, y .- ub, 0 .* y)))
    end
    net, loss
end

function log_interval_net(widths, activation=Flux.elu)
    net = Chain([Dense(widths[i-1], widths[i], i==length(widths) ? exp : activation) for i in 2:length(widths)]...)
    loss = (x, lb, ub) -> begin
        loglb = log(lb)
        logub = log(ub)
        logy = log(net(x))
        sum(ifelse.(logy .< loglb, loglb .- logy, ifelse.(logy .> logub, logy .- logub, 0 .* logy)))
    end
    net, loss
end

struct LearnedCost{T, F} <: Function
    lqr::LQRSolution{T}
    net::F

    LearnedCost{T, F}(lqr::LQRSolution{T}, net::F) where {T, F} = new{T, F}(lqr, net)
end

function LearnedCost(lqr::LQRSolution{T}, net) where T
    untracked = FluxExtensions.untrack(net)
    f = x -> untracked(x)[]
    LearnedCost{T, typeof(f)}(lqr, f)
end

function matrix_absolute_value(M::AbstractMatrix)
    fact = eigfact(M)
    S = fact[:vectors]
    M = Diagonal(abs.(fact[:values]))
    S * M * inv(S)
end

function (c::LearnedCost)(x0::StateLike, results::AbstractVector{<:LCPSim.LCPUpdate})
    lqr = c.lqr
    lqrcost = sum((r.state.state .- lqr.x0)' * lqr.Q * (r.state.state .- lqr.x0) +
                  (r.input .- lqr.u0)' * lqr.R * (r.input .- lqr.u0)
                  for r in results)
    x = qv(x0)
    q = ForwardDiff.gradient(c.net, x)
    Q = ForwardDiff.hessian(c.net, x)
    Q_psd = matrix_absolute_value(Q)
    xf = qv(results[end].state)
    lqrcost + q' * xf + 0.5 * xf' * Q_psd * xf
end

function evaluate_controller(controller,
                             state::MechanismState,
                             env::LCPSim.Environment,
                             lqrsol::LQRSolution,
                             Δt = 0.01,
                             horizon = 200,
                             solver = GurobiSolver(Gurobi.Env(), OutputFlag=0))
    results = LCPSim.simulate(state, controller, env, Δt, horizon, solver)
    running_cost = sum(results) do result
        δx = qv(result.state) - lqrsol.x0
        δu = result.input - lqrsol.u0
        δx' * lqrsol.Q * δx + δu' * lqrsol.R * δu
    end
    δxf = qv(results[end].state) - lqrsol.x0
    terminal_cost = δxf' * lqrsol.S * δxf
    @assert (terminal_cost + running_cost) ≈ lqr_cost(lqrsol, results)
    (running_cost, terminal_cost, configuration(results[end].state), velocity(results[end].state))
end

function run_evaluations(controller,
                         controller_label::AbstractString,
                         robot::AbstractModel,
                         lqrsol::LQRSolution,
                         q_ranges::AbstractVector{<:Tuple{Integer, AbstractVector}},
                         v_ranges::AbstractVector{<:Tuple{Integer, AbstractVector}};
                         Δt = 0.01, horizon = 200, solver = GurobiSolver(Gurobi.Env(), OutputFlag=0))
    x_nominal = nominal_state(robot)
    state = MechanismState(mechanism(robot))
    ranges = vcat(getindex.(q_ranges, 2), getindex.(v_ranges, 2))
    q0 = copy(configuration(state))
    v0 = copy(velocity(state))
    results = DataFrame(controller=String[],
                        q0=Vector{Float64}[],
                        v0=Vector{Float64}[],
                        Δt=Float64[],
                        horizon=Int[],
                        qf=Vector{Float64}[],
                        vf=Vector{Float64}[],
                        running_cost=Float64[],
                        terminal_cost=Float64[])
    @showprogress for evaluation_values in product(ranges...)
        q0 .= configuration(x_nominal)
        v0 .= velocity(x_nominal)
        for i in 1:length(q_ranges)
            q0[q_ranges[i][1]] = evaluation_values[i]
        end
        for i in 1:length(v_ranges)
            v0[v_ranges[i][1]] = evaluation_values[i + length(q_ranges)]
        end
        set_configuration!(state, q0)
        set_velocity!(state, v0)
        running_cost, terminal_cost, qf, vf = evaluate_controller(controller, state, environment(robot),
                                           lqrsol, Δt, horizon, solver)
        push!(results, [controller_label, q0, v0, Δt, horizon, qf, vf, running_cost, terminal_cost])
    end
    results
end



