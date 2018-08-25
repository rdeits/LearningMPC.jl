abstract type MPCSink <: Function end

struct MPCSampleSink{T, S <: Sample} <: MPCSink
    samples::Vector{S}
    keep_nulls::Bool
    lqrsol::LQRSolution{T}
    lqr_warmstart_index::Int
    learned_warmstart_index::Int
end

function MPCSampleSink(; keep_nulls::Bool=nothing,
                         lqrsol::LQRSolution=nothing,
                         lqr_warmstart_index::Int=nothing,
                         learned_warmstart_index::Int=nothing)
    NX = length(lqrsol.x0)
    NU = length(lqrsol.u0)
    T = eltype(lqrsol.x0)
    MPCSampleSink{T, Sample{NX, NU, T}}([], keep_nulls, lqrsol, lqr_warmstart_index, learned_warmstart_index)
end

Base.empty!(s::MPCSampleSink) = empty!(s.samples)

function (s::MPCSampleSink{T, S})(x::StateLike, results::MPCResults) where {T, S <: Sample}
    if s.keep_nulls || !isnull(results.lcp_updates)
        if isnull(results.lcp_updates)
            u = fill(NaN, num_velocities(x))
        else
            u = get(results.lcp_updates)[1].input
        end
        state = qv(x)
        lqr_warmstart_cost = results.warmstart_costs[s.lqr_warmstart_index]
        learned_warmstart_cost = results.warmstart_costs[s.learned_warmstart_index]
        warmstart_cost_record = WarmstartCostRecord(lqr_warmstart_cost, learned_warmstart_cost)
        sample = S(
            state,
            u,
            s.lqrsol.x0,
            s.lqrsol.u0,
            warmstart_cost_record,
            results.mip)
        push!(s.samples, sample)
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

_call_each!(fs::Tuple{}, args...) = nothing

function _call_each!(fs::Tuple, args...)
    f = first(fs)
    f(args...)
    _call_each!(Base.tail(fs), args...)
    return nothing
end

function multiplex!(f::Function...)
    function (args...)
        _call_each!(f, args...)
        return nothing
    end
end

Base.@deprecate call_each(args...) multiplex!(args...)

function live_viewer(vis::MechanismVisualizer)
    function (τ, t, x)
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

function dagger_controller(mpc_controller, net_controller)
    function (τ, t, x)
        mpc_controller(τ, t, x)
        net_controller(τ, t, x)  # run the MPC controller, then take the
                                 # action from the net controller
    end
 end

# function dagger_controller(mpc_controller, net_controller, p_mpc)
#     (τ, t, x) ->  begin
#         if rand() < p_mpc
#             mpc_controller(τ, t, x)
#             net_controller(τ, t, x)  # run the MPC controller, then take the
#                                      # action from the net controller
#         else
#             net_controller(τ, t, x)
#         end
#     end
# end

function randomize!(x::MechanismState, xstar::MechanismState, σ_q = 0.1, σ_v = 0.5)
    set_configuration!(x, configuration(xstar) .+ σ_q .* randn(num_positions(xstar)))
    set_velocity!(x, velocity(xstar) .+ σ_v .* randn(num_velocities(xstar)))
end

struct Dataset{T, S <: Sample}
    lqrsol::LQRSolution{T}
    training_data::Vector{S}
    validation_data::Vector{S}
    testing_data::Vector{S}

    function Dataset(lqrsol::LQRSolution{T}) where {T}
        NX = length(lqrsol.x0)
        NU = length(lqrsol.u0)
        new{T, Sample{NX, NU, T}}(lqrsol, [], [], [])
    end
end

regularize(regularization::Real) = regularization * 0.0

function regularize(regularization::Real, layer::Flux.Dense, others::Vararg{<:Flux.Dense, N}) where N
    (regularization .* sum(layer.W .^ 2) / length(layer.W) +  
     regularization .* sum(layer.b .^ 2) / length(layer.b) + 
     regularize(regularization, others...))
end

function interval_error(y, lb, ub)
    sum(@.(ifelse(y < lb, lb - y, ifelse(y > ub, y - ub, 0 * y))))
end

@noinline function _interval_net_loss(net, ::Type{Ty}) where {Ty}
    function(sample)
        x = sample.state
        lb = sample.mip.objective_bound
        ub = sample.mip.objective_value
        y::Ty = net(x)
        interval_error(y, lb, ub)
    end
end

function interval_net(widths, activation=Flux.elu; regularization=0.0)
    layers = Tuple([Dense(widths[i-1], widths[i], i==length(widths) ? identity : activation) for i in 2:length(widths)])
    net = Chain(layers...)
    Ty = typeof(net(zeros(first(widths))))
    sample_loss = _interval_net_loss(net, Ty)
    loss = let sample_loss = sample_loss, regularization = regularization, layers = layers
        function(sample)
            sample_loss(sample) + regularize(regularization, layers...)
        end
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
    # Q = ForwardDiff.hessian(c.net, x)
    # Q_psd = matrix_absolute_value(Q)
    xf = qv(results[end].state)
    lqrcost + q' * xf
    # lqrcost + q' * xf + 0.5 * xf' * Q_psd * xf
end

function _mimic_net_loss(net, ::Type{Ty}) where {Ty}
    function (sample)
        x = sample.state
        u = sample.input
        y::Ty = net(x)
        Flux.mse(y, u)
    end
end

function mimic_net(widths, activation=Flux.elu; regularization=0.0)
    layers = Tuple([Dense(widths[i-1], widths[i], i==length(widths) ? identity : activation) for i in 2:length(widths)])
    net = Chain(layers...)
    Ty = typeof(net(zeros(first(widths))))
    sample_loss = _mimic_net_loss(net, Ty)
    loss = let sample_loss = sample_loss, regularization = regularization, layers = layers
        function(sample)
            sample_loss(sample) + regularize(regularization, layers...)
        end
    end
    net, loss
end


function evaluate_controller(controller,
                             state::MechanismState,
                             lqrsol::LQRSolution;
                             Δt = 0.01,
                             duration::Float64 = 1.0,
                             mvis::Union{MechanismVisualizer, Void} = nothing)

    running_cost::Float64 = 0.0
    t_prev::Float64 = 0.0
    cost_tracking_controller = let controller=controller, 
                                   lqrsol=lqrsol, 
                                   x̄ = qv(state), 
                                   ū = zeros(num_velocities(state))
        function(τ, t, state)
            x̄ .= qv(state) .- lqrsol.x0
            controller(τ, t, state)
            ū .= τ .- lqrsol.u0
            dt = t - t_prev
            t_prev = t
            running_cost += dt * (x̄' * lqrsol.Q * x̄ + ū' * lqrsol.R * ū)
        end
    end

    problem = LearningMPC.simulation_problem(state, cost_tracking_controller, Δt, duration)
    solution = RigidBodySim.solve(problem, Tsit5(), abs_tol=1e-8, dt=1e-6)
    if mvis !== nothing
        setanimation!(mvis, solution)
        sleep(0.01)
    end

    copy!(state, solution.u[end])
    x̄ = qv(state) .- lqrsol.x0
    terminal_cost = x̄' * lqrsol.S * x̄

    (running_cost, terminal_cost, configuration(state), velocity(state))
    # results = LCPSim.simulate(state, controller, env, Δt, horizon, solver)
    # running_cost = sum(results) do result
    #     δx = qv(result.state) - lqrsol.x0
    #     δu = result.input - lqrsol.u0
    #     δx' * lqrsol.Q * δx + δu' * lqrsol.R * δu
    # end
    # δxf = qv(results[end].state) - lqrsol.x0
    # terminal_cost = δxf' * lqrsol.S * δxf
    # @assert (terminal_cost + running_cost) ≈ lqr_cost(lqrsol, results)
    # (running_cost, terminal_cost, configuration(results[end].state), velocity(results[end].state))
end

function run_evaluations(controller,
                         controller_label::AbstractString,
                         robot::AbstractModel,
                         lqrsol::LQRSolution,
                         q_ranges::AbstractVector{<:Tuple{Integer, AbstractVector}},
                         v_ranges::AbstractVector{<:Tuple{Integer, AbstractVector}};
                         Δt = 0.01, horizon = 200, mvis::Union{MechanismVisualizer, Void} = nothing)
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
        zero!(state)
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
        running_cost, terminal_cost, qf, vf = evaluate_controller(
            controller, state, lqrsol; Δt=Δt, duration=horizon * Δt, mvis=mvis)
        push!(results, [controller_label, q0, v0, Δt, horizon, qf, vf, running_cost, terminal_cost])
    end
    results
end



