export compose

_eval_composed(τ, t, state) = τ

function _eval_composed(τ, t, state, controller, others...)
    controller(τ, t, state)
    _eval_composed(τ, t, state, others...)
end

function compose(controllers...)
    let controllers = controllers
        function (τ, t, state)
            _eval_composed(τ, t, state, controllers...)
            τ
        end
    end
end

zero_controller(τ::AbstractVector, t::Number, state::MechanismState) = τ .= 0


function effort_limiter(mechanism)
    let effort_bounds = LCPSim.all_effort_bounds(mechanism)
        function (τ::AbstractVector, t::Number, state::MechanismState)
            τ .= clamp.(τ, effort_bounds)
            τ
        end
    end
end

function bounds_enforcer(mechanism, kp=100000., kd=0.001 * kp)
    let position_bounds = LCPSim.all_configuration_bounds(mechanism), kp = kp, kd = kd
        function (τ::AbstractVector, t::Number, state::MechanismState)
            # TODO: handle q̇ vs v correctly
            for i in 1:num_positions(state)
                if configuration(state)[i] > position_bounds[i].upper
                    violation = configuration(state)[i] - position_bounds[i].upper
                    τ[i] -= kp * violation
                    τ[i] -= kd * velocity(state)[i]
                elseif configuration(state)[i] < position_bounds[i].lower
                    violation = position_bounds[i].lower - configuration(state)[i]
                    τ[i] += kp * violation
                    τ[i] -= kd * velocity(state)[i]
                end
            end
            τ
        end
    end
end

function damper(mechanism, kd = 1.0)
    let kd = kd
        function(τ, t, state)
            v = velocity(state)
            τ .-= kd .* v
        end
    end
end

function digital_controller(state, controller, Δt; damping_kd=damping_kd)
    periodic = PeriodicController(
        similar(velocity(state)),
        Δt,
        compose(controller, effort_limiter(state.mechanism)))
    callback = DiffEqCallbacks.PeriodicCallback(periodic)
    composed = compose(periodic, bounds_enforcer(state.mechanism), damper(state.mechanism, damping_kd))
    composed, CallbackSet(callback)
end

function simulation_problem(state, controller, controller_Δt, final_time; damping_kd=0.0)
    composed_controller, callback = digital_controller(state, controller, controller_Δt; damping_kd=damping_kd)
    dynamics = Dynamics(state.mechanism, composed_controller)
    problem = ODEProblem(dynamics, state, (0., final_time), callback = callback)
end


