using AlgebraicPetri
using OrdinaryDiffEq
using DiffEqFlux, Flux
using Catalyst
using Plots
using JSON
using CSV
using DataFrames

####################
# Define epi model #
####################
SIR = LabelledReactionNet{Float64, Float64}([:S=>1000.0, :I=>0.0, :R=>0.0],
                                            (:inf=>0.0008)=>((:S,:I)=>(:I,:I)),
                                            (:rec=>0.1)=>(:I=>:R));
SVIIR = LabelledReactionNet{Float64, Float64}([:S=>1000.0, :V=>0.0, :I_U=>1.0, :I_V=>0.0, :R=>0.0],
                                            (Symbol("β_{UU}")=>0.0008)=>((:S,:I_U)=>(:I_U,:I_U)),
                                            (Symbol("β_{UV}")=>0.0008)=>((:S,:I_V)=>(:I_U,:I_V)),
                                            (Symbol("β_{VU}")=>0.0008)=>((:V,:I_U)=>(:I_V,:I_U)),
                                            (Symbol("β_{VV}")=>0.0008)=>((:V,:I_V)=>(:I_V,:I_V)),
                                            (:γ_U=>0.1)=>(:I_U=>:R),
                                            (:γ_V=>0.1)=>(:I_V=>:R),
                                            (:ν=>0.0)=>(:S=>:V));

# Initial parameter guesses. The first 2 arguments are the rate parameters (in
# order of definition) and the next 3 are state parameters
SIR_param_guess = [1e-8, 1e-4, 1e7, 1e5, 1e-10]
SVIIR_param_guess = [1e-8, 1e-8,1e-8, 1e-8,1/14, 1/14, 1e-4, 6e5, 1e1, 1e3, 1e-1, 1e-1]

# Choose model
model = SVIIR
param_guess = SVIIR_param_guess

# Calculating the instantaneous rate of infection in the population
calc_inf_sviir(s,p) = s[1]*s[3]*p[1] + s[1]*s[4]*p[2] + s[2]*s[3]*p[3] + s[2]*s[4]*p[4]
calc_inf_sir(s, p) = s[1]*s[2]*p[1]

# Define the loss function given rates and states
function make_loss(pn, prob, times, data; calc_inf=calc_inf_sir, states=Dict(), rates=Dict())
    function loss(p)
        cur_p = exp.(p)
        u0 = exp.(p[(1:ns(pn)) .+ nt(pn)])
        for (k,v) in rates
            cur_p[k] = v
        end
        for (k,v) in states
            u0[k] = v
        end
        prob′ = remake(prob, p=cur_p, u0=u0, tspan=(0.0,150.0))
        sol = solve(prob′, Tsit5())
        sum(abs2, data .- [calc_inf(sol(t), cur_p) for t in times]), sol
    end
end

# Create a Catalyst model
sviir_rxn = ReactionSystem(SVIIR)
prob = ODEProblem(sviir_rxn, concentrations(model), (0.0,150.0), vcat(rates(model), concentrations(model)))

# Ingest data
state_data = CSV.read("georgia.csv", DataFrame)

# Get the change in "reported case" numbers for each day
times = 1:(length(state_data[:, :date])-1)
data = state_data[2:end, :cases] .- state_data[1:(end-1), :cases];

# Set a range of time to average over (in days)
avg_range = 3
week_times = 1:(length(data) - avg_range * 2)
week_avg = map((avg_range+1):(length(data)-avg_range)) do i
    sum(data[(i-avg_range):(i+avg_range)]) ./ (2*avg_range+1)
end
plot(week_times, week_avg)
savefig("real_data.png")

# Generate loss function (this fixes values give in keyword `rates` and `states`).
# The fixed values included here are described in the caption of Figure 5
l_func = make_loss(model, prob, week_times, week_avg; calc_inf=calc_inf_sviir, 
                                                      rates=Dict(5=>1/14, 6=>1/14), 
                                                      states=Dict(2=>1e-9, 4=>1e-9));

function form_grid(p_bounds, n_steps)
    n_dims = size(p_bounds)[1]
    if length(n_steps)==1
        n_steps = n_steps*ones(n_dims)
    end
    steps = zeros(n_dims)
    grids = repeat(Any[nothing],n_dims)
    for ii in 1:n_dims
        steps[ii] = (p_bounds[ii,2]-p_bounds[ii,1])/n_steps[ii]
        grids[ii] = p_bounds[ii,1]:steps[ii]:p_bounds[ii,2]
    end
    mesh = Iterators.product(grids...)
    # mesh = Array{Float64,length(p_bounds)}(undef,n_steps)
    return mesh
end

function grid_search(model, op, p_bounds, n_steps, loss_func, sample_data, sample_times)
    param_settings = form_grid(p_bounds,n_steps)
    losses = zeros(length(param_settings))
    sols = repeat(Any[nothing], length(param_settings))
    p_settings = repeat(Any[nothing],length(param_settings))
    ii = 1 
    for curr_params in param_settings
        p_settings[ii] = collect(curr_params)
        ii = ii + 1
    end;
    Threads.@threads for ii in 1:length(p_settings)
        curr_params = p_settings[ii]
        loss, sol = loss_func(curr_params)
        losses[ii] = loss
        sols[ii] = sol
    end;
    min_loss, idx_min = findmin(losses)
    # println("made it")
    # println(p_settings[idx_min])
    return p_settings[idx_min], sols[idx_min], min_loss
end

function form_sample(p_bounds, n_draws)
    samples = rand(size(p_bounds)[1],n_draws)
    for ii in 1:size(p_bounds)[1]
        samples[ii,:] = samples[ii,:]*(p_bounds[ii,2]-p_bounds[ii,1]) + p_bounds[ii,1]*ones(n_draws) 
    end
    return samples
end

function sample_search(model, op, p_bounds, n_draws, loss_func, sample_data, sample_times)
    param_settings = form_sample(p_bounds,n_draws)
    losses = zeros(size(param_settings)[1])
    sols = repeat(Any[nothing], size(param_settings)[1])
    p_settings = repeat(Any[nothing],size(param_settings)[1])
    for ii in 1:size(param_settings)[1]
        curr_params = param_settings[:,ii]
        p_settings[ii] = curr_params
    end;
    Threads.@threads for ii in 1:length(p_settings)
        curr_params = p_settings[ii]
        loss, sol = loss_func(curr_params)
        losses[ii] = loss
        sols[ii] = sol
    end;
    min_loss, idx_min = findmin(losses)
    # println("made it")
    # println(p_settings[idx_min])
    return p_settings[idx_min], sols[idx_min], min_loss
end


# SVIIR_param_guess = [1e-8, 1e-8,1e-8, 1e-8,1/14, 1/14, 1e-4, 6e5, 1e1, 1e3, 1e-1, 1e-1]
p_bounds = [1e-9 1e-7; 1e-9 1e-7; 1e-9 1e-7; 1e-9 1e-7; 1/15 1/13; 1/15 1/13; 1e-5 1e-3; 1e5 7e5; 1 20; 1e2 1e4; 1e-2 0.99; 1e-2 0.99]
n_steps = 4
# p_estimate, sol_estimate, loss = grid_search(model, prob, log.(p_bounds), n_steps, l_func, week_avg, week_times)

####
# Fit parameters to uniform sample from box
####
n_draws = 100000
p_estimate, sol_estimate, loss = sample_search(model, prob, log.(p_bounds), n_draws, l_func, week_avg, week_times)

####
# Fit parameters w/ DiffEqFlux
####
# p = DiffEqFlux.sciml_train(l_func,log.(param_guess),ADAM(0.1),maxiters = 1000)
p = DiffEqFlux.sciml_train(l_func,p_estimate,ADAM(0.1),maxiters = 1000)
# @show p

####
# Plot fit data against actual data
####
n_solve = solve(remake(prob, p=exp.(p.u), u0=exp.(p.u[(1:ns(model)) .+ nt(model)])), Tsit5())
# n_solve = solve(remake(prob, p=exp.(p_estimate), u0=exp.(p_estimate[(1:ns(model)) .+ nt(model)])), Tsit5())
p_times = range(1,149, length=1000)
plot(week_times, week_avg; linewidth=4, label="Georgia Data", yrange=(0,10000), yaxis="New daily infections (persons)", xaxis="Time (days)")
plot!(p_times, [calc_inf_sviir(n_solve(t), exp.(p.u)) for t in p_times]; linewidth=4, label="Estimated Data")
# plot!(p_times, [calc_inf_sviir(n_solve(t), exp.(p_estimate)) for t in p_times]; linewidth=4, label="Estimated Data")
savefig("fit_results_grid.png")

####
# Plot simulation with population in each of the states
####
plot(n_solve; labels=reshape(String.(snames(model)), (1,ns(model))), linewidth=4, yaxis="Population (persons)", xaxis="Time (days)")
savefig("sim_results_grid.png")