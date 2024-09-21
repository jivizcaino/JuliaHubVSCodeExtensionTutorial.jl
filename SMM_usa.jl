#Set the Working Directory
#cd("C:\\Users\\lezjv\\Documents\\GitHub\\Workfiles\\STD")
cd("C:\\Users\\Nacho\\Documents\\GitHub\\Workfiles\\STD")
#-------------------------------------------------------------------
using Distributed

#if nworkers() == 1
#  addprocs(3)  # 
#end


## Set the topology
#@everywhere begin
#  Distributed.topology(:all_to_all)
#end


#-------------------------------------------------------------------
#using Pkg
#Pkg.activate("C:\\Users\\Nacho\\Documents\\GitHub\\JuliaHubVSCodeExtensionTutorial.jl\\Project.toml")
#Pkg.instantiate()
#Pkg.update()
#using LaTeXStrings, Random, Distributions, QuadGK , NLsolve, Optim , StatsBase
#using Statistics, LinearAlgebra, DataStructures, OrderedCollections
#using DataFrames, Plots, ParallelDataTransfer, SharedArrays, CSV, NaNMath
#using BlackBoxOptim, Printf
#using DifferentialEquations
@everywhere begin
  using JuliaHubVSCodeExtensionTutorial
  using JSON3
  using Distributions, Random, QuadGK, DataStructures, OrderedCollections
  using DataFrames, Plots, CSV, NaNMath, BlackBoxOptim, Printf,DifferentialEquations
end
#--------------------------------------------------------------

#-------------------------------------------------------------------
@everywhere begin
  μs0(g,ρ,A,a0)           = ( (1 - exp( - (g-ρ)*(A-a0))/(g-ρ)) )
  μs(μ0,z,ηs,β,as,a0)     = μ0*((z/(ηs*(1-β)*(as-a0) + z^(1-β)))^(β/(1-β)))
  w(wt,g,a,a0)            = wt*exp( -g*(a-a0) )
  w_E(wt,g,a,age)         = wt*exp( g*(a-age) )

  function hs(z,ηs,β,δ,as,a0)
    function ode1!(dh,h,p,a)
      ηs, β, δ = p
      dh[1] = ηs*(max(h[1],1e-10)^β) - δ*h[1]
    end
    h0 = [z]
    prob1  = ODEProblem(ode1!, h0, (a0, as), (ηs,β,δ))
    sol_1  = solve(prob1)
    return sol_1(as)[1]
  end

  function ho_esp(hs,ηo,α,δ,ao,as)
    function ode2!(dh,h,p,a)
      ηo,α,δ = p
      dh[1]  = ηo*(max(h[1],1e-10)^α) - δ*h[1]
    end
    h0     = [hs]
    prob2  = ODEProblem(ode2!,h0,(as,ao),(ηo,α,δ))
    sol_2  = solve(prob2)
    return sol_2(ao)[1]
  end

  ih(α,ηo,ρ,δ,g,A,a)    = ( (α*ηo*( 1 - exp( -(g-ρ-δ)*(A-a) ))) / (g-ρ-δ) )^(1/(1-α))
  Ia(α,ηo,ρ,δ,g,A,a0,a) = quadgk( s -> ( exp(-δ*s)*ηo*(ih(α,ηo,ρ,δ,g,A,s)^α) ),a0,a)[1] 

  #Human Capital
  #If ao <  as for all a >= as. (Pre-multiply by 1(a>as) to get the right results)
  #If ao <  as for all a >= as. (Pre-multiply by 1(a>as) to get the right results)
  Hs(z,β,α,ηs,ηo,ρ,δ,g,A,a0,as,a)        = exp(δ*(a-as))*( hs(z,ηs,β,δ,as,a0) + Ia(α,ηo,ρ,δ,g,A,a0,a) - Ia(α,ηo,ρ,δ,g,A,a0,as) )  
  Es(wt,z,ηo,ηs,α,β,ρ,δ,g,A,a0,as,a,age) = w(wt,g,age,a0)*(Hs(z,β,α,ηs,ηo,ρ,δ,g,A,a0,as,a) - ih(α,ηo,ρ,δ,g,A,a))  

  #If ao >= as for all a >= ao. (Pre-multiply by 1(a>ao) to get the right results)
  Ho(hs,α,ηo,ρ,δ,g,A,a0,as,ao,a)        = (exp(-δ*(a-ao))*( ho_esp(hs,ηo,α,δ,ao,as) + Ia(α,ηo,ρ,δ,g,A,ao,a) - Ia(α,ηo,ρ,δ,g,A,a0,ao)) ) 
  Eo(wt,hs,ηo,α,ρ,δ,g,A,a0,as,ao,a,age) = w(wt,g,age,a0)*(Ho(hs,α,ηo,ρ,δ,g,A,a0,as,ao,a) - ih(α,ηo,ρ,δ,g,A,a))  
end
#--------------------------------------------------------------

#--------------------------------------------------------------
@everywhere begin
  function simulate_moms(params,N,Πc_25_2000,Πr_25_2000,Πc_50_2000,Πr_50_2000,Πc_25_2010,Πr_25_2010,πc_25_2000,πc_25_2010,πc_50_2000)
    A     = 65
    a0    = 6
    μ_c   = 0.000
    μ_r   = 0.000
    c     = 0.000
    δ_c   = 0.025
    δ_r   = 0.025
    ρ_cr  = 0.000  
    ρ     = 0.010
    target_value = 0.0001

    σ_cc    = params[1]
    σ_rr    = params[2]
    wc      = params[3]
    wr      = params[4]
    ηo_c    = params[5]
    ηo_r    = params[6]
    α_c     = params[7]
    α_r     = params[8]
    ηs_c    = params[9]
    ηs_r    = params[10]
    β_c     = params[11]
    β_r     = params[12]
    gc      = 0.0180
    gr      = 0.0075

    μ       = [μ_c,μ_r]
    σ_cr    = ρ_cr*sqrt(σ_cc*σ_rr)
    Σ       = [σ_cc σ_cr;σ_cr σ_rr]  #Remember that this will be the var-cov mat of the normal, not the lognormal
    z       = rand(MersenneTwister(1234),MvLogNormal(μ,Σ),10*N)
    zc_base = z[1,:] .+ c
    zr_base = z[2,:] .+ c
    z_c_p99 = quantile(zc_base,0.95)
    z_r_p99 = quantile(zr_base,0.95)
    zc_base = zc_base[zc_base .< z_c_p99]
    zr_base = zr_base[zr_base .< z_r_p99]
    zc_base = sample(MersenneTwister(1234),zc_base,N,replace=true)
    zr_base = sample(MersenneTwister(1234),zr_base,N,replace=true)

    ih_c    = ih.(α_c,ηo_c,ρ,δ_c,gc,A,(a0:A))
    ih_r    = ih.(α_r,ηo_r,ρ,δ_r,gr,A,(a0:A))

    #hs_c    = hs.(zc_base',ηs_c,β_c,δ_c,(a0+1:A+1),a0)
    #ho_c    = hs.(zc_base',ηo_c,α_c,δ_c,(a0+1:A+1),a0)
    #Ia_c    = Ia.(α_c,ηo_c,ρ,δ_c,gc,A,a0,(a0:A))

    #hs_r    = hs.(zr_base',ηs_r,β_r,δ_r,(a0+1:A+1),a0)
    #ho_r    = hs.(zr_base',ηo_r,α_r,δ_r,(a0+1:A+1),a0)
    
    #Ia_r    = Ia.(α_r,ηo_r,ρ,δ_r,gr,A,a0,(a0:A))
  
    #hs_c_0  = hs_c[1,:] 
    #hs_c_6  = hs_c[7,:]
    #hs_c_13 = hs_c[12,:]
    #hs_c_16 = hs_c[17,:]

    ao = []
    #1. Comute PDVE for 25 year olds, 4 educational levels, and two occupations
    #Keep in mind that we first compute the PDVE for cognitive occupations using
    #wc=1.00 and we recompute the wage in the SMM stage
    wc_base = 1.000
    wr_base = 1.000

    #No Schooling, Cognitive
    ys  = 0
    as  = (ys .+ a0)

    hs_c_0       = zc_base
    ho_esp_ys0   = ho_esp.(hs_c_0',ηo_c,α_c,δ_c,as:A,as) #Compute human capital under full specialization after schooling
    ia           = ih_c[ys+1:end]./ho_esp_ys0
    ao           = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1

    age = 25
    Vc_25_0_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_0',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vc_50_0_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_0',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    
    #Primary, Cognitive
    ys = 6
    as = (ys .+ a0)

    #Compute human capital under full specialization after schooling
    hs_c_6     = hs.(zc_base',ηs_c,β_c,δ_c,as,a0)'
    ho_esp_ys6 = hs.(hs_c_6',ηo_c,α_c,δ_c,(as:A),a0)
    ia         = ih_c[ys+1:end]./ho_esp_ys6
    ao         = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1
    age = 25
    Vc_25_6_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_6',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vc_50_6_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_6',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    
    #Secondary, Cognitive
    ys = 11
    as = (ys .+ a0)

    #Compute human capital under full specialization after schooling
    hs_c_11     = hs.(zc_base',ηs_c,β_c,δ_c,as,a0)'
    ho_esp_ys11 = hs.(hs_c_11',ηo_c,α_c,δ_c,(as:A),a0)
    ia          = ih_c[ys+1:end]./ho_esp_ys11
    ao          = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1
    age = 25
    Vc_25_11_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_11',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vc_50_11_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_11',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    

    #Uni, Cognitive
    ys = 16
    as = (ys .+ a0)

    #Compute human capital under full specialization after schooling
    hs_c_16     = hs.(zc_base',ηs_c,β_c,δ_c,as,a0)'
    ho_esp_ys16 = hs.(hs_c_16',ηo_c,α_c,δ_c,(as:A),a0)
    ia          = ih_c[ys+1:end]./ho_esp_ys16
    ao          = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1
    age = 25
    Vc_25_16_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_16',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vc_50_16_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wc_base,hs_c_16',ηo_c,α_c,ρ,δ_c,gc,A,a0,as,ao',a0:A,age),dims=1)
    
    #No Schooling, Routine
    ys  = 0
    as  = (ys .+ a0)

    hs_r_0       = zr_base
    ho_esp_ys0   = ho_esp.(hs_r_0',ηo_r,α_r,δ_r,as:A,as) #Compute human capital under full specialization after schooling
    ia           = ih_r[ys+1:end]./ho_esp_ys0
    ao           = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1

    age = 25
    Vr_25_0_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_0',ηo_r,α_r,ρ,δ_r,gr,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vr_50_0_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_0',ηo_r,α_r,ρ,δ_r,gr,A,a0,as,ao',a0:A,age),dims=1)
    
    #Primary, Cognitive
    ys = 6
    as = (ys .+ a0)

    #Compute human capital under full specialization after schooling
    hs_r_6     = hs.(zr_base',ηs_r,β_r,δ_r,as,a0)'
    ho_esp_ys6 = hs.(hs_r_6',ηo_r,α_r,δ_r,(as:A),a0)
    ia         = ih_r[ys+1:end]./ho_esp_ys6
    ao         = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1
    age = 25
    Vr_25_6_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_6',ηo_r,α_r,ρ,δ_r,gr,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vr_50_6_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_6',ηo_r,α_r,ρ,δ_r,gr,A,a0,as,ao',a0:A,age),dims=1)
    
    #Secondary, Cognitive
    ys = 11
    as = (ys .+ a0)

    #Compute human capital under full specialization after schooling
    hs_r_11     = hs.(zr_base',ηs_r,β_r,δ_r,as,a0)'
    ho_esp_ys11 = hs.(hs_r_11',ηo_r,α_r,δ_r,(as:A),a0)
    ia          = ih_r[ys+1:end]./ho_esp_ys11
    ao          = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1
    age = 25
    Vr_25_11_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_11',ηo_r,α_r,ρ,δ_r,gr,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vr_50_11_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_11',ηo_r,α_r,ρ,δ_r,gr,A,a0,as,ao',a0:A,age),dims=1)
    
    #Uni, Cognitive
    ys = 16
    as = (ys .+ a0)

    #Compute human capital under full specialization after schooling
    hs_r_16     = hs.(zr_base',ηs_r,β_r,δ_r,as,a0)'
    ho_esp_ys16 = hs.(hs_r_16',ηo_r,α_r,δ_r,(as:A),a0)
    ia          = ih_r[ys+1:end]./ho_esp_ys16
    ao          = [findfirst(x -> x < 1, ia[:, col]) for col in 1:size(ia, 2)] .+ a0 .- 1
    age = 25
    Vr_25_16_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_16',ηo_r,α_r,ρ,δ_r,gr,A,a0,as,ao',a0:A,age),dims=1)
    age = 50
    Vr_50_16_2000 = sum(exp.(-ρ.*((a0:A).-a0)).*Eo.(wr_base,hs_r_16',ηo_r,α_r,ρ,δ_c,gr,A,a0,as,ao',a0:A,age),dims=1)
    
    # Assign the vectors to the matrix
    V_25_2000 = zeros(Float64,N,2,4)
    V_25_2000[:, 1, 1] = Vc_25_0_2000./100  
    V_25_2000[:, 1, 2] = Vc_25_6_2000./100  
    V_25_2000[:, 1, 3] = Vc_25_11_2000./100  
    V_25_2000[:, 1, 4] = Vc_25_16_2000./100  
    V_25_2000[:, 2, 1] = Vr_25_0_2000./100  
    V_25_2000[:, 2, 2] = Vr_25_6_2000./100  
    V_25_2000[:, 2, 3] = Vr_25_11_2000./100  
    V_25_2000[:, 2, 4] = Vr_25_16_2000./100  

    V_50_2000 = zeros(Float64,N,2,4)
    V_50_2000[:, 1, 1] = Vc_50_0_2000./100  
    V_50_2000[:, 1, 2] = Vc_50_6_2000./100  
    V_50_2000[:, 1, 3] = Vc_50_11_2000./100  
    V_50_2000[:, 1, 4] = Vc_50_16_2000./100  
    V_50_2000[:, 2, 1] = Vr_50_0_2000./100  
    V_50_2000[:, 2, 2] = Vr_50_6_2000./100  
    V_50_2000[:, 2, 3] = Vr_50_11_2000./100  
    V_50_2000[:, 2, 4] = Vr_50_16_2000./100  

    # Map observed probabilities to a matrix
    Probs_c_25_2000 = [Πc_25_2000["π_ns"]+Πc_25_2000["π_prim_2"]+Πc_25_2000["π_prim_4"];Πc_25_2000["π_prim_comp"]+Πc_25_2000["π_sec_2"];Πc_25_2000["π_sec_comp"]+Πc_25_2000["π_uni_2"];Πc_25_2000["π_uni"]]
    Probs_r_25_2000 = [Πr_25_2000["π_ns"]+Πr_25_2000["π_prim_2"]+Πr_25_2000["π_prim_4"];Πr_25_2000["π_prim_comp"]+Πr_25_2000["π_sec_2"];Πr_25_2000["π_sec_comp"]+Πr_25_2000["π_uni_2"];Πr_25_2000["π_uni"]]
    Probs_c_50_2000 = [Πc_50_2000["π_ns"]+Πc_50_2000["π_prim_2"]+Πc_50_2000["π_prim_4"];Πc_50_2000["π_prim_comp"]+Πc_50_2000["π_sec_2"];Πc_50_2000["π_sec_comp"]+Πc_50_2000["π_uni_2"];Πc_50_2000["π_uni"]]
    Probs_r_50_2000 = [Πr_50_2000["π_ns"]+Πr_50_2000["π_prim_2"]+Πr_50_2000["π_prim_4"];Πr_50_2000["π_prim_comp"]+Πr_50_2000["π_sec_2"];Πr_50_2000["π_sec_comp"]+Πr_50_2000["π_uni_2"];Πr_50_2000["π_uni"]]
    Probs_c_25_2010 = [Πc_25_2010["π_ns"]+Πc_25_2010["π_prim_2"]+Πc_25_2010["π_prim_4"];Πc_25_2010["π_prim_comp"]+Πc_25_2010["π_sec_2"];Πc_25_2010["π_sec_comp"]+Πc_25_2010["π_uni_2"];Πc_25_2010["π_uni"]]
    Probs_r_25_2010 = [Πr_25_2010["π_ns"]+Πr_25_2010["π_prim_2"]+Πr_25_2010["π_prim_4"];Πr_25_2010["π_prim_comp"]+Πr_25_2010["π_sec_2"];Πr_25_2010["π_sec_comp"]+Πr_25_2010["π_uni_2"];Πr_25_2010["π_uni"]]

    # Combined observed probabilities for both occupations
    observed_probs_25_2000   = [Probs_c_25_2000*πc_25_2000;Probs_r_25_2000*(1-πc_25_2000)]
    observed_probs_50_2000   = [Probs_c_50_2000*πc_50_2000;Probs_r_50_2000*(1-πc_50_2000)]
    observed_probs_25_2010   = [Probs_c_25_2010*πc_25_2010;Probs_r_25_2010*(1-πc_25_2010)]
    observed_probs_25_2000 ./= sum(observed_probs_25_2000)  # Normalize to sum to 1
    observed_probs_50_2000 ./= sum(observed_probs_50_2000)  # Normalize to sum to 1
    observed_probs_25_2010 ./= sum(observed_probs_25_2010)  # Normalize to sum to 1

    # Function to calculate the probability of each choice
    function choice_probabilities(ιc,ιr,V,wc,wr,Cc,Cr)
      #V = V_25_2000
      N, J, K = size(V)
      εc = rand(MersenneTwister(1234),Gumbel(0,ιc), N, K)  #Random utility shocks
      εr = rand(MersenneTwister(1234),Gumbel(0,ιr), N, K)  #Random utility shocks
      U           = zeros(Float64,N,J,K)
      U[:, 1, :]  = V[:, 1, :] .+ εc  # Base utility without cost adjustments
      U[:, 2, :]  = V[:, 2, :] .+ εr  # Base utility without cost adjustments

      # Apply wage adjustments
      U[:, 1, :] .*= wc
      U[:, 2, :] .*= wr

      C = zeros(Float64,J,K)
      C[1,1] = Cc[1]
      C[1,2] = Cc[2]
      C[1,3] = Cc[3]
      C[1,4] = Cc[4]
      C[2,1] = 0.00
      C[2,2] = Cr[1]
      C[2,3] = Cr[2]
      C[2,4] = Cr[3]

      # Apply cost adjustments
      for k in 1:K  
          U[:, 1, k] .-= C[1, k]
          U[:, 2, k] .-= C[2, k]
      end

      # Calculate the choice probabilities using the logit model
      exp_U  = exp.(U)
      total_exp_U = sum(exp_U, dims=(2,3))
      P     =  exp_U ./ total_exp_U

      # Aggregate the probabilities across individuals
      aggregate_probs = sum(P, dims=1) / N
      return aggregate_probs
    end

    # Objective function for SMM
    function objective(params,ιc,ιr,wr,V,observed_probs)
      wc          = params[1]
      Cc          = params[2:5]    # Costs for occupation 1 (excluding normalized)
      Cr          = params[6:8]    # Costs for occupation 2 
      simulated_probs = choice_probabilities(ιc,ιr,V,wc,wr,Cc,Cr)
      #return (1/size(observed_probs)[1])*sum( abs.((vec(simulated_probs) .- observed_probs)./observed_probs) )
      return 100*sum( abs.((vec(hcat(simulated_probs[:, 1, :],simulated_probs[:, 2, :])) .- observed_probs)) )
    end
    #--------------------------------------------------------------

    #-------------------------------------------------------------
    #Set βc and βr to match a target share of total variance
    function sqrt_term(ι, w, V)
        return sqrt((ι*π)^2/6)/std(w .* V)
    end

    # Bisection method to find the β value that makes sqrt_term close to target
    function bisection_sqrt_term(target,w,V,ι_low,ι_high,tol=1e-6,max_iter=100)
        for i in 1:max_iter
            ι_mid = (ι_low + ι_high) / 2
            value = sqrt_term(ι_mid, w, V)
            
            if abs(value - target) < tol
                return ι_mid
            elseif value > target
                ι_high = ι_mid
            else
                ι_low = ι_mid
            end
        end
        error("Bisection method did not converge")
    end

    # Initial values
    wr_bisec = wr
    wc_bisec = wc
    #wc_bisec = wr_bisec*0.65
    #target_value = 0.10

    # Adjust ιc
    ιc_low, ιc_high = 0.0, 20.0  # Initial range for ιc
    ιc = bisection_sqrt_term(target_value,wc_bisec, V_25_2000[:, 1, :], ιc_low, ιc_high)
    #@printf("Adjusted ιc: %.6f\n", ιc)

    # Adjust ιr
    ιr_low, ιr_high = 0.0, 20.0  # Initial range for βr
    ιr = bisection_sqrt_term(target_value,wr_bisec, V_25_2000[:, 2, :], ιr_low, ιr_high)
    #@printf("Adjusted ιr: %.6f\n", ιr)
    #--------------------------------------------------------------
    #BLACK BOX OPTIMIZATION
    wr_25_00    = wr
    ParamSpace  = [(wc*0.99,wc*1.01), #wc
                  (-10.00,100.00),  #Cc1
                  (-10.00,100.00),  #Cc2
                  (-10.00,100.00),  #Cc3
                  (-10.00,100.00),  #Cc4
                  (-10.00,100.00),  #Cr1
                  (-10.00,100.00),  #Cr2
                  (-10.00,100.00)]  #Cc3
    #--------------------------------------------------------------
    #adaptive_de_rand_1_bin_radiuslimited
    #25 YEAR OLDS: 2000
    opt_problem_norm_25_00    =  bbsetup(params -> objective(params,ιc,ιr,wr_25_00,V_25_2000,observed_probs_25_2000);
                                                      SearchRange=ParamSpace,TraceMode=:silent,
                                                      NumDimensions=7,MaxFuncEvals=10,
                                                      TargetFitness=0.01,Method=:adaptive_de_rand_1_bin_radiuslimited)
                                                      

    @elapsed res_para_25_2000 = bboptimize(opt_problem_norm_25_00,MaxFuncEvals = 40000)
    estimated_params_25_2000  = best_candidate(res_para_25_2000)
    wc_25_00 = estimated_params_25_2000[1]
    #--------------------------------------------------------------

    #--------------------------------------------------------------
    #25 YEAR OLDS: 2010
    #BLACK BOX OPTIMIZATION0
    wr_25_10    = wr_25_00*exp(gr*10)
    wc_25_10    = wc_25_00*exp(gc*10)
    ParamSpace_25_10  = [(wc_25_10*0.98,wc_25_10*1.02),  #wc
                      (-10.00,100.00),  #Cc1
                      (-10.00,100.00),  #Cc2
                      (-10.00,100.00),  #Cc3
                      (-10.00,100.00),  #Cc4
                      (-10.00,100.00),  #Cr1
                      (-10.00,100.00),  #Cr2
                      (-10.00,100.00)]  #Cr3
    opt_problem_norm_25_10  =  bbsetup(params -> objective(params,ιc,ιr,wr_25_10,V_25_2000,observed_probs_25_2010);
                                                      SearchRange=ParamSpace_25_10,TraceMode=:silent,
                                                      NumDimensions=8,MaxFuncEvals=10,
                                                      TargetFitness=0.01,Method=:adaptive_de_rand_1_bin_radiuslimited)

    @elapsed res_para_25_2010 = bboptimize(opt_problem_norm_25_10,MaxFuncEvals = 40000)
    estimated_params_25_2010  = best_candidate(res_para_25_2010)
    wc_25_10 = estimated_params_25_2010[1]
    wc_25_00*exp(gc*10)
    #--------------------------------------------------------------

    #--------------------------------------------------------------
    #50 YEAR OLDS: 2000
    #BLACK BOX OPTIMIZATION
    wc_50_00    = wc_25_00*exp(-gc*25)
    wr_50_00    = wr_25_00*exp(-gr*25)
    ParamSpace_50_10  = [( wc_50_00*0.95,wc_50_00*1.05),  #wc
                        (-10.00,100.00),  #Cc1
                        (-10.00,100.00),  #Cc2
                        (-10.00,100.00),  #Cc3
                        (-10.00,100.00),  #Cc4
                        (-10.00,100.00),  #Cr1
                        (-10.00,100.00),  #Cc2
                        (-10.00,100.00)]  #Cc3
    opt_problem_norm_50_10  =  bbsetup(params -> objective(params,ιc,ιr,wr_50_00,V_50_2000,observed_probs_50_2000);
                                                      SearchRange=ParamSpace_50_10,TraceMode=:silent,
                                                      NumDimensions=8,MaxFuncEvals=10,
                                                      TargetFitness=0.01,Method=:adaptive_de_rand_1_bin_radiuslimited)

    @elapsed res_para_50_2000 = bboptimize(opt_problem_norm_50_10,MaxFuncEvals = 40000)
    estimated_params_50_2000  = best_candidate(res_para_50_2000)
    estimated_params_50_2000[1]

    #--------------------------------------------------------------

    wc_25_00 = estimated_params_25_2000[1]
    wc_25_10 = estimated_params_25_2010[1]
    wc_50_00 = estimated_params_50_2000[1]

    #Example usage of choice_probabilities with the estimated parameters
    Cc_25_00    = zeros(Float64,1,4)
    Cr_25_00    = zeros(Float64,1,3)
    Cc_25_00[1] = estimated_params_25_2000[2]
    Cc_25_00[2] = estimated_params_25_2000[3]
    Cc_25_00[3] = estimated_params_25_2000[4]
    Cc_25_00[4] = estimated_params_25_2000[5]
    Cr_25_00[1] = estimated_params_25_2000[6]
    Cr_25_00[2] = estimated_params_25_2000[7]
    Cr_25_00[3] = estimated_params_25_2000[8]
    probs_25_00 = choice_probabilities(ιc,ιr,V_25_2000,wc_25_00,wr_25_00,Cc_25_00,Cr_25_00)

    Cc_50_00     = zeros(Float64,1,4)
    Cr_50_00     = zeros(Float64,1,3)
    Cc_50_00[1]  = estimated_params_50_2000[2]
    Cc_50_00[2]  = estimated_params_50_2000[3]
    Cc_50_00[3]  = estimated_params_50_2000[4]
    Cc_50_00[4]  = estimated_params_50_2000[5]
    Cr_50_00[1]  = estimated_params_50_2000[6]
    Cr_50_00[2]  = estimated_params_50_2000[7]
    Cr_50_00[3]  = estimated_params_50_2000[8]
    probs_50_00  = choice_probabilities(ιc,ιr,V_50_2000,wc_50_00,wr_50_00,Cc_50_00,Cr_50_00)

    Cc_25_10    = zeros(Float64,1,4)
    Cr_25_10    = zeros(Float64,1,3)
    Cc_25_10[1] = estimated_params_25_2010[2]
    Cc_25_10[2] = estimated_params_25_2010[3]
    Cc_25_10[3] = estimated_params_25_2010[4]
    Cc_25_10[4] = estimated_params_25_2010[5]
    Cr_25_10[1] = estimated_params_25_2010[6]
    Cr_25_10[2] = estimated_params_25_2010[7]
    Cr_25_10[3] = estimated_params_25_2010[8]

    probs_25_10 = choice_probabilities(ιc,ιr,V_25_2000,wc_25_10,wr_25_10,Cc_25_10,Cr_25_10)
    #--------------------------------------------------------------
    V_25_2000_rand = zeros(Float64,N,2,4)
    V_25_2010_rand = zeros(Float64,N,2,4) 
    V_50_2000_rand = zeros(Float64,N,2,4)   
    V_25_2000_rand[:, 1, :] = V_25_2000[:, 1, :] .+ rand(Gumbel(0,ιc),N,4)
    V_25_2000_rand[:, 2, :] = V_25_2000[:, 2, :] .+ rand(Gumbel(0,ιr),N,4)
    V_25_2010_rand[:, 1, :] = V_25_2000[:, 1, :] .+ rand(Gumbel(0,ιc),N,4)
    V_25_2010_rand[:, 2, :] = V_25_2000[:, 2, :] .+ rand(Gumbel(0,ιr),N,4)
    V_50_2000_rand[:, 1, :] = V_50_2000[:, 1, :] .+ rand(Gumbel(0,ιc),N,4)
    V_50_2000_rand[:, 2, :] = V_50_2000[:, 2, :] .+ rand(Gumbel(0,ιr),N,4)

    V_25_2000_rand[:, 1, :] .*= wc_25_00
    V_25_2000_rand[:, 2, :] .*= wr_25_00
    V_25_2000_rand[:, 1, 1] .-= Cc_25_00[1]      
    V_25_2000_rand[:, 1, 2] .-= Cc_25_00[2]
    V_25_2000_rand[:, 1, 3] .-= Cc_25_00[3]
    V_25_2000_rand[:, 1, 4] .-= Cc_25_00[4] 
    V_25_2000_rand[:, 2, 1] .-= 0.000 
    V_25_2000_rand[:, 2, 2] .-= Cr_25_00[1]   
    V_25_2000_rand[:, 2, 3] .-= Cr_25_00[2]   
    V_25_2000_rand[:, 2, 4] .-= Cr_25_00[3] 

    V_25_2010_rand[:, 1, :] .*= wc_25_10
    V_25_2010_rand[:, 2, :] .*= wr_25_10
    V_25_2010_rand[:, 1, 1] .-= Cc_25_10[1]       
    V_25_2010_rand[:, 1, 2] .-= Cc_25_10[2]
    V_25_2010_rand[:, 1, 3] .-= Cc_25_10[3]
    V_25_2010_rand[:, 1, 4] .-= Cc_25_10[4] 
    V_25_2010_rand[:, 2, 1] .-= 0.000 
    V_25_2010_rand[:, 2, 2] .-= Cr_25_10[1]   
    V_25_2010_rand[:, 2, 3] .-= Cr_25_10[2]   
    V_25_2010_rand[:, 2, 4] .-= Cr_25_10[3] 

    V_50_2000_rand[:, 1, :] .*= wc_50_00
    V_50_2000_rand[:, 2, :] .*= wr_50_00
    V_50_2000_rand[:, 1, 1] .-= Cc_50_00[1]      
    V_50_2000_rand[:, 1, 2] .-= Cc_50_00[2]
    V_50_2000_rand[:, 1, 3] .-= Cc_50_00[3]
    V_50_2000_rand[:, 1, 4] .-= Cc_50_00[4] 
    V_50_2000_rand[:, 2, 1] .-= 0.000 
    V_50_2000_rand[:, 2, 2] .-= Cr_50_00[1]   
    V_50_2000_rand[:, 2, 3] .-= Cr_50_00[2]   
    V_50_2000_rand[:, 2, 4] .-= Cr_50_00[3] 
    #--------------------------------------------------------------
    # Calculate the choice probabilities using the logit model
    exp_V_25_2000_rand       = exp.(V_25_2000_rand)
    total_exp_V_25_2000_rand = sum(exp_V_25_2000_rand, dims=(2,3))
    P_25_00                  = exp_V_25_2000_rand ./ total_exp_V_25_2000_rand

    exp_V_25_2010_rand        = exp.(V_25_2010_rand)
    total_exp_V_25_2010_rand  = sum(exp_V_25_2010_rand, dims=(2,3))
    P_25_10                   = exp_V_25_2010_rand./total_exp_V_25_2010_rand

    exp_V_50_2000_rand       = exp.(V_50_2000_rand)
    total_exp_V_50_2000_rand = sum(exp_V_50_2000_rand, dims=(2,3))
    P_50_00                  = exp_V_50_2000_rand ./ total_exp_V_50_2000_rand

    N, J, K       = size(P_25_00)
    choices_25_00 = zeros(Int, N)
    choices_25_10 = zeros(Int, N)
    choices_50_00 = zeros(Int, N)

    for i in 1:N
      individual_probs = [P_25_00[i,1,:]; P_25_00[i,2,:]]
      if all(iszero, individual_probs) || any(isnan, individual_probs) || sum(individual_probs) != 1.0
          # Handle the case where individual_probs is a vector of zeros
          choices_25_00[i] = 1  # Assign a default choice, e.g., 1
      else
          choice_distribution = Categorical(individual_probs)
          chosen_option = rand(choice_distribution)
          choices_25_00[i] = chosen_option
      end
    end

    for i in 1:N
      individual_probs = [P_25_10[i,1,:]; P_25_10[i,2,:]]
      if all(iszero, individual_probs) || any(isnan, individual_probs) || sum(individual_probs) != 1.0
          # Handle the case where individual_probs is a vector of zeros
          choices_25_10[i] = 1  
      else
          choice_distribution = Categorical(individual_probs)
          chosen_option = rand(choice_distribution)
          choices_25_10[i] = chosen_option
      end
    end

    for i in 1:N
      individual_probs = [P_50_00[i,1,:]; P_50_00[i,2,:]]
      if all(iszero, individual_probs) || any(isnan, individual_probs) || sum(individual_probs) != 1.0
          # Handle the case where individual_probs is a vector of zeros
          choices_50_00[i] = 1  
      else
          choice_distribution = Categorical(individual_probs)
          chosen_option = rand(choice_distribution)
          choices_50_00[i] = chosen_option
      end
    end

    Ic_25_2000   = vec(choices_25_00.<=4)
    Ic_25_2010   = vec(choices_25_10.<=4)
    Ic_50_2000   = vec(choices_50_00.<=4)

    zc_25_2000   = zc_base[Ic_25_2000 .==1]
    zr_25_2000   = zr_base[Ic_25_2000 .==0]

    zc_25_2010   = zc_base[Ic_25_2010 .==1]
    zr_25_2010   = zr_base[Ic_25_2010 .==0]

    zc_50_2000   = zc_base[Ic_50_2000 .==1]
    zr_50_2000   = zr_base[Ic_50_2000 .==0]

    mapping_c    = Dict(1 => 0, 2 => 6, 3 => 11, 4 => 16)
    ys_c_25_2000 = [mapping_c[choice] for choice in choices_25_00 if haskey(mapping_c, choice)]
    ys_c_25_2010 = [mapping_c[choice] for choice in choices_25_10 if haskey(mapping_c, choice)]
    ys_c_50_2000 = [mapping_c[choice] for choice in choices_50_00 if haskey(mapping_c, choice)]

    mapping_r    = Dict(5 => 0, 6 => 6, 7 => 11, 8 => 16)
    ys_r_25_2000 = [mapping_r[choice] for choice in choices_25_00 if haskey(mapping_r, choice)]
    ys_r_25_2010 = [mapping_r[choice] for choice in choices_25_10 if haskey(mapping_r, choice)]
    ys_r_50_2000 = [mapping_r[choice] for choice in choices_50_00 if haskey(mapping_r, choice)]

    ao_c_25_2000 = []
    ao  = nothing
    as  = nothing
    for i in 1:length(zc_25_2000)
      as = ys_c_25_2000[i] + a0
      ia = ih.(α_c,ηo_c,ρ,δ_c,gc,A,(as:1:A))./hs.(zc_25_2000[i],ηs_c,β_c,δ_c,(as:1:A),a0)
      ao = findfirst(x -> x < 1, ia) .- 1 .+ a0
      push!(ao_c_25_2000,ao)
    end

    ao_c_25_2010 = []
    ao  = nothing
    as  = nothing
    for i in 1:length(zc_25_2010)
      as = ys_c_25_2010[i] + a0
      ia = ih.(α_c,ηo_c,ρ,δ_c,gc,A,(as:1:A))./hs.(zc_25_2010[i],ηs_c,β_c,δ_c,(as:1:A),a0)
      ao = findfirst(x -> x < 1, ia) .- 1 .+ a0
      push!(ao_c_25_2010,ao)
    end

    ao_c_50_2000 = []
    ao  = nothing
    as  = nothing
    for i in 1:length(zc_50_2000)
      as = ys_c_50_2000[i] + a0
      ia = ih.(α_c,ηo_c,ρ,δ_c,gc,A,(as:1:A))./hs.(zc_50_2000[i],ηs_c,β_c,δ_c,(as:1:A),a0)
      ao = findfirst(x -> x < 1, ia) .- 1 .+ a0
      push!(ao_c_50_2000,ao)
    end

    ao_r_25_2000 = []
    ao  = nothing
    as  = nothing
    for i in 1:length(zr_25_2000)
      as = ys_r_25_2000[i] + a0
      ia = ih.(α_r,ηo_r,ρ,δ_r,gr,A,(as:1:A))./hs.(zr_25_2000[i],ηs_r,β_r,δ_r,(as:1:A),a0)
      ao = findfirst(x -> x < 1, ia) .- 1 .+ a0
      push!(ao_r_25_2000,ao)
    end

    ao_r_25_2010 = []
    ao  = nothing
    as  = nothing
    for i in 1:length(zr_25_2010)
      as = ys_r_25_2010[i] + a0
      ia = ih.(α_r,ηo_r,ρ,δ_r,gr,A,(as:1:A))./hs.(zr_25_2010[i],ηs_r,β_r,δ_r,(as:1:A),a0)
      ao = findfirst(x -> x < 1, ia) .- 1 .+ a0
      push!(ao_r_25_2010,ao)
    end

    ao_r_50_2000 = []
    ao  = nothing
    as  = nothing
    for i in 1:length(zr_50_2000)
      as = ys_r_50_2000[i] + a0
      ia = ih.(α_r,ηo_r,ρ,δ_r,gr,A,(as:1:A))./hs.(zr_50_2000[i],ηs_r,β_r,δ_r,(as:1:A),a0)
      ao = findfirst(x -> x < 1, ia) .- 1 .+ a0
      push!(ao_r_50_2000,ao)
    end

    hs_c_25_2000 = Hs.(zc_25_2000,β_c,α_c,ηs_c,ηo_c,ρ,δ_c,gc,A,a0,(ys_c_25_2000 .+ a0),25)
    hs_r_25_2000 = Hs.(zr_25_2000,β_r,α_r,ηs_r,ηo_r,ρ,δ_r,gr,A,a0,(ys_r_25_2000 .+ a0),25)  
    Ec_25_2000   = 1( (ys_c_25_2000 .+ a0) .<  ao_c_25_2000 ).*Eo.(wc_25_00,hs_c_25_2000,ηo_c,α_c,ρ,δ_c,gc,A,a0,(ys_c_25_2000 .+ a0),ao_c_25_2000,25,25) .+ 1( (ys_c_25_2000 .+ a0) .>= ao_c_25_2000 ).*Es.(wc_25_00,zc_25_2000,ηo_c,ηs_c,α_c,β_c,ρ,δ_c,gc,A,a0,(ys_c_25_2000 .+ a0),25,25)
    Er_25_2000   = 1( (ys_r_25_2000 .+ a0) .<  ao_r_25_2000 ).*Eo.(wr_25_00,hs_r_25_2000,ηo_r,α_r,ρ,δ_r,gr,A,a0,(ys_r_25_2000 .+ a0),ao_r_25_2000,25,25) .+ 1( (ys_r_25_2000 .+ a0) .>= ao_r_25_2000 ).*Es.(wr_25_00,zr_25_2000,ηo_r,ηs_r,α_r,β_r,ρ,δ_r,gr,A,a0,(ys_r_25_2000 .+ a0),25,25)
    
    hs_c_25_2010 = Hs.(zc_25_2010,β_c,α_c,ηs_c,ηo_c,ρ,δ_c,gc,A,a0,(ys_c_25_2010 .+ a0),25)
    hs_r_25_2010 = Hs.(zr_25_2010,β_r,α_r,ηs_r,ηo_r,ρ,δ_r,gr,A,a0,(ys_r_25_2010 .+ a0),25)  
    Ec_25_2010   = 1( (ys_c_25_2010 .+ a0) .<  ao_c_25_2010 ).*Eo.(wc_25_10,hs_c_25_2010,ηo_c,α_c,ρ,δ_c,gc,A,a0,(ys_c_25_2010 .+ a0),ao_c_25_2010,25,25) .+ 1( (ys_c_25_2010 .+ a0) .>= ao_c_25_2010 ).*Es.(wc_25_10,zc_25_2010,ηo_c,ηs_c,α_c,β_c,ρ,δ_c,gc,A,a0,(ys_c_25_2010 .+ a0),25,25)
    Er_25_2010   = 1( (ys_r_25_2010 .+ a0) .<  ao_r_25_2010 ).*Eo.(wr_25_10,hs_r_25_2010,ηo_r,α_r,ρ,δ_r,gr,A,a0,(ys_r_25_2010 .+ a0),ao_r_25_2010,25,25) .+ 1( (ys_r_25_2010 .+ a0) .>= ao_r_25_2010 ).*Es.(wr_25_10,zr_25_2010,ηo_r,ηs_r,α_r,β_r,ρ,δ_r,gr,A,a0,(ys_r_25_2010 .+ a0),25,25)
    
    hs_c_50_2000 = Hs.(zc_50_2000,β_c,α_c,ηs_c,ηo_c,ρ,δ_c,gc,A,a0,(ys_c_50_2000 .+ a0),50)
    hs_r_50_2000 = Hs.(zr_50_2000,β_r,α_r,ηs_r,ηo_r,ρ,δ_r,gr,A,a0,(ys_r_50_2000 .+ a0),50)  
    Ec_50_2000   = 1( (ys_c_50_2000 .+ a0) .<  ao_c_50_2000 ).*Eo.(wc_50_00,hs_c_50_2000,ηo_c,α_c,ρ,δ_c,gc,A,a0,(ys_c_50_2000 .+ a0),ao_c_50_2000,50,50) .+ 1( (ys_c_50_2000 .+ a0) .>= ao_c_50_2000 ).*Es.(wc_50_00,zc_50_2000,ηo_c,ηs_c,α_c,β_c,ρ,δ_c,gc,A,a0,(ys_c_50_2000 .+ a0),50,50)
    Er_50_2000   = 1( (ys_r_50_2000 .+ a0) .<  ao_r_50_2000 ).*Eo.(wr_50_00,hs_r_50_2000,ηo_r,α_r,ρ,δ_r,gr,A,a0,(ys_r_50_2000 .+ a0),ao_r_50_2000,50,50) .+ 1( (ys_r_50_2000 .+ a0) .>= ao_r_50_2000 ).*Es.(wr_50_00,zr_50_2000,ηo_r,ηs_r,α_r,β_r,ρ,δ_r,gr,A,a0,(ys_r_50_2000 .+ a0),50,50)
    
    ys_c_25_2000 = ys_c_25_2000[Ec_25_2000 .> 0.00]
    ys_r_25_2000 = ys_r_25_2000[Er_25_2000 .> 0.00]
    ys_c_25_2010 = ys_c_25_2010[Ec_25_2010 .> 0.00]
    ys_r_25_2010 = ys_r_25_2010[Er_25_2010 .> 0.00]
    ys_c_50_2000 = ys_c_50_2000[Ec_50_2000 .> 0.00]
    ys_r_50_2000 = ys_r_50_2000[Er_50_2000 .> 0.00]

    Ec_25_2000 = Ec_25_2000[Ec_25_2000 .> 0.00]
    Ec_25_2000 = isempty(Ec_25_2000) ? [Inf] : Ec_25_2000

    Er_25_2000 = Er_25_2000[Er_25_2000 .> 0.00]
    Er_25_2000 = isempty(Er_25_2000) ? [Inf] : Er_25_2000
    
    Ec_25_2010 = Ec_25_2010[Ec_25_2010 .> 0.00]
    Ec_25_2010 = isempty(Ec_25_2010) ? [Inf] : Ec_25_2010

    Er_25_2010 = Er_25_2010[Er_25_2010 .> 0.00]
    Er_25_2010 = isempty(Er_25_2010) ? [Inf] : Er_25_2010

    Ec_50_2000 = Ec_50_2000[Ec_50_2000 .> 0.00]
    Ec_50_2000 = isempty(Ec_50_2000) ? [Inf] : Ec_50_2000

    Er_50_2000 = Er_50_2000[Er_50_2000 .> 0.00]
    Er_50_2000 = isempty(Er_50_2000) ? [Inf] : Er_50_2000

    results = OrderedDict{String,Float64}()
    results["mean_log_Ec_25_2000"]          = round(mean(log.(Ec_25_2000)), digits=3) 
    results["mean_log_Er_25_2000"]          = round(mean(log.(Er_25_2000)), digits=3)
    results["mean_log_Ec_25_2010"]          = round(mean(log.(Ec_25_2010)), digits=3) 
    results["mean_log_Er_25_2010"]          = round(mean(log.(Er_25_2010)), digits=3)   
    results["std_log_Ec_25_2000"]           = round(std(log.(Ec_25_2000)), digits=3) 
    results["std_log_Er_25_2000"]           = round(std(log.(Er_25_2000)), digits=3) 
    results["mean_log_Ec_25_2000_ys_16"]    = isempty(ys_c_25_2000[ys_c_25_2000 .== 16]) ? Inf : round(mean(log.(Ec_25_2000[ys_c_25_2000 .== 16])), digits=3)
    results["mean_log_Er_25_2000_ys_16"]    = isempty(ys_r_25_2000[ys_r_25_2000 .== 16]) ? Inf : round(mean(log.(Er_25_2000[ys_r_25_2000 .== 16])), digits=3)
    results["mean_log_Ec_25_2000_ys_11"]    = isempty(ys_c_25_2000[ys_c_25_2000 .== 11]) ? Inf : round(mean(log.(Ec_25_2000[ys_c_25_2000 .== 11])), digits=3)
    results["mean_log_Er_25_2000_ys_11"]    = isempty(ys_r_25_2000[ys_r_25_2000 .== 11]) ? Inf : round(mean(log.(Er_25_2000[ys_r_25_2000 .== 11])), digits=3)
    results["mean_log_Ec_50_2000"]          = round(mean(log.(Ec_50_2000)), digits=3) 
    results["mean_log_Er_50_2000"]          = round(mean(log.(Er_50_2000)), digits=3)
    return results
  end
end

#--------------------------------------------------------------------------------
#SPECIFY DATA MOMENTS
@everywhere begin
  DataMoments = OrderedDict{String,Float64}()
  DataMoments["mean_log_Ec_25_2000"]   = 2.727       
  DataMoments["mean_log_Er_25_2000"]   = 2.420       
  DataMoments["mean_log_Ec_25_2010"]   = 3.103       
  DataMoments["mean_log_Er_25_2010"]   = 2.751  
  DataMoments["stdv_logEc_25_00"]      = 0.548
  DataMoments["stdv_logEr_25_00"]      = 0.526
  DataMoments["logEc_ys16"]            = 2.805
  DataMoments["logEr_ys16"]            = 2.619
  DataMoments["logEc_ys11"]            = 2.541
  DataMoments["logEr_ys11"]            = 2.401
  DataMoments["mean_log_Ec_50_2000"]   = 3.33 
  DataMoments["mean_log_Er_50_2000"]   = 3.15  
  #DataMoments["logEc_ys6"]             = 2.380
  #DataMoments["logEr_ys6"]             = 2.170
  #DataMoments["logEc_10_ys6"]          = 3.248
  #DataMoments["logEr_10_ys6"]          = 2.437
end

@everywhere begin
  #SPECIFY THE WEIGHTING MATRIX
  dm_vals  = collect(values(DataMoments))
  #W        =  (1 ./ abs.(dm_vals))
  W        =  1.00
end

@everywhere begin
  #DEFINE THE LOSS FUNCTION 
  function compute_loss(params,N,Πc_25_2000,Πr_25_2000,Πc_50_2000,Πr_50_2000,Πc_25_2010,Πr_25_2010,πc_25_2000,πc_25_2010,πc_50_2000,DataMoments)
    sim_mom  = collect(values(simulate_moms(params,N,Πc_25_2000,Πr_25_2000,Πc_50_2000,Πr_50_2000,Πc_25_2010,Πr_25_2010,πc_25_2000,πc_25_2010,πc_50_2000)))
    SSE      = sum( abs.(  (sim_mom .- values(DataMoments)) ).*W )
  end
end

#--------------------------------------------------------------
@everywhere begin
  #Educational Structure in 2000 (USA)
  Πc_25_2000 = OrderedDict("π_ns" => 0.007,"π_prim_2"  => 0.026,"π_prim_4" => 0.058,"π_prim_comp" => 0.135,"π_sec_2" => 0.063,"π_sec_comp" => 0.358,"π_uni_2"  => 0.164,"π_uni" => 0.188)
  Πr_25_2000 = OrderedDict("π_ns" => 0.050,"π_prim_2"  => 0.147,"π_prim_4" => 0.246,"π_prim_comp" => 0.266,"π_sec_2" => 0.068,"π_sec_comp" => 0.191,"π_uni_2"  => 0.023,"π_uni" => 0.010)
  Πc_50_2000 = OrderedDict("π_ns" => 0.020,"π_prim_2"  => 0.043,"π_prim_4" => 0.099,"π_prim_comp" => 0.095,"π_sec_2" => 0.031,"π_sec_comp" => 0.224,"π_uni_2"  => 0.075,"π_uni" => 0.414)
  Πr_50_2000 = OrderedDict("π_ns" => 0.166,"π_prim_2"  => 0.248,"π_prim_4" => 0.320,"π_prim_comp" => 0.137,"π_sec_2" => 0.022,"π_sec_comp" => 0.081,"π_uni_2"  => 0.009,"π_uni" => 0.018)

  #Educational Structure in 2010
  Πc_25_2010 = OrderedDict("π_ns" => 0.006,"π_prim_2"  => 0.019,"π_prim_4" => 0.009,"π_prim_comp" => 0.068,"π_sec_2" => 0.051,"π_sec_comp" => 0.275,"π_uni_2"  => 0.222,"π_uni" => 0.349)
  Πr_25_2010 = OrderedDict("π_ns" => 0.015,"π_prim_2"  => 0.092,"π_prim_4" => 0.057,"π_prim_comp" => 0.249,"π_sec_2" => 0.124,"π_sec_comp" => 0.368,"π_uni_2"  => 0.063,"π_uni" => 0.032)

  πc_25_2000 = 0.302
  πc_50_2000 = 0.399
  πc_25_2010 = 0.328
end
#--------------------------------------------------------------
@everywhere begin
  N = 1000
  f_SSE(params) = compute_loss(params,N,Πc_25_2000,Πr_25_2000,Πc_50_2000,Πr_50_2000,Πc_25_2010,Πr_25_2010,πc_25_2000,πc_25_2010,πc_50_2000,DataMoments)                                              
end

@everywhere begin
  ParamSpace    = [(0.01,5.00),       #σ_cc
                   (0.01,5.00),       #σ_rr
                  (0.05 ,12.00),      #wc                
                  (0.05 ,12.00),      #wr
                  (0.010,0.175),      #ηo_c 
                  (0.010,0.175),      #ηo_r                        
                  (0.01 ,0.50),       #α_c                
                  (0.01 ,0.50),       #α_r
                  (0.180 ,0.50),      #ηs_c 
                  (0.180 ,0.50),      #ηs_r        
                  (0.05 ,0.50),       #β_c             
                  (0.05 ,0.50),       #β_r
                  (0.0001,0.025),     #gc             
                  (0.0001,0.025)]     #gr
end


#Best candidate found: [1.32572, 1.27935, 1.12539, 3.58418, 0.0940523, 0.108794, 0.269216, 0.205891, 0.763845, 0.260334, 0.425676, 0.358607]

#=
@everywhere begin
  x0_1 = [0.668284, 0.10000, 2.350, 7.500, 0.1032, 0.080145, 0.27000, 0.24000, 0.40000, 0.30000, 0.0180, 0.0075]
  x0_2 = [0.119116, 0.436606, 7.14199, 3.24426, 0.0719985, 0.0250895, 0.0304356, 0.0902823, 0.16014, 0.368885, 0.290427, 0.203623, 0.00386362, 0.0192517]
  x0_3 = [0.119116, 0.436606, 7.14199, 3.24426, 0.0719985, 0.0250895, 0.0304356, 0.0902823, 0.16014, 0.368885, 0.290427, 0.203623, 0.00386362, 0.0192517]
  x0_4 = [0.259856, 0.575783, 8.75284, 5.50769, 0.0423331, 0.0256126, 0.0383456, 0.0362173, 0.186142, 0.158567, 0.320196, 0.413851, 0.0145268, 0.0061316]
  x0_5 = [0.13099, 1.11802, 9.53921, 3.43716, 0.139539, 0.111527, 0.212741, 0.0650087, 0.150087, 0.172455, 0.446242, 0.138868, 0.0199314, 0.00902507]
  x0_6 = [0.345916, 0.846632, 7.85866, 4.24132, 0.093411, 0.137842, 0.283302, 0.206142, 0.318091, 0.31171, 0.122788, 0.241492, 0.017149, 0.0120245]
  x0_7 = [1.63806, 0.0368019, 6.07131, 4.61988, 0.131602, 0.132433, 0.416256, 0.400466, 0.285924, 0.406623, 0.460711, 0.440975, 0.00317352, 0.00984838]
  sleep(180)
el_res_para_5 = @elapsed res_para_4  = bboptimize(opt_problem_distri,MaxFuncEvals = 1200)
  
  x0   = [x0_1,x0_2,x0_3,x0_4,x0_5,x0_6,x0_7] 
end
=#
@everywhere begin
  x0_1 = [0.793034, 0.637955, 9.92422, 6.30334, 0.0820409, 0.151018, 0.349079, 0.178861, 0.259658, 0.201212, 0.0686504, 0.316916, 0.0168809, 0.00141555]     
  x0_2 = [0.668284, 0.10000, 2.350, 7.500, 0.1032, 0.080145, 0.27000, 0.24000, 0.40000, 0.30000, 0.0180, 0.0075, 0.0168809, 0.00141555] 
  x0_3 = [0.854889, 2.94503, 3.78461, 3.29875, 0.160988, 0.0951782, 0.221612, 0.405409, 0.49114, 0.364121, 0.201609, 0.390706, 0.0116729, 0.0152163]
  x0   = [x0_1,x0_2,x0_3] 
end


#--------------------------------------------------------------------------------
#RUN THE OPTIMIZATION PROCEDURE TO FIND THE PARAMETERS
@everywhere begin    
  #opt_problem_norm   =  bbsetup(f_SSE;x0,SearchRange=ParamSpace,TraceMode=:verbose,NumDimensions=12,MaxFuncEvals=10000,Method=:adaptive_de_rand_1_bin_radiuslimited)
  opt_problem_distri =  bbsetup(f_SSE;x0,SearchRange=ParamSpace,TraceMode=:verbose,NumDimensions=14,MaxFuncEvals=2000,Method=:adaptive_de_rand_1_bin_radiuslimited,Workers = workers())
  #opt_problem_distri =  bbsetup(f_SSE;x0,SearchRange=ParamSpace,TraceMode=:verbose,NumDimensions=12,MaxFuncEvals=10000,Method=:adaptive_de_rand_1_bin_radiuslimited)
end
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
#:adaptive_de_rand_1_bin_radiuslimited
el_res_para_0 = @elapsed res_para_0  = bboptimize(opt_problem_distri,MaxFuncEvals = 25)
sleep(180)
el_res_para_1 = @elapsed res_para_1  = bboptimize(opt_problem_distri,MaxFuncEvals = 90)
sleep(180)
el_res_para_2 = @elapsed res_para_2  = bboptimize(opt_problem_distri,MaxFuncEvals = 350)
sleep(180)
el_res_para_3 = @elapsed res_para_3  = bboptimize(opt_problem_distri,MaxFuncEvals = 430)
sleep(180)
el_res_para_4 = @elapsed res_para_4  = bboptimize(opt_problem_distri,MaxFuncEvals = 800)
el_res_para_5 = @elapsed res_para_4  = bboptimize(opt_problem_distri,MaxFuncEvals = 1000)
sleep(180)
el_res_para_6 = @elapsed res_para_6  = bboptimize(opt_problem_distri,MaxFuncEvals = 1200)
sleep(180)
el_res_para_7 = @elapsed res_para_7  = bboptimize(opt_problem_distri,MaxFuncEvals = 1400)
sleep(180)
el_res_para_8 = @elapsed res_para_8  = bboptimize(opt_problem_distri,MaxFuncEvals = 1600)

#el_res_para_3 = @elapsed res_para_3  = bboptimize(opt_problem_distri,MaxFuncEvals = 160)
#el_res_para_4 = @elapsed res_para_4  = bboptimize(opt_problem_distri,MaxFuncEvals = 200)
#el_res_para_5 = @elapsed res_para_5  = bboptimize(opt_problem_distri,MaxFuncEvals = 260)
#el_res_para_6 = @elapsed res_para_6  = bboptimize(opt_problem_distri,MaxFuncEvals = 300)
#el_res_para_6 = @elapsed res_para_6  = bboptimize(opt_problem_distri,MaxFuncEvals = 400)
#el_res_para_7 = @elapsed res_para_7  = bboptimize(opt_problem_distri,MaxFuncEvals = 500)
#el_res_para_8 = @elapsed res_para_8  = bboptimize(opt_problem_distri,MaxFuncEvals = 700)

el_res_para_9 = @elapsed res_para_9  = bboptimize(opt_problem_distri,MaxFuncEvals = 900)
#el_res_para_91 = @elapsed res_para_91  = bboptimize(opt_problem_distri,MaxFuncEvals = 1100)
#el_res_para_92 = @elapsed res_para_92  = bboptimize(opt_problem_distri,MaxFuncEvals = 1300)
#el_res_para_93 = @elapsed res_para_93  = bboptimize(opt_problem_distri,MaxFuncEvals = 1500)
#el_res_para_94 = @elapsed res_para_94  = bboptimize(opt_problem_distri,MaxFuncEvals = 1700)
#el_res_para_95 = @elapsed res_para_95  = bboptimize(opt_problem_distri,MaxFuncEvals = 1900)

el_res_para_10 = @elapsed res_para_10  = bboptimize(opt_problem_distri,MaxFuncEvals = 2000)

#el_res_para_97 = @elapsed res_para_97  = bboptimize(opt_problem_distri,MaxFuncEvals = 2200)
#el_res_para_98 = @elapsed res_para_98  = bboptimize(opt_problem_distri,MaxFuncEvals = 2150)
#el_res_para_10 = @elapsed res_para_10  = bboptimize(opt_problem_distri,MaxFuncEvals = 2500)
#sleep(60*5)
#el_res_para_11 = @elapsed res_para_11  = bboptimize(opt_problem_distri,MaxFuncEvals = 2750)
#sleep(60*5)
#el_res_para_12 = @elapsed res_para_12  = bboptimize(opt_problem_distri,MaxFuncEvals = 3000)
#sleep(60*5)
#el_res_para_13 = @elapsed res_para_13  = bboptimize(opt_problem_distri,MaxFuncEvals = 3250)
#sleep(60*5)
#el_res_para_14 = @elapsed res_para_14  = bboptimize(opt_problem_distri,MaxFuncEvals = 3050)
#sleep(60*5)
#el_res_para_15 = @elapsed res_para_15  = bboptimize(opt_problem_distri,MaxFuncEvals = 3250)
#sleep(60*5)
#el_res_para_15 = @elapsed res_para_15  = bboptimize(opt_problem_distri,MaxFuncEvals = 3750)
#sleep(60*5)
#el_res_para_15 = @elapsed res_para_15  = bboptimize(opt_problem_distri,MaxFuncEvals = 4000)
#sleep(60*5)
#el_res_para_16 = @elapsed res_para_16  = bboptimize(opt_problem_distri,MaxFuncEvals = 4500)
#sleep(60*5)
#el_res_para_17 = @elapsed res_para_17  = bboptimize(opt_problem_distri,MaxFuncEvals = 4500)
#sleep(60*5)
#el_res_para_18 = @elapsed res_para_18  = bboptimize(opt_problem_distri,MaxFuncEvals = 5000)
#sleep(60*5)
#el_res_para_19 = @elapsed res_para_19  = bboptimize(opt_problem_distri,MaxFuncEvals = 5500)
#sleep(60*5)
#el_res_para_20 = @elapsed res_para_20  = bboptimize(opt_problem_distri,MaxFuncEvals = 5670)

el_res_para_21 = @elapsed res_para_21  = bboptimize(opt_problem_distri,MaxFuncEvals = 6000)

@info "Finished computation. Best Candidate: " best_candidate(res_para_10)
@info "Finished computation. Best Fitness: ", best_fitness(res_para_10)


results = Dict(
    :params       => best_candidate(res_para_10),
    :fitness      => best_fitness(res_para_10),
    :compute_time => el_res_para_10
)

open("results.json", "w") do io
    JSON3.pretty(io, results)
end

ENV["RESULTS"] = JSON3.write(results)
ENV["RESULTS_FILE"] = "results.json"

#Optimization stopped after 59 steps and 5323.19 seconds
#Total function evaluations = 101
#Best candidate found: [1.13047, 0.701001, 2.98798, 5.57137, 0.149266, 0.0659309, 0.357215, 0.435029, 0.421228, 0.384372, 0.579461, 0.330578, 0.00280923, 0.0240709]

#Improvements/step = Inf
#Total function evaluations = 42
#Best candidate found: [0.119116, 0.436606, 7.14199, 3.24426, 0.0719985, 0.0250895, 0.0304356, 0.0902823, 0.16014, 0.368885, 0.290427, 0.203623, 0.00386362, 0.0192517]
#Fitness: 3.989000000
#971.8948639

#Total function evaluations = 41
#Best candidate found: [1.34005, 0.63251, 8.1383, 7.91541, 0.137548, 0.0404029, 0.243291, 0.429979, 0.374904, 0.170433, 0.372375, 0.449883, 0.00656405, 0.00803516]        
#Fitness: 6.289000000
#945.0906414

#Total function evaluations = 1301
#Best candidate found: [0.158083, 1.17033, 8.75284, 5.50769, 0.0423331, 0.0162293, 0.0383456, 0.12719, 0.155837, 0.158567, 0.109492, 0.174193, 0.00742637, 0.00464812]
#Fitness: 2.465000000
#7456.1582823

#Total function evaluations = 1501
#Best candidate found: [0.804304, 0.535822, 4.82941, 5.1334, 0.131756, 0.0443801, 0.0259875, 0.0403372, 0.232735, 0.179389, 0.0540461, 0.194248, 0.00643722, 0.0136332]
#Fitness: 2.348000000
#6550.8437138

#Total function evaluations = 1701
#Best candidate found: [0.804304, 0.535822, 4.82941, 5.1334, 0.131756, 0.0443801, 0.0259875, 0.0403372, 0.232735, 0.179389, 0.0540461, 0.194248, 0.00643722, 0.0136332]
#Fitness: 2.348000000
#6819.0923723

#Total function evaluations = 1901
#Best candidate found: [0.0903731, 0.689334, 7.17721, 4.25469, 0.0888119, 0.0537701, 0.0670776, 0.0261428, 0.191014, 0.16513, 0.054402, 0.199067, 0.00649658, 0.00163299]
#Fitness: 2.084000000
#6786.6676369

#Total function evaluations = 2151
#Best candidate found: [0.134824, 1.12516, 9.02693, 3.58708, 0.143124, 0.0688218, 0.212381, 0.045744, 0.15224, 0.202697, 0.433269, 0.152533, 0.0185587, 0.014037]
#Fitness: 1.833000000
#480.2746346

#Best candidate found: [0.13099, 1.11802, 9.53921, 3.43716, 0.139539, 0.111527, 0.212741, 0.0650087, 0.150087, 0.172455, 0.446242, 0.138868, 0.0199314, 0.00902507]
#Fitness: 1.711000000
#9604.3131978

☺#Total function evaluations = 5671
☺#Best candidate found: [0.13099, 1.11802, 9.53921, 3.43716, 0.139539, 0.111527, 0.212741, 0.0650087, 0.150087, 0.172455, 0.446242, 0.138868, 0.0199314, 0.00902507]
☺#Fitness: 1.711000000
  
#--------------------------------------------------------------------------------
#After fixing the seedTotal function evaluations = 251

#Best candidate found: [1.63806, 0.0368019, 6.07131, 4.61988, 0.131602, 0.132433, 0.416256, 0.400466, 0.285924, 0.406623, 0.460711, 0.440975, 0.00317352, 0.00984838]      
#Fitness: 3.595000000
#7459.4824074