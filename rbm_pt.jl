using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using StatsFuns: logistic



# ==============================================================================
#  CONFIGURATION
# ==============================================================================

const Binary = 0          # 1 → Ising spins {-1,+1};  0 → binary units {0,1}

# const WEIGHTS_FILE = "Ising2D"
# const WEIGHTS_FILE = "Ising3D"
# const WEIGHTS_FILE = "EA3D"
# const WEIGHTS_FILE = "BSK"
const WEIGHTS_FILE = "Weights/GWGM_weights.dat"
# const WEIGHTS_FILE = "Weights/Rows_weights.dat"
# const WEIGHTS_FILE = "Weights/MNIST20h_weights.dat"
# const WEIGHTS_FILE = "Weights/MNIST500h_weights.dat"

# Simulation parameters
const NSim     = 1          # Number of independent simulations
const NTherm   = 1024       # Thermalization sweeps
const NSampl   = 512        # Number of independent samples to collect
const τ_sampl  = 1024       # Autocorrelation time (sweeps between samples)
const τ_swap   = 1          # Swaps between samples
const vis_only = false      # Keep only the visible array or keep the hole v-h space
const SAMPLER  = :cd1       # Sampler choice: :mh (Metropolis–Hastings) or :cd1 (Contrastive Divergence-1)

const j = 1.f0            # Ising interaction
const L = 12              # Ising lattice size
const p = 0.5             # EA probability of setting a positive interaction

const σ = 1.f0            # SK std
const μ = 0               # SK mean
const Lv = 64             # SK number of visibles
const Lh = 64             # SK number of hiddens


# Temperature grid
const β_vec = Float32.(collect(0.1:0.1:1.7))
# const β_vec = 1 ./ Float32.(collect(0.05:0.05:2.5))
# const β_vec = Float32.([collect(0.002:0.002:0.01); [0.03, 0.05]; collect(0.1:0.1:0.8); collect(0.85:0.05:1.2); collect(1.3:0.1:1.7)])
# const β_vec = Float32.(collect(0.5:0.1:2.5))
# const β_vec = Float32.([collect(0.5:0.1:0.85); collect(0.9:0.02:1.1); collect(1.12:0.05:1.40); collect(1.42:0.02:1.62); collect(1.7:0.1:3.0); collect(3.5:0.5:10.0)])





# ==============================================================================
#  LATTICE → RBM TRANSFORMATION
# ==============================================================================

"""
    Ising2D_to_RBM(j, L) → W

Given the Nearest-neighbour interaction between particles, 
this function returns a RBM weights matrix with a 2D-Ising like interaction

- j == Interaction strength (assumed the same for every particle)
- L == Size of the lattice, such that L² = N
- Returns the extended weights matrix W. Bias (b, c) are set to zero
"""
function Ising2D_to_RBM(j::Float32, h::Float32, L::Int)
    N  = L^2;  Nv = N÷2;  Nh = N÷2
    m  = zeros(Float32, Nv+1, Nh+1)
    J  = zeros(Float32, Nv, Nh)
    for ii in 1:Nv
        # Row where we are
        n   = 1 + (2*(ii-1)) ÷ L

        # Left and right particles
        j_l = ii - 1 + (1 - mod(n,2));  j_r = ii + (1 - mod(n,2))
        # PBC
        if j_l <= (n-1)*L÷2  j_l += L÷2  end
        if j_r >   n   *L÷2  j_r -= L÷2  end

        # Top and bottom particles
        j_b = ii + L÷2;  j_t = ii - L÷2
        # PBC
        if j_b >  Nv  j_b -= Nv  end
        if j_t <= 0   j_t += Nv  end

        # Put the corresponding values to the weights matrix
        J[ii, [j_l,j_r]] .= j
        J[ii, [j_b,j_t]] .= j
    end
    m[2:end,2:end] = J
    m[2:end, 1]   .= h
    m[1, 2:end]   .= h
    return m
end

"""
    Ising3D_to_RBM(j, L) → W

Given the Nearest-neighbour interaction between particles, 
this function returns a RBM weights matrix with a 3D-Ising like interaction

- j == Interaction strength (assumed the same for every particle)
- L == Size of the lattice, such that L² = N
- Returns the extended weights matrix W. Bias (b, c) are set to zero
"""
function Ising3D_to_RBM(j::Float32, h::Float32, L::Int)
    N  = L^3;  Nv = N÷2;  Nh = N÷2
    m  = zeros(Float32, Nv+1, Nh+1)
    J  = zeros(Float32, Nv, Nh)
    for ii in 1:Nv
        # Row where we are
        n   = 1 + (2*(ii-1)) ÷ L
        # Layer where we are
        l   = 1 + (2*(ii-1)) ÷ (L^2)

        # Left and right particles
        j_l = ii   - mod(n+l,2);  j_r = ii+1 - mod(n+l,2)
        # PBC
        if j_l <= (n-1)*L÷2  j_l += L÷2  end
        if j_r >   n   *L÷2  j_r -= L÷2  end

        # Top and bottom particles
        j_b = ii + L÷2;  j_t = ii - L÷2
        # PBC
        if j_b >  l*(L^2÷2)      j_b -= L^2÷2  end
        if j_t <= (l-1)*(L^2÷2)  j_t += L^2÷2  end

        # Up and down particles
        j_u = ii + L^2÷2;  j_d = ii - L^2÷2
        # PBC
        if j_u >  Nv  j_u -= Nv  end
        if j_d <= 0   j_d += Nv  end

        # Put the corresponding values to the weights matrix
        J[ii, [j_l,j_r]] .= j
        J[ii, [j_b,j_t]] .= j
        J[ii, [j_u,j_d]] .= j
    end
    m[2:end,2:end] = J
    m[2:end, 1]   .= h
    m[1, 2:end]   .= h
    return m
end


# ==============================================================================
#  ENERGY (single replica, used for initialisation and PT swaps)
# ==============================================================================

"""
    replica_energy(v, h, W, k) → scalar

Physical energy  E = vᵀ W h  for replica column k.
(All arrays include the bias row/column at index 1.)
"""
@inline function replica_energy(v::AbstractMatrix, h::AbstractMatrix,
                                W::AbstractMatrix, k::Int)
    # dot over the physical (non-bias) rows/columns only
    s = 0.0f0
    @inbounds for j in 1:size(h,1), i in 1:size(v,1)
        s += v[i,k] * W[i,j] * h[j,k]
    end
    return s          # physical energy  E = vᵀWh
end



# ==============================================================================
#  METROPOLIS–HASTINGS SWEEP  (one call = N single-spin proposals)
# ==============================================================================

"""
    mh_sweep!(v, h, W, β_vec, bufs)

One MH sweep over all replicas simultaneously.  Units are visited in sequential
order (systematic sweep), which gives better coverage per call than random picks.

`bufs` is a named-tuple of pre-allocated work arrays (see `make_buffers`).
"""
function mh_sweep!(v::Matrix{Float32}, h::Matrix{Float32},
                   W::Matrix{Float32}, β_vec::Vector{Float32}, bufs)

    NTemp = size(v, 2)
    Nv    = size(v, 1) - 1
    Nh    = size(h, 1) - 1
    ΔE    = bufs.ΔE
    rands = bufs.rands

    # ---------- visible units ----------
    @inbounds for i in 2:Nv+1
        # ΔE_k = (W[i,:] · h[:,k]) · (2 v[i,k] - 1 + Binary)
        #   → accept if log(u) < -β · ΔE   (Metropolis criterion)
        rand!(rands)   # fill NTemp uniform random numbers in-place
        for k in 1:NTemp
            field_k = 0.0f0
            for j in 1:Nh+1
                field_k += W[i,j] * h[j,k]
            end
            ΔE_k = -field_k * (2f0*v[i,k] - 1f0 + Float32(Binary))
            if log(rands[k]) < β_vec[k] * ΔE_k
                v[i,k] = Float32(1 - Binary) - v[i,k]
            end
        end
    end

    # ---------- hidden units ----------
    @inbounds for j in 2:Nh+1
        rand!(rands)
        for k in 1:NTemp
            field_k = 0.0f0
            for i in 1:Nv+1
                field_k += v[i,k] * W[i,j]
            end
            ΔE_k = -field_k * (2f0*h[j,k] - 1f0 + Float32(Binary))
            if log(rands[k]) < β_vec[k] * ΔE_k
                h[j,k] = Float32(1 - Binary) - h[j,k]
            end
        end
    end

    return nothing
end


# ==============================================================================
#  CD-1 SWEEP
# ==============================================================================

"""
    cd1_sweep!(v, h, W, β_vec, bufs)

One step of block Gibbs sampling (CD-1) for all replicas simultaneously.

For each replica k the update is:
    p(h_j = 1 | v, β) = σ( β_k · (1+Binary) · W[:,j]ᵀ v )
    p(v_i = 1 | h, β) = σ( β_k · (1+Binary) · W[i,:] · h )

The factor (1+Binary) makes the logistic argument correct for both
Ising ({-1,+1}, scale=2) and binary ({0,1}, scale=1) units.

The bias unit (index 1) is kept fixed at 1 throughout.
"""
function cd1_sweep!(v::Matrix{Float32}, h::Matrix{Float32},
                    W::Matrix{Float32}, β_vec::Vector{Float32}, bufs)

    # Keep the number of visible and the number of hiddens
    Nv, Nh = size(W) .- 1
    Nsampl = size(v, 2)

    # Acceptance probability of the hidden array
    ph_acc = logistic.((1 .+ Binary)*(W'v .* β_vec')); ph_acc[1, :] .= 1
    uh = rand(Nh+1, Nsampl)
    h[uh .<  ph_acc] .= 1
    h[uh .>= ph_acc] .= -1*Binary

    # Acceptance probability of the visible array
    pv_acc = logistic.((1 .+ Binary)*((W)h .* β_vec')); pv_acc[1, :] .= 1
    uv = rand(Nv+1, Nsampl)
    v[uv .<  pv_acc] .= 1
    v[uv .>= pv_acc] .= -1*Binary


    return nothing
end


# ==============================================================================
#  PARALLEL TEMPERING SWAP
# ==============================================================================

"""
    pt_swap!(v, h, Ene, β_vec, acc_count)

One sequential pass of nearest-neighbour replica exchanges.
Updates `Ene` in-place after each accepted swap.
`acc_count[k]` is incremented whenever the swap between replicas k and k+1
is accepted (for diagnostics).
"""
function pt_swap!(v::Matrix{Float32}, h::Matrix{Float32},
                  Ene::Vector{Float32}, β_vec::Vector{Float32},
                  acc_count::Vector{Int})

    NTemp = length(β_vec)
    @inbounds for k in 1:NTemp-1
        # Correct PT criterion: detailed balance requires
        #   Δ = (β_{k+1} - β_k) · (E_k - E_{k+1})
        # where E = vᵀWh (physical energy, already stored with this sign).
        # Accept if log(u) < Δ.
        Δ = (Ene[k] - Ene[k+1])*(β_vec[k+1] - β_vec[k])
        if log(rand()) < Δ
            # swap columns k and k+1
            v[:,k], v[:,k+1] = v[:,k+1], v[:,k]
            h[:,k], h[:,k+1] = h[:,k+1], h[:,k]
            Ene[k], Ene[k+1] = Ene[k+1], Ene[k]
            acc_count[k] += 1
        end
    end
    return nothing
end


# ==============================================================================
#  ENERGY VECTOR (all replicas at once, used at initialisation and diagnostics)
# ==============================================================================

"""
    compute_energies!(Ene, v, h, W)

Fill `Ene[k]` = physical energy of replica k = vᵀWh.
"""
function compute_energies!(Ene::Vector{Float32},
                           v::Matrix{Float32}, h::Matrix{Float32},
                           W::Matrix{Float32})
    NTemp = size(v, 2)
    @inbounds for k in 1:NTemp
        Ene[k] = replica_energy(v, h, W, k)
    end
    return nothing
end



# ==============================================================================
#  OBSERVABLES
# ==============================================================================

"""
    magnetization(v) → Vector{Float32}

Per-replica absolute magnetisation of the visible layer (Ising convention).
Works for both Binary=1 (spins ∈ {-1,+1}) and Binary=0 (units ∈ {0,1}).
"""
function magnetization(v::Matrix{Float32})
    Nv    = size(v, 1)
    NTemp = size(v, 2)
    m = Vector{Float32}(undef, NTemp)
    @inbounds for k in 1:NTemp
        s = 0.0f0
        for i in 1:Nv
            s += v[i,k]
        end
        m[k] = abs(s) / Nv
    end
    return m
end

"""
    overlap(v1, v2) → Vector{Float32}

Edwards–Anderson overlap q = |∑_i σ¹_i σ²_i| / Nv for each replica.
Always computed in Ising convention (±1); conversion applied internally.
"""
function overlap(v1::Matrix{Float32}, v2::Matrix{Float32})
    Nv    = size(v1, 1)
    NTemp = size(v1, 2)
    q = Vector{Float32}(undef, NTemp)
    scale = Float32(Binary == 1 ? 1 : 2)   # {0,1} → {-1,+1}: 2x-1
    offset = Float32(Binary == 1 ? 0 : -1)
    @inbounds for k in 1:NTemp
        s = 0.0f0
        for i in 2:Nv+1
            s1 = v1[i,k] * scale + offset
            s2 = v2[i,k] * scale + offset
            s += s1 * s2
        end
        q[k] = abs(s) / Nv
    end
    return q
end

# ==============================================================================
#  PRE-ALLOCATED BUFFERS
# ==============================================================================

make_buffers(NTemp::Int) = (
    ΔE    = Vector{Float32}(undef, NTemp),
    rands = Vector{Float64}(undef, NTemp),
)


# ==============================================================================
#  INITIALISATION HELPERS
# ==============================================================================

function init_replicas(Nv::Int, Nh::Int, NTemp::Int)
    spin_vals = Binary == 0 ? Float32[0, 1] : Float32[-1, 1]
    v = vcat(ones(Float32, 1, NTemp),
             Float32.(rand(spin_vals, Nv, NTemp)))
    h = vcat(ones(Float32, 1, NTemp),
             Float32.(rand(spin_vals, Nh, NTemp)))
    return v, h
end

# ==============================================================================
#  MAIN SIMULATION
# ==============================================================================

function run_simulation(W, β_vec;
                        NSim=2, NTherm=2048, NSampl=512, τ_sampl=100, sampler=:mh, vis_only=false)

    Nv, Nh = size(W) .- 1
    NTemp   = length(β_vec)

    # Storage for final results
    magne_mean = zeros(Float32, NTemp, NSim)
    qEA_mean   = zeros(Float32, NTemp, NSim)
    magne_std  = zeros(Float32, NTemp, NSim)
    qEA_std    = zeros(Float32, NTemp, NSim)

    # Choose sweep function at compile time (no runtime branch in hot loop)
    sweep! = sampler == :cd1 ? cd1_sweep! : mh_sweep!
    sampler_name = sampler == :cd1 ? "CD-1 (block Gibbs)" : "Metropolis–Hastings"


    # Swap acceptance counters (diagnostic)
    acc1 = zeros(Int, NTemp-1)
    acc2 = zeros(Int, NTemp-1)
    total_swaps = 0

    println("## Parallel Tempering — RBM thermodynamics")
    println("# Sampler:              $(sampler_name)")
    println("# Nv=$(Nv)  Nh=$(Nh)  NTemp=$(NTemp)  NTherm=$(NTherm)  NSampl=$(NSampl)  τSamp=$(τ_sampl)  τSwap=$(τ_swap)")
    println("# vis_only=$(vis_only)  Binary=$(Binary)")
    flush(stdout)

    bufs1 = make_buffers(NTemp)
    bufs2 = make_buffers(NTemp)

    Ene1 = Vector{Float32}(undef, NTemp)
    Ene2 = Vector{Float32}(undef, NTemp)
    
    for isim in 1:NSim
        println("\n# === Simulation $(isim)/$(NSim) ==="); flush(stdout)

        magne_all = zeros(Float32, NTemp, NSampl)
        qEA_all   = zeros(Float32, NTemp, NSampl)

        # ---- Initialise two independent replica sets ----
        v1, h1 = init_replicas(Nv, Nh, NTemp)
        v2, h2 = init_replicas(Nv, Nh, NTemp)

        compute_energies!(Ene1, v1, h1, W)
        compute_energies!(Ene2, v2, h2, W)

        fill!(acc1, 0); fill!(acc2, 0)
        total_swaps = 0

        # ---- Thermalization ----
        for ii in 1:NTherm
            for _ in 1:τ_swap
                sweep!(v1, h1, W, β_vec, bufs1)
                sweep!(v2, h2, W, β_vec, bufs2)
            end

            compute_energies!(Ene1, v1, h1, W)
            compute_energies!(Ene2, v2, h2, W)

            pt_swap!(v1, h1, Ene1, β_vec, acc1)
            pt_swap!(v2, h2, Ene2, β_vec, acc2)
            total_swaps += 1
        end

        println("# Thermalization done.")
        swap_rates = (acc1 .+ acc2) ./ (2 * max(total_swaps, 1))
        println("# Swap acceptance rates (min/mean/max): ",
                round(minimum(swap_rates), digits=3), " / ",
                round(mean(swap_rates),    digits=3), " / ",
                round(maximum(swap_rates), digits=3))
        flush(stdout)

        # ---- Sampling ----
        fill!(acc1, 0); fill!(acc2, 0)
        total_swaps = 0
        for isample in 1:NSampl
            # τ sweeps between samples
            for _ in 1:τ_sampl
                for _ in 1:τ_swap
                    sweep!(v1, h1, W, β_vec, bufs1)
                    sweep!(v2, h2, W, β_vec, bufs2)
                end

                compute_energies!(Ene1, v1, h1, W)
                compute_energies!(Ene2, v2, h2, W)

                pt_swap!(v1, h1, Ene1, β_vec, acc1)
                pt_swap!(v2, h2, Ene2, β_vec, acc2)
                total_swaps += 1
            end

            # Record observables: average the two replica sets to reduce variance
            if vis_only  x1 = v1[2:end,:];                x2 = v2[2:end,:]
            else         x1 = [v1[2:end,:]; h1[2:end,:]]; x2 = [v2[2:end,:]; h2[2:end,:]]
            end
            m1 = magnetization(x1)
            m2 = magnetization(x2)
            magne_all[:, isample] = (m1 .+ m2) ./ 2

            q = overlap(x1, x2)
            qEA_all[:, isample] = q

            if mod(isample, max(NSampl÷10, 1)) == 0
                println("# Sample $(isample)/$(NSampl)"); flush(stdout)
            end
        end

        # ---- Statistics ----
        magne_mean[:, isim] = vec(mean(magne_all, dims=2))
        magne_std[:,  isim] = vec(std( magne_all, dims=2))
        qEA_mean[:,   isim] = vec(mean(qEA_all,   dims=2))
        qEA_std[:,    isim] = vec(std( qEA_all,   dims=2))

        # ---- Per-simulation output ----
        swap_rates = (acc1 .+ acc2) ./ (2 * max(total_swaps, 1))
        println("# Sampling swap rates (min/mean/max): ",
                round(minimum(swap_rates), digits=3), " / ",
                round(mean(swap_rates),    digits=3), " / ",
                round(maximum(swap_rates), digits=3))

        print("# β:     "); for x in β_vec                     print(x, "\t") end; println()
        print("# m:     "); for x in magne_mean[:, isim]       print(round(x, digits=4), "\t") end; println()
        print("# m_std: "); for x in magne_std[:,  isim]       print(round(x, digits=4), "\t") end; println()
        print("# qEA:   "); for x in qEA_mean[:,   isim]       print(round(x, digits=4), "\t") end; println()
        print("# qEA_std: "); for x in qEA_std[:,  isim]       print(round(x, digits=4), "\t") end; println()
        flush(stdout)
    end

    return β_vec, magne_mean, magne_std, qEA_mean, qEA_std
end


# ==============================================================================
#  ENTRY POINT
# ==============================================================================


# println(WEIGHTS_FILE)
if WEIGHTS_FILE == "Ising2D"
    W = Ising2D_to_RBM(j, h, L)
elseif WEIGHTS_FILE == "Ising3D"
    W = Ising3D_to_RBM(j, h, L)
elseif WEIGHTS_FILE == "EA3D"
    W = Float32.(Ising3D_to_RBM(j, L)); idx = findall(x->x==1, W); W[shuffle(idx)[1:trunc(Int, length(idx)*p)]] .*= -1
elseif WEIGHTS_FILE == "BSK"
    W = zeros(Float32, Lv+1, Lh+1); W[2:end, 2:end] = randn(Float32, Lv, Lh).*σ .+ μ
else
    W = Float32.(readdlm(WEIGHTS_FILE))
end

β_out, m_mean, m_std, q_mean, q_std = run_simulation(
    W, β_vec;
    NSim     = NSim,
    NTherm   = NTherm,
    NSampl   = NSampl,
    τ_sampl  = τ_sampl,
    sampler  = SAMPLER,
    vis_only = vis_only,
)

