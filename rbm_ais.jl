using DelimitedFiles
using LogExpFunctions
using LinearAlgebra
using Statistics
using StatsFuns: logistic
using Random
using CUDA

CUDA.allowscalar(true)

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

const Binary = 0          # 1 → Ising spins {-1,+1};  0 → binary units {0,1}

# const WEIGHTS_FILE = "Ising2D"
# const WEIGHTS_FILE = "Ising3D"
# const WEIGHTS_FILE = "EA3D"
# const WEIGHTS_FILE = "BSK"
# const WEIGHTS_FILE = "Weights/GWGM_weights.dat"
# const WEIGHTS_FILE = "Weights/Rows_weights.dat"
const WEIGHTS_FILE = "Weights/MNIST20h_weights.dat"
# const WEIGHTS_FILE = "Weights/MNIST500h_weights.dat"


const Nβ      = 1_024     # Number of AIS annealing steps
const NAsampl = 1_024     # Number of AIS samples
const NBsampl = 1_024     # Number of samples used to estimate the bias
const NSim    = 1         # Total number of independent simulations

const j = 1.f0            # Ising interaction
const L = 12              # Ising lattice size
const p = 0.5             # EA probability of setting a positive interaction

const σ = 1.f0            # SK std
const μ = 0               # SK mean
const Lv = 64             # SK number of visibles
const Lh = 64             # SK number of hiddens

const trans    = false    # If we want to transpose the matrix or not
const vis_only = false    # Keep only the visible array or keep the hole v-h space
const lnZ_vis  = false    # Integrate the hidden part when calculating the partition function


const Bias_method = "Sgh" # "Sgh" | "Sgh_x" | "Sgh_h" | "PInv" | "B0"

# const T_range = Float32.([1.0, 4.51, 9.0])
# const T_range = Float32.(collect(2.25:0.004:2.29))
# const T_range = Float32.(collect(0.9f0:0.01f0:1.1f0))
# const T_range = Float32.(collect(0.05:0.1:2.0))
# const T_range = 1 ./ Float32.(collect(0.5:0.5:3.0))
# const T_range = 1 ./ Float32.([collect(0.002:0.002:0.01); [0.03, 0.05]; collect(0.1:0.1:0.8); collect(0.85:0.05:1.2); collect(1.3:0.1:1.7)])
# const T_range = 1 ./ Float32.(collect(0.5:0.2:2.5))
const T_range = [1.f0]
# const T_range = Float32.(collect(0.1:0.1:1.7))


# ==============================================================================
#  LATTICE → RBM HELPERS
# ==============================================================================

"""
    Ising2D_to_RBM(j, L) → W

Given the Nearest-neighbour interaction between particles, 
this function returns a RBM weights matrix with a 2D-Ising like interaction

- j == Interaction strength (assumed the same for every particle)
- L == Size of the lattice, such that L² = N
- Returns the extended weights matrix W. Bias (b, c) are set to zero
"""
function Ising2D_to_RBM(j::Float32, L::Int)
    N  = L^2;  Nv = N÷2;  Nh = N÷2
    W  = zeros(Float32, Nv+1, Nh+1)
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
    W[2:end,2:end] = J
    return W
end


"""
    Ising3D_to_RBM(j, L) → W

Given the Nearest-neighbour interaction between particles, 
this function returns a RBM weights matrix with a 3D-Ising like interaction

- j == Interaction strength (assumed the same for every particle)
- L == Size of the lattice, such that L² = N
- Returns the extended weights matrix W. Bias (b, c) are set to zero
"""
function Ising3D_to_RBM(j::Float32, L::Int)
    N  = L^3;  Nv = N÷2;  Nh = N÷2
    W  = zeros(Float32, Nv+1, Nh+1)
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
    W[2:end,2:end] = J
    return W
end


# ==============================================================================
#  BIAS ESTIMATION
# ==============================================================================

"""
    compute_bias(W, NBsampl) → (B, C)

Estimate the reference-distribution biases from `NBsampl` random configurations.

- Uses only the physical block `W[2:end,2:end]` so the bias row/column of W
  does not contaminate the field estimate.
- Sign convention: ξ = fraction of configurations where the field is positive,
  i.e. the mean activation under a random input. This is the correct direction.
- Returns column vectors of length Nv+1 and Nh+1 (index 1 is the bias unit = 0).
"""
function compute_bias(W::CuMatrix{Float32}, NBsampl::Int)
    Nv, Nh = size(W) .- 1
    J = W[2:end, 2:end]   # physical block only, no bias row/col

    # Random {0,1} or {-1,+1} configurations (physical units only)
    spin_lo = Float32(-Binary)   # -1 (Ising) or 0 (binary)
    h_rand = CUDA.rand(Float32, Nh, NBsampl) .< 0.5f0
    v_rand = CUDA.rand(Float32, Nv, NBsampl) .< 0.5f0
    h_rand = Float32.(h_rand) .* (1f0 - spin_lo) .+ spin_lo   # map to {spin_lo, 1}
    v_rand = Float32.(v_rand) .* (1f0 - spin_lo) .+ spin_lo

    # Mean activation:  ξ_v = < (sign(J h) + 1) / 2 >_h,  and vice versa
    field_v = J  * h_rand    # Nv × NBsampl
    field_h = J' * v_rand    # Nh × NBsampl
    ξv = vec(mean((-sign.(field_v) .+ 1f0) ./ 2f0, dims=2))   # ∈ [0,1]
    ξh = vec(mean((-sign.(field_h) .+ 1f0) ./ 2f0, dims=2))   # ∈ [0,1]

    # Shrink away from 0 and 1 to avoid log(0) / log(∞)
    ξv .= ξv .+ 0.05f0 .- 0.1f0 .* ξv
    ξh .= ξh .+ 0.05f0 .- 0.1f0 .* ξh

    # Convert mean activation to logistic bias
    if Binary == 0
        B_phys = log.(1f0 ./ ξv .- 1f0)
        C_phys = log.(1f0 ./ ξh .- 1f0)
    else
        B_phys = atanh.(-2f0 .* ξv .+ 1f0)
        C_phys = atanh.(-2f0 .* ξh .+ 1f0)
    end

    # Prepend the bias unit (always 0)
    B = CuArray([0f0; Array(B_phys)])
    C = CuArray([0f0; Array(C_phys)])
    return B, C
end


"""
    compute_bias_pinv(W) → (B, C)

Pseudoinverse-based bias: solve  J h = b  and  Jᵀ v = c  for the ground-state
configuration, then convert to logistic biases.

Correction vs original: `pinv(J') * c`, not `pinv(J)' * c`.
"""
function compute_bias_pinv(W::CuMatrix{Float32})
    J = Array(W[2:end, 2:end])
    b = Array(W[2:end, 1])
    c = Array(W[1, 2:end])

    ξh = pinv(J)  * b      # solve  J  h = b
    ξv = pinv(J') * c      # solve  Jᵀ v = c   (corrected from pinv(J)')

    clamp!(ξv, 0f0, 1f0 - eps(Float32))
    clamp!(ξh, 0f0, 1f0 - eps(Float32))

    ξv .= ξv .+ 0.05f0 .- 0.1f0 .* ξv
    ξh .= ξh .+ 0.05f0 .- 0.1f0 .* ξh

    if Binary == 0
        B_phys = log.(1f0 ./ ξv .- 1f0)
        C_phys = log.(1f0 ./ ξh .- 1f0)
    else
        B_phys = atanh.(2f0 .* ξv .- 1f0)
        C_phys = atanh.(2f0 .* ξh .- 1f0)
    end

    B = CuArray([0f0; Float32.(B_phys)])
    C = CuArray([0f0; Float32.(C_phys)])
    return B, C
end


"""
    make_bias_matrices(W, method, NBsampl, NAsampl) → (B, C)

Build the full (Nv+1)×NAsampl and (Nh+1)×NAsampl bias matrices according to
the chosen method.  Only "Sgh" variants call `compute_bias`.
"""
function make_bias_matrices(W::CuMatrix{Float32}, method::String,
                             NBsampl::Int, NAsampl::Int)
    Nv, Nh = size(W) .- 1

    if method == "B0"
        B = CUDA.zeros(Float32, Nv+1, NAsampl)
        C = CUDA.zeros(Float32, Nh+1, NAsampl)

    elseif method ∈ ("Sgh", "Sgh_x", "Sgh_h")
        # Compute bias NAsampl times from NBsampl random configurations.
        B = CUDA.zeros(Float32, Nv+1, NAsampl)
        C = CUDA.zeros(Float32, Nh+1, NAsampl)
        # Assuming different bias for each independent sample.
        # If you want to use the same bias for all samples, comment the next 3 lines and uncomment the following 3 lines
        for ii = 1:NAsampl
            B[:, ii], C[:, ii] = compute_bias(W, NBsampl)
        end
        # b_col, c_col = compute_bias(W, NBsampl)
        # B = repeat(b_col, 1, NAsampl)
        # C = repeat(c_col, 1, NAsampl)
        if method == "Sgh_x"  C .= 0f0  end
        if method == "Sgh_h"  B .= 0f0  end

    elseif method == "PInv"
        b_col, c_col = compute_bias_pinv(W)
        B = repeat(b_col, 1, NAsampl)
        C = repeat(c_col, 1, NAsampl)

    else
        error("Unknown Bias_method: $method")
    end

    return B, C
end


# ==============================================================================
#  CD-1 STEP  (in-place, pre-allocated buffers)
# ==============================================================================

"""
    cd1!(v, h, W, B, C, bufs)

One step of block Gibbs sampling (CD-1).  All arrays are modified in-place;
random buffers in `bufs` are refilled with `rand!` to avoid GPU allocations.

Note: the minus sign in the logistic argument has intentionally been omitted
as per the caller's convention (absorbed into the sign of W).
"""
function cd1!(v::CuMatrix{Float32}, h::CuMatrix{Float32},
              W::CuMatrix{Float32}, B::CuMatrix{Float32}, C::CuMatrix{Float32},
              bufs::NamedTuple)

    scale = 1f0 + Float32(Binary)   # 1 (Ising) or 2 (binary)
    spin_lo = Float32(-Binary)      # -1 or 0

    # Hidden units: p(h=1 | v) = σ(scale · (Wᵀv + C))
    mul!(bufs.field_h, W', v)
    bufs.field_h .+= C
    bufs.ph .= logistic.(scale .* bufs.field_h)
    bufs.ph[1, :] .= 1f0            # bias unit always on
    rand!(bufs.uh)
    h .= ifelse.(bufs.uh .< bufs.ph, 1f0, spin_lo)

    # Visible units: p(v=1 | h) = σ(scale · (Wh + B))
    mul!(bufs.field_v, W, h)
    bufs.field_v .+= B
    bufs.pv .= logistic.(scale .* bufs.field_v)
    bufs.pv[1, :] .= 1f0
    rand!(bufs.uv)
    v .= ifelse.(bufs.uv .< bufs.pv, 1f0, spin_lo)

    return nothing
end


# ==============================================================================
#  ANNEALED IMPORTANCE SAMPLING
# ==============================================================================

"""
    ais(W, B, C, Nβ, NAsampl) → (v, h, lnZ)

Annealed Importance Sampling for an RBM.

The annealing path interpolates between the reference distribution
(biases B, C; no interactions) at β=0 and the target (full W) at β=1.

The log-weight increment at step k is:
    Δ ln w = (1/Nβ) · (vᵀWh - v·B - h·C)
where vᵀWh uses only the physical block (no bias units), and B[1]=C[1]=0
so the bias-unit terms vanish automatically.

Returns the final configurations (v, h) and the per-sample log-weights `lnZ`.
"""
function ais(W::CuMatrix{Float32}, B::CuMatrix{Float32}, C::CuMatrix{Float32},
             Nβ::Int, NAsampl::Int)

    Nv, Nh = size(W) .- 1
    spin_lo = Float32(-Binary)
    spin_hi = 1f0

    # Pre-allocate buffers for cd1!
    bufs = (
        field_h = CUDA.zeros(Float32, Nh+1, NAsampl),
        field_v = CUDA.zeros(Float32, Nv+1, NAsampl),
        ph      = CUDA.zeros(Float32, Nh+1, NAsampl),
        pv      = CUDA.zeros(Float32, Nv+1, NAsampl),
        uh      = CUDA.zeros(Float32, Nh+1, NAsampl),
        uv      = CUDA.zeros(Float32, Nv+1, NAsampl),
    )

    # Random initial configurations
    v = ifelse.(CUDA.rand(Float32, Nv+1, NAsampl) .< 0.5f0, spin_hi, spin_lo)
    h = ifelse.(CUDA.rand(Float32, Nh+1, NAsampl) .< 0.5f0, spin_hi, spin_lo)
    v[1, :] .= 1f0;  h[1, :] .= 1f0    # bias units always on

    # Reference partition function:
    #   ln Z_0 = sum_i ln(1 + exp(B_i)) + sum_j ln(1 + exp(C_j))   [Ising]
    #   ln Z_0 = N·ln2 + sum_i ln cosh(B_i) + sum_j ln cosh(C_j)   [binary]
    B_phys = B[2:end, :]   # (Nv) × NAsampl
    C_phys = C[2:end, :]
    if Binary == 0
        lnZ_0 = vec(sum(log1pexp.(B_phys), dims=1) .+
                    sum(log1pexp.(C_phys), dims=1))
    else
        lnZ_0 = CUDA.fill(Float32(Nv + Nh) * log(2f0), NAsampl) .+
                vec(sum(logcosh.(B_phys), dims=1)) .+
                vec(sum(logcosh.(C_phys), dims=1))
    end

    lnZ = CUDA.zeros(Float32, NAsampl)

    # Scratch arrays to avoid allocations inside the loop
    W_INP = similar(W)
    B_INP = similar(B)
    C_INP = similar(C)
    # For computing vᵀWh per sample without materialising a full matrix product:
    #   En[k] = v[:,k] · (W * h[:,k])  →  use batched dot via mul! + sum
    vWh_tmp = CUDA.zeros(Float32, Nv+1, NAsampl)   # W_INP * h, reused

    inv_Nβ = 1f0 / Float32(Nβ)

    for iβ in 0:Nβ-1
        β   = Float32(iβ)   / Float32(Nβ)
        βp1 = Float32(iβ+1) / Float32(Nβ)

        # Annealed weights and biases
        W_INP .= β .* W
        B_INP .= (1f0 - β) .* B
        C_INP .= (1f0 - β) .* C

        # One CD-1 step under the annealed distribution
        cd1!(v, h, W_INP, B_INP, C_INP, bufs)

        if lnZ_vis == false
            # Log-weight increment:  Δlnw = (1/Nβ) · (vᵀWh - v·B - h·C)
            # Compute vᵀWh efficiently: mul!(vWh_tmp, W, h) then colwise dot with v
            mul!(vWh_tmp, W, h)
            # En[k] = sum_i v[i,k] * vWh_tmp[i,k]  →  sum(v .* vWh_tmp, dims=1)
            En = vec(sum(v .* vWh_tmp, dims=1))
            E0 = vec(sum(v .* B, dims=1) .+ sum(h .* C, dims=1))
            lnZ .+= (En .- E0) .* inv_Nβ
        
        else
            # @views avoids materialising copies of these sub-blocks every β-step
            # (W, B, C don't change within this loop, only β and βp1 do).
            @views J = W[2:end, 2:end]
            @views b = W[2:end, 1]
            @views c = W[1, 2:end]
            @views v_phys = v[2:end, :]
            @views B_phys = B[2:end, :]
            @views C_phys = C[2:end, :]
 
            # Compute this terms only once
            Jv_c   = J' * v_phys .+ c
            argp1  = βp1 .* Jv_c .+ (1f0 - βp1) .* C_phys
            arg0   = β   .* Jv_c .+ (1f0 - β)   .* C_phys
 
            # The first term is the same either you're working with [0, 1] or with [-1, 1]
            Term1 = vec(sum(v_phys .* (b .- B_phys), dims=1)) .* inv_Nβ
 
            if Binary == 0
                Term2 = vec(sum(log1pexp.(argp1), dims=1))
                Term3 = vec(sum(log1pexp.(arg0), dims=1))
 
            elseif Binary == 1
                Term2 = vec(sum(logcosh.(argp1), dims=1))
                Term3 = vec(sum(logcosh.(arg0), dims=1))
 
            end
 
            lnZ .+= Term1 + Term2 - Term3
        end


        

        if mod(iβ, Nβ÷10) == 0
            println("  AIS step $(iβ)/$(Nβ)  (β=$(round(β,digits=3)))"); flush(stdout)
        end
    end

    return v, h, lnZ .+ lnZ_0
end


# ==============================================================================
#  OBSERVABLES
# ==============================================================================

"""
    compute_observables(v, h, W, lnZ_samples, N, Nv, Nh)

Returns a NamedTuple with importance-weighted estimates of:
  - lnZ      : log partition function
  - e1/e2/e4 : first/second/fourth moment of energy per spin
  - m1/m2/m4 : first/second/fourth moment of magnetisation
  - q1/q2/q4 : first/second/fourth moment of overlap q_EA
"""
function compute_observables(v::CuMatrix{Float32}, h::CuMatrix{Float32},
    W::CuMatrix{Float32},
    lnZ_samples::CuVector{Float32},
    N::Int, Nv::Int, Nh::Int)

    NAsampl = size(v, 2)

    # Importance weights (numerically stable)
    ω   = exp.(lnZ_samples .- maximum(lnZ_samples))
    Ω_mat = ω * ω'                              # NAsampl × NAsampl, on GPU

    # Zero the diagonal and lower triangle using index arithmetic — fully on GPU,
    # no tril/triu calls needed.
    # upper_mask[i,j] = 1 if j > i, else 0
    rows = CuArray(collect(Int32, 1:NAsampl))   # column vector
    cols = CuArray(collect(Int32, 1:NAsampl))'  # row vector
    upper_mask = Float32.(cols .> rows)          # NAsampl × NAsampl, on GPU
    Ω_mat .*= upper_mask                        # keep strict upper triangle only

    lnZ = logsumexp(Array(lnZ_samples)) - log(NAsampl)

    # Energy per spin: e = vᵀWh / N  (sign convention matches W)
    vWh_tmp = CUDA.zeros(Float32, Nv+1, NAsampl)
    mul!(vWh_tmp, W, h)
    en_all = vec(sum(v .* vWh_tmp, dims=1)) ./ Float32(N)

    # Magnetisation per spin (all units, Ising convention)
    if vis_only  x = CuArray(v[2:end,:]);               Nd = Nv
    else         x = CuArray([v[2:end,:]; h[2:end,:]]); Nd = N
    end
    m_raw  = vec(sum(x, dims=1)) ./ Float32(Nd)
    m1_all = abs.(m_raw)
    m2_all = m_raw .^ 2
    m4_all = m_raw .^ 4


    # Overlap q_EA (strict upper triangle of pairwise overlaps)
    if Binary == 0  x = x .* 2f0 .- 1f0  end       # map {0,1} → {-1,+1}
    q_mat  = (x' * x) ./ Float32(Nd)
    q1_mat = abs.(q_mat) .* upper_mask
    q2_mat = q1_mat .^ 2
    q4_mat = q1_mat .^ 4

    # Importance-weighted averages
    sum_ω = sum(ω)
    sum_Ω = sum(Ω_mat)

    weighted(f, w, s) = sum(f .* w) / s

    return (
        lnZ = lnZ,
        e1  = weighted(en_all,    ω,     sum_ω),
        e2  = weighted(en_all.^2, ω,     sum_ω),
        e4  = weighted(en_all.^4, ω,     sum_ω),
        m1  = weighted(m1_all,    ω,     sum_ω),
        m2  = weighted(m2_all,    ω,     sum_ω),
        m4  = weighted(m4_all,    ω,     sum_ω),
        q1  = weighted(q1_mat,    Ω_mat, sum_Ω),
        q2  = weighted(q2_mat,    Ω_mat, sum_Ω),
        q4  = weighted(q4_mat,    Ω_mat, sum_Ω),
    )
end


# ==============================================================================
#  MAIN LOOP
# ==============================================================================

for ii = 1:NSim
    println("## AIS — RBM thermodynamics")
    println("# Nβ=$(Nβ)  NAsampl=$(NAsampl)  NBsampl=$(NBsampl)  method=$(Bias_method)")
    flush(stdout)

    if WEIGHTS_FILE == "Ising2D"
        W_raw = CuArray(Float32.(Ising2D_to_RBM(j, L)))
    elseif WEIGHTS_FILE == "Ising3D"
        W_raw = CuArray(Float32.(Ising3D_to_RBM(j, L)))
    elseif WEIGHTS_FILE == "EA3D"
        W_raw = Float32.(Ising3D_to_RBM(j, L)); idx = findall(x->x==1, W_raw); W_raw[shuffle(idx)[1:trunc(Int, length(idx)*p)]] .*= -1; W_raw = CuArray(W_raw)
    elseif WEIGHTS_FILE == "BSK"
        W_raw = CUDA.zeros(Float32, Lv+1, Lh+1); W_raw[2:end, 2:end] = randn(Float32, Lv, Lh).*σ .+ μ
    else
        W_raw = CuArray(Float32.(readdlm(WEIGHTS_FILE)))
    end

    if trans  W_raw = CuArray(transpose(W_raw))  end
    Nv, Nh = size(W_raw) .- 1
    N = Nv + Nh

    println("# Nv=$(Nv)  Nh=$(Nh)  N=$(N)")
    println("# Transpose: $(trans)")
    println("# Pick only visibles: $(vis_only)")
    println("# Integrate lnZ: $(lnZ_vis)")
    println("# Temperatures: $(T_range)")
    flush(stdout)

    for Temp in T_range
        β    = 1f0 / Temp
        W    = W_raw ./ Temp     # absorb temperature into weights

        println("\n# === T=$(Temp)  β=$(β) ===")
        println("# Building bias matrices...")
        flush(stdout)

        B, C = make_bias_matrices(W, Bias_method, NBsampl, NAsampl)

        println("# Running AIS (Nβ=$(Nβ), NAsampl=$(NAsampl))...")
        flush(stdout)

        v, h, lnZ_all = ais(W, B, C, Nβ, NAsampl)

        println("# Computing observables...")
        flush(stdout)

        obs = compute_observables(v, h, W_raw, lnZ_all, N, Nv, Nh)

        println("# Averages:  ",
            Float32(Temp),     "\t",
            Float32(obs.lnZ),  "\t",
            Float32(obs.e1),   "\t",
            Float32(obs.e2),   "\t",
            Float32(obs.e4),   "\t",
            Float32(obs.m1),   "\t",
            Float32(obs.m2),   "\t",
            Float32(obs.m4),   "\t",
            Float32(obs.q1),   "\t",
            Float32(obs.q2),   "\t",
            Float32(obs.q4),
        )
        flush(stdout)

        CUDA.reclaim()   # return GPU memory between temperatures
    end
end
