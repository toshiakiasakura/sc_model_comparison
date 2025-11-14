Base.@kwdef mutable struct DegreeDist
	x::Vector{Int64} # Degree
	y::Vector{Int64} # Corresponding count
	include_zero::Bool = true
end
Base.length(dd::DegreeDist) = length(dd.x)
Base.maximum(dd::DegreeDist) = maximum(dd.x)
Base.iterate(p::DegreeDist) = (p, nothing)
Base.iterate(p::DegreeDist, nothing) = nothing
Distributions.rand(dd::DegreeDist, n::Int64) = sample(dd.x, Weights(dd.y), n)

function DegreeDist(cnt::Vector{Int64}; include_zero = true)::DegreeDist
	cnt = cnt |> countmap
	x, y = collect(pairs(cnt)) |> (x -> (first.(x), last.(x)))
	ind = sortperm(x)
	return DegreeDist(x[ind], y[ind], include_zero)
end
Distributions.mean(dd::DegreeDist) = sum(dd.x .* dd.y) / sum(dd.y)

"""Given, a vector of a probablity mass function,
return a ccdf distribution.
"""
function obtain_ccdf(y::Vector)
	return cumsum(y[end:-1:begin])[end:-1:begin]
end

function obtain_ccdf(dd::DegreeDist)::Vector{Float64}
	y = dd.y / sum(dd.y)
	y_ccdf = obtain_ccdf(y)
	return y_ccdf
end

"""
See also `turing_utils.jl` which contains
- `plot_single_setting`
- `plot_multiple_settings`
"""
function plot_ccdf!(pl::Plots.Plot, dd::DegreeDist; kwds...)
	if dd.include_zero == true
		dd = deepcopy(dd)
		dd.x = dd.x[2:end]
		dd.y = dd.y[2:end]
	end
	y_ccdf = obtain_ccdf(dd)
	plot!(pl, dd.x, log10.(y_ccdf); marker = :circle, kwds...)
end
plot_ccdf!(pl::Plots.Plot, x::Vector{Int64}; kwds...) = plot_ccdf!(pl, DegreeDist(x); kwds...)

function plot_ccdf(dd::DegreeDist; kwds...)::Plots.Plot
	y_ccdf = obtain_ccdf(dh)
	pl = plot(xaxis = :log10, xlabel = "log10(k)", ylabel = "log10(ccdf(k))", xlim = [1, 10000])
	scatter!(pl, dh.x, log10.(y_ccdf); kwds...)
	return pl
end
plot_ccdf(x::Vector{Int64}; kwds...)::Plots.Plot = plot_ccdf(DegreeDist(x); kwds...)

function plot_pdf!(pl::Plots.Plot, dd::DegreeDist; kwds...)
	y_pdf = dd.y / sum(dd.y)
	plot!(pl, dd.x, log10.(y_pdf); marker = :circle, kwds...)
end
plot_pdf!(pl::Plots.Plot, x::Vector{Int64}; kwds...) = plot_pdf!(pl, DegreeDist(x); kwds...)

function plot_pdf(dd::DegreeDist; kwds...)
	pl = plot(xlabel = "k", ylabel = "pdf", xlim = [0, 20])
	plot_pdf!(pl, dd; kwds...)
	return pl
end
plot_pdf(x::Vector{Int64}; kwds...) = plot_pdf(DegreeDist(x); kwds...)