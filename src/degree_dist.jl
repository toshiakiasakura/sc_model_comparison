Base.@kwdef mutable struct DegreeDist
	x::Vector{Int64} # Degree
	y::Vector{Int64} # Corresponding count
	include_zero::Bool = true
end

function DegreeDist(cnt::Vector{Int64}; include_zero = true)::DegreeDist
	cnt = cnt |> countmap
	x, y = collect(pairs(cnt)) |> (x -> (first.(x), last.(x)))
	ind = sortperm(x)
	return DegreeDist(x[ind], y[ind], include_zero)
end

function DegreeDist(cnt::Vector{Int64}, n_part::Int64; include_zero = true)::DegreeDist
	n_cnt1more = length(cnt)
	dd = DegreeDist(cnt; include_zero = include_zero)
	n_zero = n_part - n_cnt1more
	if n_zero < 0
		error("More participants with contacts than total participants n_cnt1more=$n_cnt1more and n_part=$n_part")
	end
	insert!(dd.x, 1, 0)
	insert!(dd.y, 1, n_zero)
	return dd
end

dd_to_df(dd::DegreeDist)::DataFrame = DataFrame(x = dd.x, y = dd.y)
function dd_to_df(dd::DegreeDist, strat::String)::DataFrame
	df = dd_to_df(dd)
	df[!, :strat] .= strat
	return df
end

function DegreeDist(df::DataFrame)::DegreeDist
	if "x" in names(df) && "y" in names(df)
		if length(df[:, :x]) == length(unique(df[:, :x]))
			return DegreeDist(df[:, :x], df[:, :y], true)
		else
			error("x values are not unique")
		end
	else
		error("DataFrame does not have x and y columns")
	end
end

Base.length(dd::DegreeDist) = length(dd.x)
Base.maximum(dd::DegreeDist) = maximum(dd.x)
Base.iterate(p::DegreeDist) = (p, nothing)
Base.iterate(p::DegreeDist, nothing) = nothing
Distributions.rand(dd::DegreeDist, n::Int64) = sample(dd.x, Weights(dd.y), n)
Distributions.mean(dd::DegreeDist) = sum(dd.x .* dd.y) / sum(dd.y)
Base.sum(dd::DegreeDist) = sum(dd.y)

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

function plot_ccdf!(pl::Plots.Plot, dd::DegreeDist; ytk_digit = 6, marker = :circle, kwds...)
	if dd.include_zero == true
		dd = deepcopy(dd)
		dd.x = dd.x[2:end]
		dd.y = dd.y[2:end]
	end
	y_ccdf = obtain_ccdf(dd)
	ind_incl = 1:ytk_digit
	ytk = (
		[0, -1, -2, -3, -4, -5][ind_incl],
		[L"1", L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}"][ind_incl])
	plot!(pl, dd.x, log10.(y_ccdf); marker = marker, yticks = ytk,
		kwds...)
end
plot_ccdf!(pl::Plots.Plot, x::Vector{Int64}; kwds...) = plot_ccdf!(pl, DegreeDist(x); kwds...)

function plot_pdf!(pl::Plots.Plot, dd::DegreeDist; conv_log10 = true, ytk_digit = 6, kwds...)
	y_pdf = dd.y / sum(dd.y)
	y_pdf = conv_log10 == true ? log10.(y_pdf) : y_pdf
	ind_incl = 1:ytk_digit
	if conv_log10 == true
		ytk = (
			[0, -1, -2, -3, -4, -5][ind_incl],
			[L"1", L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}"][ind_incl])
	else
		ytk = true
	end
	plot!(pl, dd.x, y_pdf; marker = :circle, yticks=ytk, kwds...)
end
plot_pdf!(pl::Plots.Plot, x::Vector{Int64}; kwds...) = plot_pdf!(pl, DegreeDist(x); kwds...)
