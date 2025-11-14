function default_plot_setting()
	gr(fontfamily = "Helvetica",
		foreground_color_legend = nothing,
		background_color_legend = nothing,
		#titlefontsize = 11, tickfontsize = 10,
		#legendfontsize = 8, legendtitlefontsize = 8,
		#labelfontsize = 10,
		grid = true, tick_direction = :out,
		size = (600, 450))
end

function value_counts(df::DataFrame, by::Vector; cname = :nrow)::DataFrame
	return @pipe groupby(df, by) |>
				 combine(_, nrow => cname)
end
value_counts(df::DataFrame, by::Symbol; cname = :nrow)::DataFrame =
	value_counts(df, [by]; cname = cname)

vec_vec_to_matrix(v::Vector) = mapreduce(permutedims, vcat, v)
