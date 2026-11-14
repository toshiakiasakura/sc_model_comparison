function default_plot_setting()
	gr(fontfamily = "Helvetica",
		foreground_color_legend = nothing,
		background_color_legend = nothing,
		titlefontsize = 11, tickfontsize = 10,
		legendfontsize = 8, legendtitlefontsize = 8,
		labelfontsize = 10,
		grid = true, tick_direction = :out,
		size = (600, 450))
end

function read_survey_master_data()
	path = "../dt_surveys_master/survey_mastersheet_info.xlsx"
	return XLSX.readtable(path, "Sheet1") |> DataFrame
end

const model_abbr = Dict(
	"ZeroInfNegativeBinomial" => "ZInf-NB",
	"ZeroInfPoissonLogNormal" => "ZInf-PLN",
	"ZeroInfPoissonLomax" => "ZInf-PLomax"
)

const setting_pretty = Dict(
	"all" => "All",
	"home" => "Home",
	"non-home" => "Non-home"
)

boldstring(s::AbstractString) = map(c -> 'a' ≤ c ≤ 'z' ? c + ('𝐚'-'a') : 'A' ≤ c ≤ 'Z' ? c + ('𝐀'-'A') : c, s)