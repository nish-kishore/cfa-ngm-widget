library(tidyverse)
library(flextable)
library(scales)
library(here)

#' Create an output table for NGM results for MPOX
#' @description
#' Function to simplify and streamline creation of MPOX output tables
#' with a focus on deaths rather than infections
#'
#' @param file `str` Name of file which will be results csv to be read in from
#' the results section of the project
#' @param title `str` Title to show on top of figure
#' @param footnote `str` Footnote with model specifics
#'
#'
#' @returns `flextable` Returns a flextable object
#'
create_results_table <- function(file, title, footnote){

  #file <- "kenya_001.csv"

  dat <- read_csv(paste0(here("results", file)))
  n_groups <- select(dat, starts_with("inf")) |> ncol()

  group_names <- select(dat, starts_with("inf")) |>
    names() |>
    str_replace_all("infections_", "")

  group_out <- lapply(group_names, function(x){
    select(dat, ends_with(x)) |>
      set_names(c("Inf.", "D. per Inf.", "Total D.")) |>
      mutate(
        `Inf.` = percent(`Inf.` / `Inf.`[1], accuracy = 1),
        `D. per Inf.` = percent(`D. per Inf.` / `D. per Inf.`[1], accuracy = 1),
        `Total D.` = percent(`Total D.` / `Total D.`[1], accuracy = 1)
      )
  })

  pop_out <- dat |>
    select(Re, IFR = ifr,
           "D. per Inf." = deaths_per_prior_infection,
           "Total D." = deaths_after_G_generations)|>
    mutate(
      `D. per Inf.` = percent(`D. per Inf.` / `D. per Inf.`[1], accuracy = 1),
      `Total D.` = percent(`Total D.` / `Total D.`[1], accuracy = 1),
      IFR = percent(IFR/IFR[1], accuracy = 1)
    )

  out <- bind_cols(dat[,1], pop_out, group_out) |>
    mutate_all(as.character)

  out[1,] <- dat[1,] |>
    mutate_if(is.numeric, list(~ifelse(. > 10, round(., 0), .))) |>
    mutate_all(as.character)

  footnote <- as_paragraph(footnote)

  out |>
    select(-starts_with("Inf")) |>
    flextable() |>
    #delete_columns(j = 1:n_groups * 3 - 2 + 5) |>
    set_header_labels(values = c("Intervention", "Re", "IFR", "Deaths per Infection", "Total Deaths",
                                 rep(c("Deaths per Infection", "Total Deaths"), n_groups))) |>
    add_header_row(values = c("Simulation", "Total Population", group_names), colwidths = c(1,4,rep(2, n_groups))) |>
    align(i = 1, j = NULL, align = "center", part = "header") |>
    vline(i = NULL, j = c(1,5,(1:n_groups * 2) + 5)) |>
    border_outer() |>
    merge_at(i = c(1,2), j = 1, part = "header") |>
    bg(i = 1, j = NULL, part = "body", bg = "gray55")  |>
    set_caption(caption = title) |>
    add_footer_lines(value = footnote)


}

create_results_table(
  file = "kenya_001.csv",
  title = "Simulation `kenya_001`: Subcutaneous (SC), Intradermal (ID) and Behavior Change (BC)",
  footnote = "Meta-population model including Nairobi, Mombasa and Biasu\nVaccine Distribution: CSW(0.45), HCW(0.1), TD(0.35), Adult(0.1)\nVaccine limit: 10700"
  )

create_results_table(
  file = "kenya_002.csv",
  title = "Simulation `kenya_002`: Subcutaneous (SC), Intradermal (ID) and Behavior Change (BC) + optimized vaccine distribution",
  footnote = "Meta-population model including Nairobi, Mombasa and Biasu\nVaccine Distribution: CSW(0.31), HCW(0.1), TD(0.49), Adult(0.1)\nVaccine limit: 10700"
)
