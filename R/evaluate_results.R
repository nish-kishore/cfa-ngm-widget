library(tidyverse)
library(flextable)
library(scales)
library(here)

file <- "kenya_001.csv"

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

out |>
  select(-starts_with("Inf")) |>
  flextable() |>
  #delete_columns(j = 1:n_groups * 3 - 2 + 5) |>
  set_header_labels(values = c("Intervention", "Re", "IFR", "D. per Inf.", "Total D.",
                               rep(c("D. per Inf.", "Total D."), n_groups))) |>
  add_header_row(values = c("Sim.", "Gen. Pop.", group_names), colwidths = c(1,4,rep(2, n_groups))) |>
  align(i = 1, j = NULL, align = "center", part = "header") |>
  vline(i = NULL, j = c(1,5,(1:n_groups * 2) + 5))


