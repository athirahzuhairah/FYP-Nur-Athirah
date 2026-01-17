library(dplyr)
library(lubridate)
library(ggplot2)
library(broom)

data_obj2 <- read.csv(file.choose(), header=TRUE)

#convert date column to Date format
df <- data_obj2 %>%
  mutate(date = dmy(date))

#filter selected high-density states
df <- df %>% 
  filter(Negeri %in% c("Selangor", "W.P. Kuala Lumpur", "Pulau Pinang"))

#create weekly aggregated data
df_weekly <- df %>%
  mutate(week = floor_date(date, unit = "week")) %>%
  group_by(Negeri, week) %>%
  summarise(deaths = sum(bid == 1, na.rm = TRUE), .groups = "drop") %>%
  arrange(Negeri, week)

#first vaccine dose
#define intervention date for first dose
intervention_date <- ymd("2021-02-24")  # Dos pertama

df_weekly_1 <- df_weekly %>%
  group_by(Negeri) %>%
  mutate(
    time = row_number(),
    intervention = ifelse(week >= intervention_date, 1, 0),
    time_after = ifelse(
      week >= intervention_date,
      time - which(week >= intervention_date)[1] + 1,
      0
    )
  ) %>%
  ungroup()

#plot weekly deaths with first dose intervention
ggplot(df_weekly_1, aes(x = week, y = deaths, color = Negeri)) +
  geom_line() +
  geom_vline(xintercept = as.numeric(intervention_date),
             color = "red", linetype = "dashed") +
  labs(title = "Trend Kematian Mingguan (Dos Pertama)",
       subtitle = "Garis merah = Permulaan Vaksin Dos Pertama",
       x = "Minggu", y = "Bilangan Kematian (BID)") +
  theme_minimal()

#Interrupted Time Series model for first dose
model_all1 <- lm(deaths ~ time + intervention + time_after,
                 data = df_weekly_1)
summary(model_all1)
confint(model_all1)

#second vaccine dose
#define intervention date for first dose
intervention_date <- ymd("2022-04-14")  # Dos kedua

df_weekly_2 <- df_weekly %>%
  group_by(Negeri) %>%
  mutate(
    time = row_number(),
    intervention = ifelse(week >= intervention_date, 1, 0),
    time_after = ifelse(
      week >= intervention_date,
      time - which(week >= intervention_date)[1] + 1,
      0
    )
  ) %>%
  ungroup()

#plot weekly deaths with second dose intervention
ggplot(df_weekly_2, aes(x = week, y = deaths, color = Negeri)) +
  geom_line() +
  geom_vline(xintercept = as.numeric(intervention_date),
             color = "blue", linetype = "dashed") +
  labs(title = "Trend Kematian Mingguan (Dos Kedua)",
       subtitle = "Garis biru = Permulaan Vaksin Dos Kedua",
       x = "Minggu", y = "Bilangan Kematian (BID)") +
  theme_minimal()

#Interrupted Time Series model for second dose
model_all2 <- lm(deaths ~ time + intervention + time_after,
                 data = df_weekly_2)
summary(model_all2)
confint(model_all2)

#identify the week with the largest increase in deathsdf_peaks <- df_weekly %>%
  group_by(Negeri) %>%
  arrange(week) %>%
  mutate(change = deaths - lag(deaths)) %>%
  filter(!is.na(change)) %>%
  slice_max(change, n = 1, with_ties = FALSE) %>%
  select(Negeri, week, deaths, change)

print(df_peaks)