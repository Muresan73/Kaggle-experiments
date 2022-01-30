using DataFrames
using CSV

df = CSV.File("data/train.csv") |> DataFrame

describe(df)

