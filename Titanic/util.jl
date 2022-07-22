using DataFrames
using CSV

df = CSV.File("Titanic/data/train.csv") |> DataFrame

df[!,:Sex] = df[!,:Sex] .== "male"
rename!(df,:Sex => :Male)
df.Embarked = categorical(df.Embarked)
df[!,:Survived] = convert.(Bool,df[!,:Survived])


df.FamilySize = df.SibSp .+ df.Parch

family_df = df[:, Not([:SibSp,:Parch])]
sub_df = df[:, Not([:SibSp,:Parch,:Name,:Ticket,:Cabin,:PassengerId])]
sub_df.Age:Vector{Float64} = coalesce.(sub_df.Age, 100)
dropmissing!(sub_df)




using GLM
using StatsModels
using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(sub_df, .75);

train, test = TrainTestSplit(sub_df, .75);

mm = glm(@formula(Survived ~ Pclass    + Fare ), train, Binomial(), LogitLink());
mmage = glm(@formula(Survived ~ Pclass + Age  + Fare ), train, Binomial(), LogitLink());

m1 = mm.model
m2 = mmage.model



model = glm(@formula(Survived ~ Age + Pclass + Male  + Fare + Embarked ), train, Binomial(), LogitLink());
pred = predict(model,test);

# GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, LinearAlgebra.Cholesky{Float64, Matrix{Float64}}}}:

# Coefficients:
# ─────────────────────────────────────────────────────────────────────
#           Coef.  Std. Error      z  Pr(>|z|)    Lower 95%   Upper 95%
# ─────────────────────────────────────────────────────────────────────
# x1   0.758563    0.353013     2.15    0.0316   0.066671     1.45046
# x2  -0.642236    0.126749    -5.07    <1e-06  -0.890658    -0.393813
# x3   0.00731155  0.00288977   2.53    0.0114   0.00164769   0.0129754
# ─────────────────────────────────────────────────────────────────────
# GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, LinearAlgebra.Cholesky{Float64, Matrix{Float64}}}}:

# Coefficients:
# ──────────────────────────────────────────────────────────────────────
#           Coef.  Std. Error      z  Pr(>|z|)    Lower 95%    Upper 95%
# ──────────────────────────────────────────────────────────────────────
# x1   1.30533     0.38909      3.35    0.0008   0.542727     2.06793
# x2  -0.670986    0.128588    -5.22    <1e-06  -0.923014    -0.418957
# x3  -0.010849    0.00302629  -3.58    0.0003  -0.0167804   -0.00491754
# x4   0.00689647  0.00288883   2.39    0.0170   0.00123448   0.0125585
# ──────────────────────────────────────────────────────────────────────

# GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, LinearAlgebra.Cholesky{Float64, Matrix{Float64}}}}:

# Coefficients:
# ──────────────────────────────────────────────────────────────────────
#           Coef.  Std. Error       z  Pr(>|z|)    Lower 95%   Upper 95%
# ──────────────────────────────────────────────────────────────────────
# x1   3.56503     0.493586      7.22    <1e-12   2.59761      4.53244
# x2  -0.974322    0.153062     -6.37    <1e-09  -1.27432     -0.674325
# x3  -2.6382      0.224447    -11.75    <1e-31  -3.07811     -2.19829
# x4   0.00265361  0.00284555    0.93    0.3511  -0.00292357   0.0082308
# x5   0.196767    0.446724      0.44    0.6596  -0.678797     1.07233
# x6  -0.545483    0.27033      -2.02    0.0436  -1.07532     -0.015646
# ──────────────────────────────────────────────────────────────────────
# GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, LinearAlgebra.Cholesky{Float64, Matrix{Float64}}}}
# GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, LinearAlgebra.Cholesky{Float64, Matrix{Float64}}}}