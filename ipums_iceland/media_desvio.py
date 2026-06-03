import pandas as pd
import numpy as np

df = pd.read_csv("IpumsICELAND_pessoa_homonimo.csv")

df["qtd_homonimos_validos"] = pd.to_numeric(
    df["qtd_homonimos_validos"],
    errors="coerce"
)

df = df.dropna(subset=["qtd_homonimos_validos"])

df["homonyms_minus_one"] = (
    df["qtd_homonimos_validos"] - 1
).clip(lower=0)

mean = df["homonyms_minus_one"].mean()
std = df["homonyms_minus_one"].std(ddof=0)
median = df["homonyms_minus_one"].median()

p5 = np.percentile(df["homonyms_minus_one"], 5)
p10 = np.percentile(df["homonyms_minus_one"], 10)
p90 = np.percentile(df["homonyms_minus_one"], 90)

print(f"Mean   : {mean:.4f}")
print(f"Std    : {std:.4f}")
print(f"Median : {median:.4f}")
print(f"P5     : {p5:.4f}")
print(f"P10    : {p10:.4f}")
print(f"P90    : {p90:.4f}")