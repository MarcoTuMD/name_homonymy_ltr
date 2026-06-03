from ipumspy import readers
import pandas as pd
from pprint import pprint
import numpy as np
from contextlib import redirect_stdout
import json
from scipy.stats import t

def calculo_media(df, iteration):
    df = df.copy()
    df = df[df["label"] == 1].copy()

    df["idade_chefe"] = pd.to_numeric(df["idade_chefe"], errors="coerce")
    df["idade_candidato"] = pd.to_numeric(df["idade_candidato"], errors="coerce")

    df = df.dropna(subset=["idade_chefe", "idade_candidato", "relacao"])
    df = df[~df["idade_chefe"].isin([999])]
    df = df[~df["idade_candidato"].isin([999])]

    df = df[~df["relacao"].isin(["Unknown", "Non-relative"])].copy()

    df["diff_idade"] = df["idade_candidato"] - df["idade_chefe"]

    stats_df = (
        df.groupby("relacao")["diff_idade"]
          .agg(mean="mean", std="std", qtd="count")
          .reset_index()
          .sort_values("qtd", ascending=False)
    )

    stats_dict = (
        stats_df.set_index("relacao")[["mean", "std", "qtd"]]
          .round({"mean": 2, "std": 2})
          .to_dict(orient="index")
    )
    with open(f"estatisticas_relacionamento_{iteration}.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=4, ensure_ascii=False)

    return stats_df, stats_dict

def estimar_tstudent_por_relacao(df_train, k):
    stats = {}

    for rel, group in df_train.groupby("relacao"):

        idade_chefe = group["idade_chefe"]
        idade_cand  = group["idade_candidato"]

        diff = idade_cand - idade_chefe
        diff = diff.dropna().values

        if len(diff) < 5:
            continue

        df_param, loc, scale = t.fit(diff)

        stats[rel] = {
            "df": float(df_param),
            "loc": float(loc),
            "scale": float(scale),
            "qtd": int(len(diff))
        }

    with open(f"estatisticas_relacao_tstudent_fold{k}.json", "w") as f:
        json.dump(stats, f, indent=4)

    return stats

for k in range(5):
    df_train = pd.read_csv(f"fold{k}_train_nao_balanceado.csv")
    stats = estimar_tstudent_por_relacao(df_train, k)
    print(k, stats)
