# Heuristica

            # (0.50 * age_score) + 
            # (0.40 * cosseno_endereco) + 
            # (0.10 * cosseno_profissao)

import json
import pandas as pd

def heuristica(df):
    df = df.copy()
    df["age_score"] = df["age_score"].fillna(0)
    df["cos_endereco"] = df["cos_endereco"].fillna(0)
    df["cos_profissao"] = df["cos_profissao"].fillna(0)

    df["heuristica_score"] = (
        0.5 * df["age_score"] + 
        0.4 * df["cos_endereco"] + 
        0.1 * df["cos_profissao"]
    )

    return df

def rankear_por_heuristica(df):
    df = df.copy()
    df = df.sort_values(["qid", "heuristica_score"], ascending=[True, False]).reset_index(drop=True)

    df["rank"] = df.groupby("qid")["heuristica_score"].rank(method="first", ascending=False)
    return df

def metricas_heuristica(df):
    results = {}
    df = df.copy()

    rank_posi = (df[df["label"] == 1]
              .groupby("qid")["rank"]
              .min()
    )

    #mrr
    mrr = (1 / rank_posi).mean()

    #recall@k
    for k in [1, 5, 10, 15, 20]:
        hits_at_k = (rank_posi <= k).mean()
        results[f"Recall@{k}"] = hits_at_k

    results["MRR"] = mrr
    return results

    
for k in range(5):
    df_test = pd.read_csv(f"fold{k}_test.csv")
    df_heuristica = heuristica(df_test)
    df_rank = rankear_por_heuristica(df_heuristica)
    df_rank.to_csv(f"rank_heuristica_fold{k}.csv", index=False)
    metricas = metricas_heuristica(df_rank)
    for metrica, valor in metricas.items():
        print(f"Fold {k} - {metrica}: {valor:.4f}")
    df_heuristica.to_csv(f"heuristica_fold{k}.csv", index=False)