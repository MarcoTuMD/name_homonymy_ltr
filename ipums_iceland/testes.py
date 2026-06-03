import pandas as pd
import ast

def quantidade_dados():
    #quantidades de dados em cada treino val e teste
    for k in range(5):
        for split in ["train", "val", "test"]:
            df = pd.read_csv(f"fold{k}_{split}.csv")
            num_qid = df["qid"].nunique()
            print(f"Fold {k} - {split}: {num_qid} qids")

def quantidade_dados_pair():
    #quantidades de dados em cada treino val e teste
    for k in range(5):
        df_train = pd.read_csv(f"fold{k}_train.csv")
        print(f"Fold {k} - train pair-wise: {len(df_train)}")
        print(f"Fold {k} - train pair-sise por qid: {df_train['qid'].nunique()}")


def quantidade_por_label_train():
    #quantidade de positivos e negativos em cada treino
    for k in range(5):
        df = pd.read_csv(f"fold{k}_train.csv")
        label_counts = df["label"].value_counts()
        print(f"Fold {k} - Train:")
        print(label_counts)

def media_desvio():
    for k in range(5):
        for split in ["val", "test"]:
            df = pd.read_csv(f"fold{k}_{split}.csv")

            # número de candidatos por query
            candidatos_por_qid = df.groupby("qid").size()

            media = candidatos_por_qid.mean()
            desvio = candidatos_por_qid.std()

            print(f"Fold {k} - {split}")
            print("  Média candidatos:", media)
            print("  Desvio padrão:", desvio)

def mediana_percentil():
    df = pd.read_csv("IpumsICELAND_pessoa_homonimo.csv")
    df["n_homonimos"] = df["lista_homonimos_validos"].apply(len)
    df["n_homonimos_minus1"] = df["n_homonimos"] - 1
    mediana = df["n_homonimos_minus1"].median()
    p90 = df["n_homonimos_minus1"].quantile(0.90)
    p10 = df["n_homonimos_minus1"].quantile(0.10)
    p5  = df["n_homonimos_minus1"].quantile(0.05)

    print("Mediana:", mediana)
    print("Percentil 90:", p90)
    print("Percentil 10:", p10)
    print("Percentil 5:", p5)

quantidade_dados()
quantidade_por_label_train()
media_desvio()
quantidade_dados_pair()
mediana_percentil()