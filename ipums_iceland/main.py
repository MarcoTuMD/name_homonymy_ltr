import sys
from ipumspy import readers
import pandas as pd
from pprint import pprint
import numpy as np
import datetime
from contextlib import redirect_stdout
import json
import codes
import gc


data_path = "data.dat"
ddi_path = "data.xml"

ddi_codesbook = readers.read_ipums_ddi(ddi_path)
df = readers.read_microdata(ddi_codesbook, data_path)

# usado para gerar csv e json com as explicacoes das variaveis. so rodar um unica vez, depois comentar as linhas abaixo

# df.to_csv("ipums.csv", index=False)


# output = [str(x) for x in ddi_codesbook.data_description]

# with open("data_description.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, indent=4, ensure_ascii=False)

def delete_columns(df, columns_to_delete):
    df = df.drop(columns=columns_to_delete, errors="ignore")
    return df

def data_clean(df):
    df = delete_columns(df, ["YEAR", "HHWT", "PERWT", "GQ", "COUNTRY", "ENUMOP", "REGIONW"])
    #SAMPLE, SERIAL and PERNUM uniquely identify every person in the database
    df = df.copy()

    df['id'] = df['SAMPLE'].astype(str) + '_' + df['SERIAL'].astype(str) + '_' + df['PERNUM'].astype(str)
    df_pessoas_distintas = df[(
            df['COMMUNEIS'].notna() | 
            (~df['OCCISCO'].isin([97, 98, 99])) | 
            (~df['OCCHISCO'].isin([99999])) | 
            df[['COMMUNEIS', 'AGE']].notna().any(axis=1)
        )].copy()
    df_pessoas_distintas = df_pessoas_distintas[~df_pessoas_distintas["AGE"].isin([999])]
    df_pessoas_distintas = df_pessoas_distintas[~df_pessoas_distintas["RELATE"].isin([5, 9])]
    del df
    gc.collect()
    #df_pessoas_distintas.to_csv("ipums_new.csv", index=False)
    df_pessoas_distintas['COMMUNIESSTR'] = df_pessoas_distintas['COMMUNEIS'].map(codes.codigo_para_communeis)
    df_pessoas_distintas['OCCISCOSTR'] = df_pessoas_distintas['OCCISCO'].map(codes.codigo_para_profissoes)
    df_pessoas_distintas['OCCHISCOSTR'] = df_pessoas_distintas['OCCHISCO'].map(codes.codigo_para_occhisco)
    df_pessoas_distintas['RELATESTR'] = df_pessoas_distintas['RELATE'].map(codes.relation)
    df_pessoas_distintas["RELATEDSTR"] = df_pessoas_distintas["RELATED"].map(codes.related)
    
    cols_saida = ["id", "NAMEFRST", "NAMELAST", "AGE", "OCCHISCO", "OCCHISCOSTR", "COMMUNEIS", "COMMUNIESSTR"]
    df_pessoas = df_pessoas_distintas[cols_saida].copy()
    df_pessoas.to_csv("IpumsICELAND_distinct_pessoa_dados.csv", index=False)

    return df_pessoas_distintas

def homonimos(df):
    df_agg = df.groupby(['NAMEFRST', 'NAMELAST']).agg(
        qtd_homonimos_validos=('id', 'count'),
        lista_ids=('id', list),
    ).reset_index()

    df = pd.merge(df, df_agg, on=['NAMEFRST', 'NAMELAST'])

    df['lista_homonimos_validos'] = [
        [{'id': homonimo_id, 'is_person_2': int(homonimo_id == row_id)} for homonimo_id in lista]
        for row_id, lista in zip(df['id'], df['lista_ids'])
    ]
    df.drop(columns=['lista_ids'], inplace=True)
    #df.to_csv("ipums_homonimos.csv", index=False)

    chefes = df[df['RELATEDSTR'] == "Head"][['id', 'SERIAL']]
    membros = df[(df['RELATEDSTR'] != "Head")][['id', 'RELATE', 'SERIAL']]
    df_pares = pd.merge(
        membros,
        chefes,
        on='SERIAL',
        suffixes=('_parente', '_chefe')
    )

    df_pares = pd.merge(
        df_pares,
        df.loc[
            df['qtd_homonimos_validos'] > 1,
            ['id', 'qtd_homonimos_validos', 'lista_homonimos_validos']
        ],
        left_on='id_parente',
        right_on='id',
        how="inner"
    )
    df_pares.drop(columns=['id'], inplace=True)

    df_pares.to_csv("IpumsICELAND_pessoa_homonimo.csv", index=False)
    return df_pares

def endereco(df):
    df = df.copy()
    df['id_end'] = df['COMMUNEIS'].astype(str)
    df_endereco = df.groupby('id_end').agg(
        endereco=('COMMUNIESSTR', 'first'),
        qtd_pessoas_endereco=('PERNUM', 'count'),
        lista_ids=('id', list),
    ).reset_index()
    df_endereco.to_csv("IpumsICELAND_endereco.csv", index=False)
    return df_endereco

# def pessoas_distintas(df):
    # dados cadastrais das pessoas, uma linha por pessoa com id nome e dados
    #calc a idade em relacao a data do censo
    # na hr de fazer o join tem q ver se o par tem tudo
    #conferir se tem algum dado que todo mundo tem pra fzr join
    # <id, nome, ano nasc, id_ocupacao, nome_ocupacao, id_mun_residencia, nome_mun_residencia, id_mun_nacimento, nome_mun_nacimento>


def profissoes(df):
    df = df.copy()
    df['id_profissao'] = df['OCCISCO'].astype(str) + '_' + df['OCCHISCO'].astype(str)
    df_profissoes = df.groupby('id_profissao').agg(
        OCCISCO=('OCCISCOSTR', 'first'),
        OCCHISCO=('OCCHISCOSTR', 'first'),
        qtd_pessoas_profissao=('PERNUM', 'count'),
        lista_ids =('id', list),
    ).reset_index()
    df_profissoes.to_csv("IpumsICELAND_profissoes.csv", index=False)
    return df_profissoes

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

    # with open("estatisticas_relacionamento.json", "w", encoding="utf-8") as f:
    #     json.dump(stats_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    df_pessoas_distintas = data_clean(df)
    # for k in range(5):
    #     df_train = pd.read_csv(f"fold{k}_train.csv")
    #     stats_df, stats_dict = calculo_media(df_train, iteration=k)
    #     print(k, stats_df.head(5))

    df_homonimos = homonimos(df_pessoas_distintas)
    # df_check = df_homonimos.merge(
    #     df_pessoas_distintas[["id", "AGE"]],
    #         left_on="id_parente",
    #         right_on="id",
    #         how="left"
    #     )

    # print((df_check["AGE"] >= 900).sum())

    # df_check2 = df_homonimos.merge(
    #     df_pessoas_distintas[["id", "AGE"]],
    #     left_on="id_chefe",
    #     right_on="id",
    #     how="left"
    # )

    # print(df_check2["AGE"].isna().sum())

    df_endereco = endereco(df_pessoas_distintas)
    df_profissoes = profissoes(df_pessoas_distintas)
