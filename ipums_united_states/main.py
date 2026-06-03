from data import *
from xgb import *
from heuristica import *
from util import *
from process import *
import sys
from ipumspy import readers
from pprint import pprint
import numpy as np
import locale

import datetime
from contextlib import redirect_stdout

import os
import subprocess

pd.options.display.float_format = lambda x: f'{x:,.6f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
data_path = "sample.dat"
ddi_path = "ddi.xml"

sys.stdout = open('ipums2.log', 'w')
sys.stderr = sys.stdout

nome_arquivo = "out.txt"
hora_inicio = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

with open(nome_arquivo, "w", encoding="utf-8") as f:
    f.write(f"=== Início em: {hora_inicio} ===\n\n")


def pares():
    ddi = readers.read_ipums_ddi(ddi_path)
    ipums_df = readers.read_microdata(ddi, data_path)
    process_pares(ipums_df)

def models():

    pares_df = carregar_dataframe("IpumsUSA_pessoa_homonimo_folds.parquet", colunas=['id_parente', 'RELATED', 'id_chefe', 'id', 'lista_homonimos_validos', 'fold', 'qtd_homonimos_validos'])
    pessoas_df = carregar_dataframe("IpumsUSA_distinct_pessoa_dados.csv", colunas=['id', 'STATEICP', 'COUNTYICP', 'CITY', 'AGE', 'id_endereco', 'id_profissao', 'qtd_homonimos_validos'])
    adress_embedding = carregar_dataframe("adress_embeddings", colunas=['id_endereco', 'embeddings_base'])
    occupation_embedding = carregar_dataframe("occupation_embeddings", colunas=['id_profissao', 'embeddings_base'])

    process_heuristica_final(pessoas_df, pares_df, adress_embedding, occupation_embedding)
    monitorar_ram()

def folds():
    pares_df = carregar_dataframe("IpumsUSA_pessoa_homonimo.parquet")
    pessoas_df = carregar_dataframe("IpumsUSA_distinct_pessoa_dados.csv")
    adress_embedding = carregar_dataframe("adress_embeddings", colunas=['id_endereco', 'embeddings_base'])
    occupation_embedding = carregar_dataframe("occupation_embeddings", colunas=['id_profissao', 'embeddings_base'])

    pares_df = set_folds(pares_df)

    salvar_dataframe(pares_df, "IpumsUSA_pessoa_homonimo_folds.parquet")

if __name__ == "__main__":
    pares()
    # folds()
    # models()

    hora_final = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print_out(f"\n\n=== Fim em: {hora_final} ===\n\n")


