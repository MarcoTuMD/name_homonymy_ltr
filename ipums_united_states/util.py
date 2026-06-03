import psutil
import os
import pandas as pd
nome_arquivo = "out.txt"
dir_path = "dados"

def print_out(texto):
    with open(nome_arquivo, "a", encoding="utf-8") as f:
        f.write(f"{texto}\n")

def monitorar_ram(txt = ""):
    processo = psutil.Process(os.getpid())
    memoria_bytes = processo.memory_info().rss
    memoria_mb = memoria_bytes / (1024 ** 2)
    memoria_gb = memoria_bytes / (1024 ** 3)
    
    status = f"{txt}\nUso de RAM Atual: {memoria_mb:.2f} MB ({memoria_gb:.2f} GB)"
    print_out(status)
    return status

def salvar_dataframe(df, nome_arquivo, dire=dir_path):
    caminho_completo = os.path.join(dire, nome_arquivo)
    extensao = nome_arquivo.split('.')[-1].lower()

    print(f"Salvando {len(df):,} registros em {caminho_completo}...")

    if extensao == 'csv':
        df.to_csv(caminho_completo, index=False, sep=',', encoding='utf-8')
    elif extensao == 'parquet':
        df.to_parquet(caminho_completo, index=False, engine='pyarrow')
    elif extensao in ['xlsx', 'xls']:
        df.to_excel(caminho_completo, index=False)
    elif extensao == 'json':
        df.to_json(caminho_completo, orient='records', indent=4)
    elif extensao == 'jsonl':
        df.to_json(caminho_completo, orient='records', lines=True)
    elif extensao == 'pkl':
        df.to_pickle(caminho_completo)
    else:
        print(f"extensão '.{extensao}' não suportada.")
        return

    print("Concluído!")

def carregar_dataframe(caminho_arquivo, dire=dir_path, colunas=None, tipos=None):
    caminho_completo = os.path.join(dire, caminho_arquivo)
    if not os.path.exists(caminho_completo):
        print(f"Erro: O arquivo '{caminho_completo}' não foi encontrado.")
        return None

    extensao = caminho_arquivo.split('.')[-1].lower()
    if extensao == caminho_arquivo:
        extensao = 'parquet'

    print(f"lendo ({extensao}): {caminho_completo}...")

    try:
        if extensao  == 'csv':
            df = pd.read_csv(caminho_completo, low_memory=False, usecols=colunas)
        elif extensao == 'parquet':
            df = pd.read_parquet(caminho_completo, columns=colunas)
        elif extensao in ['xlsx', 'xls']:
            df = pd.read_excel(caminho_completo)
        elif extensao == 'json':
            df = pd.read_json(caminho_completo)
        elif extensao == 'jsonl':
            df = pd.read_json(caminho_completo, lines=True)
        elif extensao == 'pkl':
            df = pd.read_pickle(caminho_completo)
        else:
            print(f"extensao '.{extensao}' não suportada .")
            return None
        
        print(f" {len(df):,} registros carregados.")
        return df
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

def print_full(df, n=None, n_lista=5):
    df_to_print = df.head(n) if n is not None else df

    def format_list(x):
        if isinstance(x, list):
            if len(x) > n_lista:
                return str(x[:n_lista])[:-1] + ", ...]"
            return str(x)
        return x

    cols_object = df_to_print.select_dtypes(include=['object']).columns
    df_display = df_to_print.copy()
    for col in cols_object:
        df_display[col] = df_display[col].apply(format_list)
    
    with pd.option_context(
        'display.max_rows', None, 
        'display.max_columns', None, 
        'display.width', 2000, 
        'display.max_colwidth', None,
        'display.colheader_justify', 'center'
    ):
        print(df_display)
    
    print("-" * 30)
    print(f"TAMANHO: {df.shape[0]} linhas x {df.shape[1]} colunas")
    if n is not None:
        print(f"Nota: Exibindo apenas as primeiras {n} linhas.")
    print("-" * 30)
