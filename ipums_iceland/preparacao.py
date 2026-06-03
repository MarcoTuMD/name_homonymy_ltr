import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import ast
import re
import codes
from sklearn.metrics.pairwise import cosine_similarity
import json
from scipy.stats import t

_float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


#balanceamento de classe so no treino
#mudar o relacao pra string
#ao inves do exbedding colocar os scores do coseno e o age_score (nao botar coluna de idade)
#depois q tiver as infos calc heuristica
# os scores podem ser nulos, mas ai cria uma flag em todos (treino val teste) de missing score correspondente (coluna extra de 0 e 1)

df_homonimos = pd.read_csv("IpumsICELAND_pessoa_homonimo.csv")
df_homonimos["RELATESTR"] = df_homonimos["RELATE"].map(codes.relation)
df_profissao = pd.read_csv("IpumsICELAND_profissoes_com_embeddings.csv")
df_pessoas = pd.read_csv("IpumsICELAND_distinct_pessoa_dados.csv")
df_end = pd.read_csv("IpumsICELAND_endereco_com_embeddings.csv")
# parsear a lista. de homonimos
def parse_lista_homonimos(lista_str):
    if pd.isna(lista_str):
        return []
    s = str(lista_str)
    return ast.literal_eval(s)

def build_5fold_train_val_test(
    df: pd.DataFrame,
    qid_col: str = "qid",
    n_folds: int = 5,
    val_frac: float = 0.10,
    seed: int = 42
):
    df = df.copy()

    # qids únicos embaralhados
    qids = df[qid_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)

    n_qids = len(qids)
    qid_folds = np.array_split(qids, n_folds)

    df["fold"] = -1
    for k in range(n_folds):
        df.loc[df[qid_col].isin(qid_folds[k]), "fold"] = k

    by_iteration = {}

    for k in range(n_folds):
        test_qids = set(qid_folds[k])

        trainval_qids = np.concatenate([qid_folds[i] for i in range(n_folds) if i != k])

        # embaralha os qids de train+val
        rng_k = np.random.default_rng(seed + 1000 + k)
        trainval_qids = trainval_qids.copy()
        rng_k.shuffle(trainval_qids)

        # 10% dos qids de train+val vão para validação
        n_val = int(val_frac * len(trainval_qids))
        val_qids = set(trainval_qids[:n_val])
        train_qids = set(trainval_qids[n_val:])

        df_k = df.copy()
        df_k["split"] = "train"
        df_k.loc[df_k[qid_col].isin(val_qids), "split"] = "val"
        df_k.loc[df_k[qid_col].isin(test_qids), "split"] = "test"

        by_iteration[k] = df_k

    return df, by_iteration

def explode_candidates(df, k):
    #explodir pra uma linha por candidato
    chefes_household = df.groupby("SERIAL")["id_chefe"].nunique()
    household_validos = set(chefes_household[chefes_household==1].index)
    rows = []
    for _, row in df.iterrows():
        if row["SERIAL"] not in household_validos:
            continue
        qid = row['qid']
        id_chefe = row['id_chefe']
        id_parente = row['id_parente']
        relacao = row["RELATESTR"]

        for homonimo in row['lista_homonimos_validos']:
            id_homonimo = homonimo['id']
            label = 1 if id_homonimo == id_parente else 0
            rows.append({
                'qid': qid,
                'id_chefe': id_chefe,
                'relacao': relacao,
                'id_candidato': id_homonimo,
                'label': label
            })
    df_exploded = pd.DataFrame(rows)
    df_exploded.to_csv(f"pares{k}.csv", index=False)
    return df_exploded

# def make_pairwise(df):
#     df_pos = df[["qid", "id_chefe", "RELATE", "id_parente"]].copy()
#     df_pos = df_pos.rename(columns={"RELATE": "relacao", "id_parente": "id_candidato"})
#     df_pos["label"] = 1
#     return df_pos

def parse_emb(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        s = x.strip()
        sl = s.lower()
        if sl in ("", "[]", "nan", "none", "null"):
            return None
        nums = _float_re.findall(s)
        if not nums:
            return None
        return np.asarray([float(v) for v in nums], dtype=np.float32)
    return None

def join_end(df_pairs):
    df_pairs = df_pairs.copy()
    #juntar o endereco pra cada par
    df_pessoas["COMMUNEIS"] = df_pessoas["COMMUNEIS"].astype(str)
    df_end["id_end"] = df_end["id_end"].astype(str)
    df_pairs["id_chefe"] = df_pairs["id_chefe"].astype(str)
    df_pairs["id_candidato"] = df_pairs["id_candidato"].astype(str)

    df_pessoas_end = df_pessoas.merge(df_end[["id_end", "embeddings_base"]], left_on="COMMUNEIS", right_on="id_end", how="left")[["id", "embeddings_base"]].rename(columns={"embeddings_base": "emb_end"})
    df_pairs_endereco = df_pairs.merge(df_pessoas_end.rename(columns={"id": "id_chefe", "emb_end": "emb_end_chefe"}), on="id_chefe", how="left")
    df_pairs_endereco = df_pairs_endereco.merge(df_pessoas_end.rename(columns={"id": "id_candidato", "emb_end": "emb_end_candidato"}), on="id_candidato", how="left")
    return df_pairs_endereco

def cos_endereco(df):
    df = df.copy()
    def calc_cos(row):
        u = parse_emb(row["emb_end_chefe"])
        v = parse_emb(row["emb_end_candidato"])
        if u is None or v is None:
            return np.nan
        return float(cosine_similarity([u], [v])[0][0])

    df["cos_endereco"] = df.apply(calc_cos, axis=1)
    #remover as colunas de embedding depois de calcular o coseno
    df.drop(columns=["emb_end_chefe", "emb_end_candidato"], inplace=True)
    df["missing_cos_endereco"] = df["cos_endereco"].isna().astype(int)
    return df

def join_profissao(df_pairs_endereco):
    df_pairs_endereco = df_pairs_endereco.copy()
    profs = df_profissao.copy()

    df_pairs_endereco["id_chefe"] = df_pairs_endereco["id_chefe"].astype(str)
    df_pairs_endereco["id_candidato"] = df_pairs_endereco["id_candidato"].astype(str)

    # transformar lista_ids de string para lista
    profs["lista_ids"] = profs["lista_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # explodir: uma linha por pessoa
    profs_exploded = profs[["lista_ids", "embeddings_base"]].explode("lista_ids")
    profs_exploded = profs_exploded.rename(
        columns={"lista_ids": "id", "embeddings_base": "emb_prof"}
    )

    profs_exploded["id"] = profs_exploded["id"].astype(str)

    # join com chefe
    out = df_pairs_endereco.merge(
        profs_exploded.rename(columns={"id": "id_chefe", "emb_prof": "emb_prof_chefe"}),
        on="id_chefe",
        how="left"
    )

    # join com candidato
    out = out.merge(
        profs_exploded.rename(columns={"id": "id_candidato", "emb_prof": "emb_prof_candidato"}),
        on="id_candidato",
        how="left"
    )

    return out

def cos_profissao(df):
    df = df.copy()
    def calc_cos(row):
        u = parse_emb(row["emb_prof_chefe"])
        v = parse_emb(row["emb_prof_candidato"])
        if u is None or v is None:
            return np.nan
        return float(cosine_similarity([u], [v])[0][0])

    df["cos_profissao"] = df.apply(calc_cos, axis=1)
    #remover as colunas de embedding depois de calcular o coseno
    df.drop(columns=["emb_prof_chefe", "emb_prof_candidato"], inplace=True)
    df["missing_cos_profissao"] = df["cos_profissao"].isna().astype(int)
    return df

def join_idade(df):
    #juntar a idade pra cada par
    df = df.copy()

    df_pessoas["id"] = df_pessoas["id"].astype(str)
    df["id_chefe"] = df["id_chefe"].astype(str)
    df["id_candidato"] = df["id_candidato"].astype(str)
    df_pessoas["AGE"] = pd.to_numeric(df_pessoas["AGE"], errors="coerce")

    df_pessoa_idade = df_pessoas[["id", "AGE"]].rename(columns={"AGE": "idade"})
    df_idade = df.merge(df_pessoa_idade.rename(columns={"id": "id_chefe", "idade": "idade_chefe"}), on="id_chefe", how="left")
    df_idade = df_idade.merge(df_pessoa_idade.rename(columns={"id": "id_candidato", "idade": "idade_candidato"}), on="id_candidato", how="left")
    return df_idade

def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return 0
    return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2) / (2*sigma**2))

def age_score(df, k):
    df = df.copy()
    with open(f"estatisticas_relacao_tstudent_fold{k}.json", "r") as f:
        stats_relacao = json.load(f)

    def calc_age_score(row):
        rel = row["relacao"]

        if rel not in stats_relacao:
            return np.nan

        idade_chefe = row["idade_chefe"]
        idade_candidato = row["idade_candidato"]

        if pd.isna(idade_chefe) or pd.isna(idade_candidato):
            return np.nan

        diff = idade_candidato - idade_chefe

        params = stats_relacao[rel]

        df_param = params["df"]
        loc = params["loc"]
        scale = params["scale"]

        altura = t.pdf(diff, df=df_param, loc=loc, scale=scale)
        altura_max = t.pdf(loc, df=df_param, loc=loc, scale=scale)

        if altura_max == 0:
            return np.nan

        return altura / altura_max

    df["age_score"] = df.apply(calc_age_score, axis=1)

    df["missing_age_score"] = df["age_score"].isna().astype(int)
    return df

def prioridade_linha(df):
    df = df.copy()

    def has_embedding(x):
        if x is None:
            return 0
        if isinstance(x, float) and np.isnan(x):
            return 0
        if isinstance(x, list):
            return int(len(x) > 0)
        if isinstance(x, str):
            s = x.strip().lower()
            return int(s not in ("", "[]", "nan", "none", "null"))
        return 1

    df["prioridade"] = 0

    df["prioridade"] += df["emb_end_chefe"].apply(has_embedding)
    df["prioridade"] += df["emb_end_candidato"].apply(has_embedding)
    df["prioridade"] += df["emb_prof_chefe"].apply(has_embedding)
    df["prioridade"] += df["emb_prof_candidato"].apply(has_embedding)

    df["prioridade"] += df["idade_chefe"].notna().astype(int)
    df["prioridade"] += df["idade_candidato"].notna().astype(int)

    return df

def balancear_treino(df_train, max_ratio=10):
    balanced = []

    for qid, group in df_train.groupby("qid"):
        df_pos = group[group["label"] == 1]
        df_neg = group[group["label"] == 0]

        n_pos = len(df_pos)
        if n_pos == 0:
            continue

        max_neg = max_ratio * n_pos

        if len(df_neg) > max_neg:
            df_neg = (
                df_neg
                .sort_values(by="prioridade", ascending=False)
                .head(max_neg)
            )

        balanced.append(pd.concat([df_pos, df_neg]))

    return (
        pd.concat(balanced)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

def onehot_relacao(df):
    df = df.copy()
    df["is_child"] = (df["relacao"] == "Child").astype(int)
    df["is_spouse_partner"] = (df["relacao"] == "Spouse/partner").astype(int)
    df["is_other_relative"] = (df["relacao"] == "Other relative").astype(int)
    return df

def match_endereco(df):
    df = df.copy()
    df["match_exato_endereco"] = ((df["emb_end_chefe"].notna()) & (df["emb_end_candidato"].notna())).astype(int)
    return df

def build_model_datasets_from_dfk(df_k: pd.DataFrame,
                                 out_prefix: str,
                                 save: bool = True,
                                 k: int = 0):
    itens = explode_candidates(df_k, k)

    qid_to_split = df_k.set_index("qid")["split"].to_dict()
    itens["split"] = itens["qid"].map(qid_to_split)

    train_k = itens[itens["split"] == "train"].drop(columns=["split"]).copy()
    val_k   = itens[itens["split"] == "val"].drop(columns=["split"]).copy()
    test_k  = itens[itens["split"] == "test"].drop(columns=["split"]).copy()


    df_train = join_end(train_k)
    df_val   = join_end(val_k)
    df_test  = join_end(test_k)

    df_train = join_profissao(df_train)
    df_val   = join_profissao(df_val)
    df_test  = join_profissao(df_test)

    df_train = join_idade(df_train)
    df_val   = join_idade(df_val)
    df_test  = join_idade(df_test)


    df_train = prioridade_linha(df_train)
    df_train = balancear_treino(df_train, max_ratio=10)

    df_train = match_endereco(df_train)
    df_val   = match_endereco(df_val)
    df_test  = match_endereco(df_test)

    df_train = cos_endereco(df_train)
    df_val   = cos_endereco(df_val)
    df_test  = cos_endereco(df_test)
    df_train = cos_profissao(df_train)
    df_val   = cos_profissao(df_val)
    df_test  = cos_profissao(df_test)
    df_train = age_score(df_train, k)
    df_val   = age_score(df_val, k)
    df_test  = age_score(df_test, k)

    df_train = onehot_relacao(df_train)
    df_val   = onehot_relacao(df_val)
    df_test  = onehot_relacao(df_test)

    if save:
        df_train.to_csv(f"{out_prefix}_train.csv", index=False)
        df_val.to_csv(f"{out_prefix}_val.csv", index=False)
        df_test.to_csv(f"{out_prefix}_test.csv", index=False)

    return df_train, df_val, df_test

def build_model_train(df_k: pd.DataFrame, 
                                 out_prefix: str,
                                 save: bool = True,
                                 k: int = 0):
    itens = explode_candidates(df_k)
    qid_to_split = df_k.set_index("qid")["split"].to_dict()
    itens["split"] = itens["qid"].map(qid_to_split)

    train_k = itens[itens["split"] == "train"].drop(columns=["split"]).copy()


    df_train = join_end(train_k)

    df_train = join_profissao(df_train)

    df_train = join_idade(df_train)

    df_train = onehot_relacao(df_train)
    df_train = match_endereco(df_train)
    if save:
        df_train.to_csv(f"{out_prefix}_train_nao_balanceado.csv", index=False)

    return df_train
#join com o distinct pessoa pra por idade

df_homonimos['lista_homonimos_validos'] = df_homonimos['lista_homonimos_validos'].apply(parse_lista_homonimos)
#colocar o qid pra manter sequencial
df_homonimos = df_homonimos.reset_index(drop=True)
#shuffle
df_homonios_shuffled = df_homonimos.sample(frac=1, random_state=42).reset_index(drop=True)
df_homonios_shuffled['qid'] = df_homonios_shuffled.index.astype(int)


# df_explode = explode_candidates(df_homonios_shuffled)
# df_homonios_shuffled["peso_qid"] = df_homonios_shuffled["lista_homonimos_validos"].apply(len)
df_all, fold_dfs = build_5fold_train_val_test(df_homonios_shuffled, qid_col = "qid", n_folds=5, val_frac=0.10, seed=42)

# salva por fold
for k, df_k in fold_dfs.items():
    df_k.to_csv(f"homonimos_fold{k}.csv", index=False)
    df_train, df_val, df_test = build_model_datasets_from_dfk(
        df_k,
        out_prefix=f"fold{k}",
        save=True,
        k=k
    )
# for k, df_k in fold_dfs.items():
#     df_k.to_csv(f"homonimos_fold{k}.csv", index=False)
#     df_train = build_model_train(df_k, out_prefix=f"fold{k}", save=True, k=k)
# na hora de criar o treino, fazer proporcao de 1 pra 10 (1 posi pra cada 10 negativos) pra balancear o treino. Na validacao e teste manter a proporcao original
#calc as metricas no treino antes de balancear
# a prioridade pra cortar e ter mais colunas preenchidas
