import pandas as pd
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import sys
import numpy as np
from tabulate import tabulate
import gc
import ast
from adress_code import *

from scipy import stats
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRanker
import lightgbm as lgb
import shap
from scipy.stats import wilcoxon
import json

from util import *

SPLITS = [
    {"id": 1, "train":[1,2,3,4], "test":5},
    {"id": 2, "train":[1,2,3,5], "test":4},
    {"id": 3, "train":[1,2,4,5], "test":3},
    {"id": 4, "train":[1,3,4,5], "test":2},
    {"id": 5, "train":[2,3,4,5], "test":1},
]


def get_target_cols(df, suffix):
    cols = {
        'id': f'join_id_{suffix}',
        'AGE': f'age_{suffix}',
        'CITY': f'city_{suffix}',
        'COUNTYICP': f'county_{suffix}',
        'STATEICP': f'state_{suffix}',
        'id_endereco': f'id_end_{suffix}',
        'id_profissao': f'id_prof_{suffix}'
    }
    return df[list(cols.keys())].rename(columns=cols)

def process_statistic(df_pessoas, df_pares, adress, occupation):
    print("INICIANDO: ")
    print("Geral")
    print(f"DF total pares: {len(df_pares)} pares")
    print(f"Quantidade pessoas distintas: {len(df_pessoas)}")
    print(f"Quantidade endereços distintos: {len(adress)}")
    print(f"Quantidade profissões distintas distintos: {len(occupation)}")

    homonimos_unicos = df_pares['qtd_homonimos_validos']
    outros_homonimos = homonimos_unicos - 1
    estatisticas = {
        'Média': outros_homonimos.mean(),
        'Desvio Padrão': outros_homonimos.std(),
        'Mediana': outros_homonimos.median(),
        'Percentil 90': np.percentile(outros_homonimos, 90),
        'Percentil x [5]': stats.percentileofscore(outros_homonimos, 5),
        'Percentil x [10]': stats.percentileofscore(outros_homonimos, 10),
        'Somatório (sem n-1)': homonimos_unicos.sum()
    }
    print(f"{'MÉTRICA [pares] (N-1)':<20} | {'VALOR':<10}")
    print("-" * 35)
    for metrica, valor in estatisticas.items():
        print(f"{metrica:<20} | {valor:<10.4f}")
    
    homonimos_unicos = df_pessoas['qtd_homonimos_validos']
    outros_homonimos = homonimos_unicos - 1
    estatisticas = {
        'Média': outros_homonimos.mean(),
        'Desvio Padrão': outros_homonimos.std(),
        'Mediana': outros_homonimos.median(),
        'Percentil 90': np.percentile(outros_homonimos, 90)
    }
    print(f"{'MÉTRICA [pessoas] (N-1)':<20} | {'VALOR':<10}")
    print("-" * 35)
    for metrica, valor in estatisticas.items():
        print(f"{metrica:<20} | {valor:<10.4f}")

        

def process_heuristica_final(df_pessoas, df_pares, adress, occupation):
    
    list_statistic = {}
    
    print("INICIANDO: ", flush=True)
    print("Geral")
    print(f"DF total pares: {len(df_pares)} pares")
    print(f"Quantidade pessoas distintas: {len(df_pessoas)}")
    print(f"Quantidade endereços distintos: {len(adress)}")
    print(f"Quantidade profissões distintas distintos: {len(occupation)}")

    list_statistic["total_pares"] = len(df_pares)
    list_statistic["enderecos"] = len(adress)
    list_statistic["profissoes"] = len(occupation)

    homonimos_unicos = df_pares['qtd_homonimos_validos']
    outros_homonimos = homonimos_unicos - 1
    estatisticas = {
        'Média': outros_homonimos.mean(),
        'Desvio Padrão': outros_homonimos.std(),
        'Mediana': outros_homonimos.median(),
        'Percentil 90': np.percentile(outros_homonimos, 90),
        'Percentil x [5]': stats.percentileofscore(outros_homonimos, 5),
        'Percentil x [10]': stats.percentileofscore(outros_homonimos, 10),
        'Somatório (sem n-1)': homonimos_unicos.sum()
    }
    print(f"{'MÉTRICA [pares] (N-1)':<20} | {'VALOR':<10}")
    print("-" * 35)
    for metrica, valor in estatisticas.items():
        print(f"{metrica:<20} | {valor:<10.4f}")

    list_statistic["geral"] = estatisticas

    del homonimos_unicos, outros_homonimos
    gc.collect()

    for split in SPLITS:
        estatistica_split = {}
        print(f"---------------------------------- SPLIT {split.get('id')} -----------------------------------------------", flush=True)

        total_train_df = df_pares[df_pares['fold'].isin(split['train'])]
        train_indices = total_train_df.sample(frac=0.9, random_state=42).index

        train_df = total_train_df.loc[train_indices].copy()
        valid_df = total_train_df.loc[~total_train_df.index.isin(train_indices)].copy()
        test_df = df_pares[df_pares['fold'] == split['test']]
        del total_train_df 
        gc.collect()

        estatistica_split["data_treino"] = len(train_df)
        estatistica_split["data_validacao"] = len(valid_df)
        estatistica_split["data_teste"] = len(test_df)

        train_pairwise, relacoes = prepare_pair_wise(split, train_df, df_pessoas, adress, occupation, mode="train")
        valid_pairwise, _ = prepare_pair_wise(split, valid_df, df_pessoas, adress, occupation, mode="valid", parametros=relacoes)
        test_pairwise, _ = prepare_pair_wise(split, test_df, df_pessoas, adress, occupation, mode="test", parametros=relacoes)
        
        print("\n")
        print(f"Tamanho do df de treino: {len(train_pairwise)}")
        print(f"Tamanho do df de validação: {len(valid_pairwise)}")
        print(f"Tamanho do df de teste: {len(test_pairwise)}")
        print("\n", flush=True)

        estatistica_split["data_treino_pairwise"] = len(train_pairwise)
        estatistica_split["data_treino_label1"] = len(train_pairwise[train_pairwise["label"] == 1])
        estatistica_split["data_treino_label0"] = len(train_pairwise[train_pairwise["label"] == 0])

        # preparação das colunas para os modelos
        train_pairwise, features = prepare_models_colluns(split, train_pairwise, df_pessoas, adress, occupation)
        valid_pairwise, _ = prepare_models_colluns(split, valid_pairwise, df_pessoas, adress, occupation)
        test_pairwise, _ = prepare_models_colluns(split, test_pairwise, df_pessoas, adress, occupation)
        
        #dados gerais
        homonimos_unicos = valid_pairwise[valid_pairwise["label"] == 1]['qtd_homonimos_validos']
        outros_homonimos = homonimos_unicos - 1
        print(f"[Valid] Média de lista de homonimos [n-1]: {outros_homonimos.mean()}")
        print(f"[Valid] Desvio de lista de homonimos [n-1]: {outros_homonimos.std()}")
        estatistica_split["data_valid_media"] = outros_homonimos.mean()
        estatistica_split["data_valid_desvio"] = outros_homonimos.std()
        homonimos_unicos = test_pairwise[test_pairwise["label"] == 1]['qtd_homonimos_validos']
        outros_homonimos = homonimos_unicos - 1
        print(f"[Teste] Média de lista de homonimos [n-1]: {outros_homonimos.mean()}")
        print(f"[Teste] Desvio de lista de homonimos [n-1]: {outros_homonimos.std()}", flush=True)
        estatistica_split["data_test_media"] = outros_homonimos.mean()
        estatistica_split["data_test_desvio"] = outros_homonimos.std()

        #salvando os dados de treino, validação e teste
        n_split = split.get('id')
        salvar_dataframe(train_pairwise, f"split-{n_split}-train.csv", dire=f"results/{n_split}")
        salvar_dataframe(valid_pairwise, f"split-{n_split}-valid.csv", dire=f"results/{n_split}")
        salvar_dataframe(test_pairwise, f"split-{n_split}-test.csv", dire=f"results/{n_split}")

        # heuristica
        label = "heuristic"
        print("Inference heuristic", flush=True)
        heuristica_metrics = inferencia (test_pairwise, label, split, features, None, mode="heuristic")
        estatistica_split["heuristica"] = heuristica_metrics

        #xgboost
        label = "xgboost_ranker"
        print("Training model XGBoost", flush=True)
        xgb_model = train_xgb_ranker(train_pairwise, valid_pairwise, features, split.get("id"))
        print("Inference model XGBoost", flush=True)
        xgboost_metrics = inferencia (test_pairwise, label, split, features, xgb_model, mode="ranker")
        estatistica_split["xgboost"] = xgboost_metrics

        # lgbm (mesmos dados)
        label = "lgbm_ranker"
        print("Training model LGBM", flush=True)
        model_lgbm = train_lgbm_ranker(train_pairwise, valid_pairwise, features, split.get("id"))
        print("Inference model LGBM", flush=True)
        lgbm_metrics = inferencia (test_pairwise, label, split, features, model_lgbm, mode="ranker")
        estatistica_split["lgbm"] = lgbm_metrics
        lgbm_explainer (test_pairwise, split, features, model_lgbm)

        # regressão logística
        features.append("is_missing_city")
        label = "regressao_ranker"
        print("Training model Regressão Logistica", flush=True)
        reg_log_model = train_reg_log_classifier(train_pairwise, valid_pairwise, features, split.get("id"))
        print("Inference model Regressão Logistica", flush=True)
        regressao_metrics = inferencia (test_pairwise, label, split, features, reg_log_model, mode="classifier")
        estatistica_split["regressao"] = regressao_metrics

        list_statistic[split.get("id")] = estatistica_split

    metrics = ["recall@1", "recall@5", "recall@10", "recall@15", "recall@20", "mrr"]
    hipotese = {}
    for metric in metrics:
        metric_hipotese = {}
        recall_model_xgb = [list_statistic[i]["xgboost"][metric] for i in range(1, 6)]
        recall_model_lgbm = [list_statistic[i]["lgbm"][metric] for i in range(1, 6)]
        stat, p_value = wilcoxon(recall_model_xgb, recall_model_lgbm, alternative='greater')

        print(metric)
        print("    Wilcoxon statistic:", stat)
        print("    p-value:", p_value)
        metric_hipotese["stat"] = stat
        metric_hipotese["p_value"] = p_value
        hipotese[metric] = metric_hipotese
    list_statistic["hipotese"] = hipotese


    with open('results/result.json', 'w', encoding='utf-8') as f:
        json.dump(list_statistic, f, indent=4, ensure_ascii=False, default=json_serial)

def json_serial(obj):
    if isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
def inferencia_batch (df_test, features, model, mode):

    X_test = df_test[features]

    # heuristica
    if mode == "heuristic":
        cols_to_fill = []
        X_test[cols_to_fill] = X_test[cols_to_fill].fillna(0)
        scores = (
            0.50 * X_test["age_score"] +
            0.40 * X_test["cos_end"] +
            0.10 * X_test["cos_prof"]
        )

    #classifier
    if mode == "classifier":
        cols_to_fill = [
            "city_match"
        ]
        X_test[cols_to_fill] = X_test[cols_to_fill].fillna(0)

        scores = model.predict_proba(X_test)[:, 1]

    if mode == "ranker":
        scores = model.predict(X_test)

    df_test["score"] = scores

    metrics_per_id = df_test.groupby(["id"]).apply(ranking_metrics)
    return metrics_per_id.reset_index(drop=True)

def inferencia (df_test, label, split, features, model, mode="ranker"):
    if df_test is None or df_test.empty:
        print(f"O DataFrame de teste para {label} {split} está vazio.")
        return None

    batch_metrics = inferencia_batch(df_test, features, model, mode)
    
    df_results = pd.DataFrame(batch_metrics)
    
    output_file = f"results/{split.get('id')}/result--{label}--{split.get('id')}.csv"
    df_results.to_csv(output_file, index=False)
    
    df_results = df_results[df_results["mrr"].notna()]
    
    metrics_to_print = ["recall@1", "recall@5", "recall@10", "recall@15", "recall@20", "mrr"]
    
    existing_cols = [c for c in metrics_to_print if c in df_results.columns]
    metrics_mean = df_results[existing_cols].mean()

    print(f"\n{'='*40}")
    print(f"RESUMO DE PERFORMANCE DIRETA")
    print(f"Label: {label} | Split: {split}")
    print(f"{'='*40}")
    print(metrics_mean)
    print(f"{'='*40}\n")
    
    return metrics_mean

def ranking_metrics(group):
    group_id = group.name

    group = group.sort_values("score", ascending=False)

    labels = group["label"].values
    relevant_positions = np.where(labels == 1)[0]

    n_relevant = len(relevant_positions)

    # Se não há relevantes, estoura erro
    if n_relevant == 0:
        #print(f"Nenhum parente encontrado id={group_id}")
        return pd.Series({
            "id": group_id,
            "recall@1": np.nan,
            "recall@5": np.nan,
            "recall@10": np.nan,
            "recall@15": np.nan,
            "recall@20": np.nan,
            "mrr": np.nan,
            "count": np.nan
        })

    # Recall@1
    recall_at_1 = int(labels[0] == 1) / n_relevant

    # Recall@5
    recall_at_5 = labels[:5].sum() / n_relevant

    # Recall@10
    recall_at_10 = labels[:10].sum() / n_relevant

    # Recall@10
    recall_at_15 = labels[:15].sum() / n_relevant

    # Recall@10
    recall_at_20 = labels[:20].sum() / n_relevant

    # MRR
    first_relevant_rank = relevant_positions[0] + 1
    mrr = 1.0 / first_relevant_rank

    return pd.Series({
        "id": group_id,
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "recall@15": recall_at_15,
        "recall@20": recall_at_20,
        "mrr": mrr,
        "count": len(labels)
    })

def train_xgb_ranker (df_train, df_valid, features, split):


    df_train = df_train.sort_values("id")
    df_valid = df_valid.sort_values("id")
    
    X_train = df_train[features]
    Y_train = df_train["label"]

    X_valid = df_valid[features]
    Y_valid =  df_valid["label"]
   
    group_train = (
        df_train
        .groupby("id")
        .size()
        .values
    )
    
    group_valid = (
        df_valid
        .groupby("id")
        .size()
        .values
    )

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        Y_train,
        group=group_train,
        eval_set=[(X_valid, Y_valid)],
        eval_group=[group_valid],
        verbose=20
    )

    model.save_model(f"results/{split}/modelo_xgb.json")

    return model

def train_lgbm_ranker(df_train, df_valid, features, split):
    df_train = df_train.sort_values("id")
    df_valid = df_valid.sort_values("id")
    
    X_train = df_train[features]
    Y_train = df_train["label"]

    X_valid = df_valid[features]
    Y_valid =  df_valid["label"]
    
    group_train = df_train.groupby("id").size().tolist()
    group_valid = df_valid.groupby("id").size().tolist()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train,
        Y_train,
        group=group_train,
        eval_set=[(X_valid, Y_valid)],
        eval_group=[group_valid],
        eval_at=[1, 5, 10],
        callbacks=[
            lgb.log_evaluation(period=20)
        ]
    )

    model.booster_.save_model(f"results/{split}/modelo_lgbm.txt")

    return model

def lgbm_explainer (df_test, split, features, model_lgbm):
    explainer = shap.TreeExplainer(model_lgbm)
    
    X = df_test[features]
    shap_values = explainer.shap_values(X)
    
    importance = np.abs(shap_values).mean(axis=0)
    
    df_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    total = df_importance["importance"].sum()
    
    feature_groups = {
        "address": ["county_match", "state_match", "city_match", "cos_end"],
        "age": ["age_score"],
        "profession": ["cos_prof"],
        "relationship_indicators": [
            'is_301',
            'is_201',
            'is_901',
            'is_701',
            'is_501',
            'is_1031',
            'is_801',
            'is_601',
            'is_303',
            'is_401',
            'is_1041',
            'is_1021',
            'is_302',
            'is_1001', 
            'is_1011',
            'is_1034',
            'is_903',
            'is_502',
            'is_702',
            'is_1032',
            'is_904',
            'is_304',
            'is_402'
        ]
    }    

    group_importance = []
    for group, features in feature_groups.items():
        
        imp = df_importance[
            df_importance["feature"].isin(features)
        ]["importance"].sum()
        
        group_importance.append({
            "group": group,
            "importance": imp
        })
    
    df_group = pd.DataFrame(group_importance)
    
    df_group["relative_importance"] = (
        100 * (df_group["importance"] / total)
    )
    print("LGBM IMPORTANCE ====================")
    print(df_group, flush=True)
    output_file = f"results/{split.get('id')}/lgbm_explainer.csv"
    df_group.to_csv(output_file, index=False)

def train_reg_log_classifier (df_train, df_valid, features, split):
    cols_to_fill = [
        "city_match"
    ]

    X_train = df_train[features]
    X_train[cols_to_fill] = X_train[cols_to_fill].fillna(0)
    Y_train = df_train["label"]

    X_valid = df_valid[features]
    X_valid[cols_to_fill] = X_valid[cols_to_fill].fillna(0)
    Y_valid =  df_valid["label"]

    scaler = StandardScaler()

    n_pos = (Y_train == 1).sum()
    n_neg = (Y_train == 0).sum()
    
    class_weight = {
        0: 1.0,
        1: n_neg / n_pos
    }
    
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        class_weight=class_weight,
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ("scaler", scaler),
        ("clf", model)
    ])

    pipeline.fit(X_train, Y_train)

    y_valid_proba = pipeline.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(Y_valid, y_valid_proba)
    
    import joblib
    joblib.dump(model, f"results/{split}/modelo_rl.pkl")

    print("Validation AUC:", auc)

    return pipeline

def prepare_models_colluns(split, df_pair_wise, df_pessoas, adress, occupation):


    # df_pair_wise['is_missing_street'] = (
    #     df_pair_wise['street_target'].isna() | df_pair_wise['street_target'].isin(['NULL', "NaN", "<NA>"]) |
    #     df_pair_wise['street_homonimo'].isna() | df_pair_wise['street_homonimo'].isin(['NULL', "NaN", "<NA>"])
    # ).astype(int)
    df_pair_wise['is_missing_city'] = (
        df_pair_wise['city_target'].isna() | df_pair_wise['city_target'].isin([0, 0.0, 'NaN', 'nan']) |
        df_pair_wise['city_homonimo'].isna() | df_pair_wise['city_homonimo'].isin([0, 0.0, 'NaN', 'nan'])
    ).astype(int)
    # df_pair_wise["street_match"] = (
    #     (df_pair_wise['street_target'] == df_pair_wise['street_homonimo']) & 
    #     (df_pair_wise['is_missing_street'] == 0)
    # ).astype(int)
    df_pair_wise['city_match'] = (
        (df_pair_wise['city_target'] == df_pair_wise['city_homonimo']) & 
        (df_pair_wise['is_missing_city'] == 0)
    ).astype(int)
    df_pair_wise['county_match'] = (
        (df_pair_wise['county_target'] == df_pair_wise['county_homonimo'])
    ).astype(int)
    df_pair_wise['state_match'] = (
        (df_pair_wise['state_target'] == df_pair_wise['state_homonimo'])
    ).astype(int)

    #hot encoding:
    
    codigos_hot = codigos_familiares_desejados.copy()
    if 101 in codigos_hot:
        codigos_hot.remove(101)

    features_hot = []
    for code in codigos_hot:
        label = "is_" + str(code)
        df_pair_wise[label] = (df_pair_wise['RELATED'] == code).astype(int)
        features_hot.append(label)

    features = [
        "age_score", 
        "city_match", 
        "county_match",
        "state_match",
        "cos_end",
        "cos_prof",
        *features_hot
    ]


    return df_pair_wise, features

def prepare_pair_wise(split, source_df, df_pessoas, adress, occupation, mode="train", parametros={}):

    print(f"[{mode}] Pares verdadeiros: {len(source_df)}")

    df_pair_wise = source_df.explode("lista_homonimos_validos").reset_index(drop=True)     
    df_pair_wise['id_homonimo'] = df_pair_wise['lista_homonimos_validos'].str.get('id')
    df_pair_wise['label'] = (df_pair_wise['id_homonimo'] == df_pair_wise['id_parente']).astype(int)
    df_pair_wise.drop(columns=["lista_homonimos_validos"], inplace=True)

    del source_df #limpa a referencia original 
    gc.collect()
    
    print(f"[{mode}] Pares explodidos pré-balanciamento: {len(df_pair_wise)}")

    #dados homonimo
    homonimo_data = get_target_cols(df_pessoas, "homonimo")
    df_pair_wise = df_pair_wise.merge(homonimo_data, left_on="id_homonimo", right_on="join_id_homonimo", how="left").drop(columns=["join_id_homonimo"])
    gc.collect()

    # dados_target
    target_data = get_target_cols(df_pessoas, "target")
    df_pair_wise = df_pair_wise.merge(target_data, left_on="id_chefe", right_on="join_id_target", how="left").drop(columns=["join_id_target"])
    gc.collect()

    
    #embeddings adress
    id_to_index = pd.Series(
        np.arange(len(adress)), 
        index=adress["id_endereco"]
    )
    E = np.vstack(adress["embeddings_base"].values).astype(np.float16)
    idx_target = id_to_index.reindex(df_pair_wise["id_end_target"]).values
    idx_hom = id_to_index.reindex(df_pair_wise["id_end_homonimo"]).values
    valid_mask = ~np.isnan(idx_target) & ~np.isnan(idx_hom)
    idx_target_valid = idx_target[valid_mask].astype(np.int64)
    idx_hom_valid = idx_hom[valid_mask].astype(np.int64)
    cos_sim = np.full(len(df_pair_wise), np.nan, dtype=np.float32)
    batch_size = 500_000
    for i in range(0, len(idx_target_valid), batch_size):
        end = i + batch_size
        
        v1 = E[idx_target_valid[i:end]].astype(np.float32)
        v2 = E[idx_hom_valid[i:end]].astype(np.float32)
        
        cos_sim[np.where(valid_mask)[0][i:end]] = np.einsum("ij,ij->i", v1, v2)
    df_pair_wise["cos_end"] = cos_sim

    #occupation embeddings
    id_to_index = pd.Series(
        np.arange(len(occupation)), 
        index=occupation["id_profissao"]
    )
    E = np.vstack(occupation["embeddings_base"].values).astype(np.float16)
    idx_target = id_to_index.reindex(df_pair_wise["id_prof_target"]).values
    idx_hom = id_to_index.reindex(df_pair_wise["id_prof_homonimo"]).values
    valid_mask = ~np.isnan(idx_target) & ~np.isnan(idx_hom)
    idx_target_valid = idx_target[valid_mask].astype(np.int64)
    idx_hom_valid = idx_hom[valid_mask].astype(np.int64)
    cos_sim = np.full(len(df_pair_wise), np.nan, dtype=np.float32)
    batch_size = 500_000
    for i in range(0, len(idx_target_valid), batch_size):
        end = i + batch_size
        
        v1 = E[idx_target_valid[i:end]].astype(np.float32)
        v2 = E[idx_hom_valid[i:end]].astype(np.float32)
        
        cos_sim[np.where(valid_mask)[0][i:end]] = np.einsum("ij,ij->i", v1, v2)
    df_pair_wise["cos_prof"] = cos_sim


    #balaciamento:
    if mode != "test":
        top_n = 10+1
        has_idade = df_pair_wise['age_homonimo'].notna()
        has_end   = df_pair_wise['city_homonimo'].notna()
        # has_prof  = df_pair_wise['occ_homonimo'].notna()
        condicoes = [
            (df_pair_wise['label'] == 1),
            (has_idade & has_end),
            ((has_idade) | (has_end )),
        ]
        valores = [10, 2, 1]
        df_pair_wise['prioridade'] = np.select(condicoes, valores, default=0).astype('int8')
        df_pair_wise = df_pair_wise.sort_values(by=['id', 'prioridade'], ascending=[True, False])
        df_balanceado = df_pair_wise.groupby('id').head(top_n).copy()

        del df_pair_wise
        gc.collect()
        df_pair_wise = df_balanceado

    print(f"[{mode}] Tamanho dos pares pós balanciamento {len(df_pair_wise)}")
    contagem = df_pair_wise['label'].value_counts()
    print(f"[{mode}]  Registros Label 1: {contagem.get(1, 0)}")
    print(f"[{mode}] Registros Label 0: {contagem.get(0, 0)}")

    # outros dados:
    # df_pair_wise = df_pair_wise.drop(columns=["lista_homonimos_validos"], inplace=True)
    df_pair_wise["age_diff"] = df_pair_wise["age_homonimo"] - df_pair_wise["age_target"]
    if mode == "train":
        df_positivos = df_pair_wise[df_pair_wise['label'] == 1] # tira estatistica apenas dos casos verdadeiros
        for relacao, grupo in df_positivos.groupby('RELATED'):
            param = t.fit(grupo['age_diff'])
            parametros[relacao] = param
    
    mapa_df = {k: v[0] for k, v in parametros.items()}
    mapa_loc = {k: v[1] for k, v in parametros.items()}
    mapa_scale = {k: v[2] for k, v in parametros.items()}

    v_df = df_pair_wise["RELATED"].map(mapa_df).values
    v_loc = df_pair_wise["RELATED"].map(mapa_loc).values
    v_scale = df_pair_wise["RELATED"].map(mapa_scale).values

    pdf_values = t.pdf(df_pair_wise['age_diff'].values, df=v_df, loc=v_loc, scale=v_scale)
    max_pdf_values = t.pdf(v_loc, df=v_df, loc=v_loc, scale=v_scale)
    
    df_pair_wise["age_score"] = (pdf_values / max_pdf_values).astype(np.float32
    )

    del v_df, v_loc, v_scale, pdf_values, max_pdf_values
    del mapa_df, mapa_loc, mapa_scale
    gc.collect()

    #heuristica 
    #df_pair_wise["heuristica_score"] = (0.5 * df_pair_wise["age_score"]) + (0.4 * df_pair_wise["cos_end"]) + (0.1 * df_pair_wise["cos_prof"])

    #print_full(df_pair_wise, n=50)
    
    print("\n", flush=True)


    if mode == "train":
        return df_pair_wise, parametros
    else:
        return df_pair_wise, None

    
def verificar_invalidos(df, coluna, valores_alvo):
    mask_invalidos = df[coluna].astype(str).isin([str(v) for v in valores_alvo])
    mask_nulos = df[coluna].isna()
    
    mask_final = mask_invalidos | mask_nulos
    
    df_problemas = df[mask_final]
    encontrou = not df_problemas.empty
    
    return encontrou, df_problemas

def set_folds(df):

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled['fold'] = (df_shuffled.index % 5) + 1
    df_shuffled['id'] = range(len(df_shuffled))
    print_full(df_shuffled['RELATED'].value_counts().reset_index(name='qtd'))
    print(f"Tamanho dos pares: {len(df_shuffled)}")
    print(f"Soma sem balanciamento: {df_shuffled['qtd_homonimos_validos'].sum()}")
    print(f"Soma com balanciamento: {df_shuffled['qtd_homonimos_validos'].clip(upper=11).sum()}")
    return df_shuffled

def calculate_recall_at_k(df, k_values=[1, 5, 10], label_name="label", rank_name="rank", group_col_name="id"):
    #com label e rank
    total_positives = df[df[label_name] == 1][group_col_name].nunique()
    
    recalls = {}
    
    for k in k_values:
        found_at_k = df[(df[label_name] == 1) & (df[rank_name] <= k)][group_col_name].nunique()
        recall = found_at_k / total_positives
        recalls[f'Recall@{k}'] = recall
        
    return recalls

def cosine_similarity_numpy(col1, col2):
    a = np.array(col1)
    b = np.array(col2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def gaussian_score(x, mean, sigma):
    if pd.isna(x) or pd.isna(mean) or pd.isna(sigma) or sigma == 0:
        return 0.0
    
    exponent = -((x - mean)**2) / (2 * (sigma**2))
    return np.exp(exponent)

def get_county_name(row):
    state = int(row['STATEICP'])
    county = int(row['COUNTYICP'])
    return county_codes.get(state, {}).get(county, "NULL")

def process_pares(df):

    df = df[df['SAMPLE'] == 193002]
    df = descartar_colunas(df, ['YEAR', 'SAMPLE', 'HHWT', 'CLUSTER', 'STRATA' ,'PERWT', 'FAMSIZE', 'MOMLOC', 'POPLOC', 'BPL', 'BPLD', 'STREET'])
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df = df.rename(columns={'index': 'id'})

    print("*******************************")
    print(f"pessoas distintas iniciais: {len(df)}")

    df_pessoas_distintas = df[(
            #df['STREET'].notna() | 
            (~df['OCC'].isin([999, 995, 986, 984])) | 
            (~df['OCC1950'].isin([999, 995, 986, 984])) | 
            (df['CITY'] != 0) | 
            ((~df['OCCSTR'].astype(str).str.upper().isin(['N', 'NONE', 'NAN'])) & df['OCCSTR'].notna()) |
            df[['STATEICP', 'COUNTYICP', 'AGE']].notna().any(axis=1)
        )].copy()
    del df

    gc.collect()
    df_pessoas_distintas['STATESTR'] = df_pessoas_distintas['STATEICP'].map(mapa_estados)
    df_pessoas_distintas['CITY'] = df_pessoas_distintas['CITY'].replace(0, np.nan) #tratamento das cidades
    df_pessoas_distintas['CITYSTR'] = df_pessoas_distintas['CITY'].map(city_codes)
    df_pessoas_distintas['COUNTYSTR'] = df_pessoas_distintas.apply(get_county_name, axis=1)
    print(f"Pessoas distintas tratadas: {len(df_pessoas_distintas)}")

    print_full(df_pessoas_distintas, n=25)

    df_agg = df_pessoas_distintas.groupby(['NAMEFRST', 'NAMELAST']).agg(
        qtd_homonimos_validos=('id', 'count'),
        lista_ids=('id', list)
    ).reset_index()

    df_pessoas_distintas = pd.merge(df_pessoas_distintas, df_agg, on=['NAMEFRST', 'NAMELAST'])

    df_pessoas_distintas['lista_homonimos_validos'] = [
        [{'id': homonimo_id, 'is_person_2': int(homonimo_id == row_id)} for homonimo_id in lista]
        for row_id, lista in zip(df_pessoas_distintas['id'], df_pessoas_distintas['lista_ids'])
    ]
    df_pessoas_distintas.drop(columns=['lista_ids'], inplace=True)

    #print_full(df, n=20)
    #salvar_dataframe(df, "newyork_people_sample.csv")
    
    ### TODO - gerar pessoas distintas
    # df_pessoas_distintas = Gerar lista de pessoas distintas -> count 1 = len(df_pessoas_distintas)
    # df_pessoas_distintas = df_pessoas_distintas Filtrar só pessoas que tem (idade OU endereco OU profissao)
    # df_agg = df_pessoas_distintas agrupa por (first_name, last_name), agrega count(*) -> qtd_homonimos_validos, agregra lista (Id_unico) -> lista_homonimos_validos
    # df_pessoas_distintas = df_pessoas_distintas join df_agg (first_name, last_name)
    

    #filtra as relações desejadas
    gq_desejados = [1, 2]
    df_pessoas_relacoes = df_pessoas_distintas[df_pessoas_distintas['GQ'].isin(gq_desejados)].copy()

    chefes = df_pessoas_relacoes[df_pessoas_relacoes['RELATED'] == 101][['id', 'SERIAL']]
    membros = df_pessoas_relacoes[df_pessoas_relacoes['RELATED'].isin(codigos_familiares_desejados) & (df_pessoas_relacoes['RELATED'] != 101)][['id', 'RELATED', 'RELSTR', 'SERIAL']]
    #membros = df_pessoas_relacoes[(df_pessoas_relacoes['RELATED'] != 101)][['id', 'RELATED', 'RELSTR', 'SERIAL']]
    df_pares = pd.merge(
        membros,
        chefes, 
        on='SERIAL', 
        suffixes=('_parente', '_chefe')
    )

    df_pares = pd.merge(
        df_pares,
        df_pessoas_distintas.loc[
            df_pessoas_distintas['qtd_homonimos_validos'] > 1, 
            ['id', 'qtd_homonimos_validos', 'lista_homonimos_validos']
        ],
        left_on='id_parente',
        right_on='id',
        how="inner"
    )
    df_pares.drop(columns=['id'])

    print_full(df_pares, n=10)
    print_full(df_pares['RELATED'].value_counts().reset_index(name='qtd'))
    salvar_dataframe(df_pares, "IpumsUSA_pessoa_homonimo.parquet")

    homonimos_extra = df_pares['qtd_homonimos_validos'] - 1
    stats = homonimos_extra.agg(['min', 'max', 'mean', 'std'])
    print("estatísticas de homônimos: ")
    print(stats)
    somatorio = df_pares['qtd_homonimos_validos'].sum()
    print(f"somatório: {somatorio}")
    print(f"quantidade de pares: {len(df_pares)}")

    ### TODO - gerar os pares
    # df_pares = Id_unico_chefe, id_unico_parente, relacao_parentesco
    # remover quando relacao_parentesco nao é um paretesco
    # df_pares = df_pares join(df_pessoas_distintas pela coluna:[Id_unico_chefe], colunas adicionar: []) join(df_pessoas_distintas pela coluna:id_unico_parente] and qtd_homonimos_validos > 1, colunas adicionar: [qtd_homonimos_validos, lista_homonimos_validos])
    # df_pares [Id_unico_chefe, id_unico_parente, relacao_parentesco, qtd_homonimos_validos, lista_homonimos_validos]
    # df_pares.to_csv("{NOME_BASE}_pessoa_homonimo.csv")
    # (qtd_homonimos_validos - 1) -> min, max, mean, std
    # len(df_pares)

    ids_usados = df_pares['id_chefe'].to_list() + df_pares['id_parente'].to_list()
    ids_homonimos = [item['id'] for sublist in df_pares['lista_homonimos_validos'] for item in sublist]
    set_pessoas_distintas_uso = set(ids_usados + ids_homonimos)
    df_pessoas_usadas = df_pessoas_distintas[df_pessoas_distintas['id'].isin(set_pessoas_distintas_uso)]
    
    
    ### TODO - filtrar pessoas distintas, apenas pessoas em uso dentro de algum relacionamento familiar
    # set_pessoas_distintas_uso = set(df_pares.Id_unico_chefe.to_list() | df_pares.id_unico_parente.to_list() |  df_pares.lista_homonimos_validos.to_list() # lista de lista -> converte para lista)
    # df_pessoas_distintas_uso = df_pessoas_distintas.Id_unico.isin(set_pessoas_distintas_uso)
    # df_pessoas_distintas_uso.to_csv("{NOME_BASE}_distinct_pessoa_dados.csv")
    # # Id_unico (id_familia + "_" +id_pessoa_dentro_familia), nome (first, last), idade, id_unico_endereco(cod_estado"-"cod_condado"-"nome_cidade"-"nome_rua) se um campo for nulo ("NULL"), endereco (cod_estado, nome_estado, cod_condado, nome_condado, nome_cidade, nome_rua), profissao (cod_profissao, nome_profissao), qtd_homonimos
    # len(df_pessoas_distintas_uso)
    
    cols_endereco = ['STATEICP', 'COUNTYICP', 'CITY']
    df_pessoas_usadas['id_endereco'] = (
        df_pessoas_usadas[cols_endereco].astype(str)
        .replace(['nan', 'None', 'NaN', '0.0', '0', '<NA>'], 'NULL')
        .fillna('NULL')
        .agg('_'.join, axis=1)
    )
    df_enderecos = df_pessoas_usadas.groupby('id_endereco').agg(
        STATEICP=('STATEICP', 'first'),
        STATESTR=('STATESTR', 'first'),
        COUNTYICP=('COUNTYICP', 'first'),
        COUNTYSTR=('COUNTYSTR', 'first'),
        CITY=('CITY', 'first'),
        CITYSTR=('CITYSTR', 'first'),
        qtd_pessoas=('PERNUM', 'count'),
        lista_pernum=('PERNUM', list),
        lista_nomes=('NAMEFRST', list)
    ).reset_index()

    cols_para_limpar = ['STATESTR', 'COUNTYSTR', 'CITYSTR'] # limpando df
    for col in cols_para_limpar:
        df_enderecos[col] = (
            df_enderecos[col]
            .astype(str) 
            .replace(['nan', 'None', 'NaN', '0.0', '0', '0.0'], 'NULL')
            .fillna('NULL')
        )
    print(f'Quantidade de endereços: {len(df_enderecos)}')
    print_full(df_enderecos, n=10)

    df_pessoas_usadas['id_profissao'] = (
        df_pessoas_usadas['OCC'].astype(str) + '_' +
        df_pessoas_usadas['OCCSTR'].astype(str).replace(['N', 'NONE', 'NAN', '<NA>'], 'NULL').str.upper().fillna('NULL')
    )
    df_profissoes = (df_pessoas_usadas.groupby('id_profissao').agg(
        OCC=('OCC', 'first'),
        OCCSTR=('OCCSTR', 'first'),
        qtd_na_profissao=('PERNUM', 'count'),
        lista_ids=('id', list)
    ).reset_index())
    print(f'Quantidade de profiss~oes: {len(df_profissoes)}')
    print_full(df_profissoes, n=10)
    salvar_dataframe(df_profissoes, f"IpumsUSA_profissoes.csv")
    salvar_dataframe(df_enderecos, f"IpumsUSA_enderecos.csv")

    salvar_dataframe(df_pessoas_usadas, "IpumsUSA_distinct_pessoa_dados.csv")
    print(f"Pessoas diferentes utilizadas: {len(df_pessoas_usadas)}")
    print_full(df_pessoas_usadas, n=10)
    

def descartar_colunas(df, lista_colunas):
    colunas_para_remover = [c for c in lista_colunas if c in df.columns]
    df.drop(columns=colunas_para_remover, inplace=True)
        
    return df