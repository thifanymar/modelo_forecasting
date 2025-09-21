# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import itertools
import joblib
# ===============================
# FUNÇÕES AUXILIARES
# ===============================

def carregar_dados(caminho_transacoes, caminho_pdv, caminho_produtos):
    """Carrega os datasets parquet e retorna como DataFrames."""
    df_transacoes = pd.read_parquet(caminho_transacoes)
    df_pdv = pd.read_parquet(caminho_pdv)
    df_produtos = pd.read_parquet(caminho_produtos)
    return df_transacoes, df_pdv, df_produtos


def preparar_dados(df_transacoes, df_pdv, df_produtos):
    """Realiza os merges, tratamento de colunas e encoding das variáveis."""
    
    # Joins
    df = pd.merge(df_transacoes, df_produtos, how="inner",
                  left_on="internal_product_id", right_on="produto")
    df = pd.merge(df, df_pdv, how="inner",
                  left_on="internal_store_id", right_on="pdv")

    # Limpeza
    df = df.drop(columns=["produto", "pdv", "descricao"])
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    # Mantém apenas quantidades positivas
    df = df[df['quantity'] >= 0]

    # Definição de colunas
    categorical_cols = ['internal_store_id', 'internal_product_id']
    
    # Remove numéricas originais
    df = df.drop(columns=['gross_value', 'net_value', 'gross_profit', 'discount', 'taxes'])

    # Label Encoding para variáveis categóricas
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Features temporais
    df['Semana'] = df['transaction_date'].dt.isocalendar().week
    df['month'] = df['transaction_date'].dt.month
    df['quarter'] = df['transaction_date'].dt.quarter

    return df, categorical_cols


def criar_agregacoes(df, categorical_cols):
    """Divide em treino e teste, agrega semanalmente e cria lags."""
    # Split treino/teste
    train_df = df[df['reference_date'] < '2022-12-01']
    test_df = df[df['reference_date'] == '2022-12-01']

    # Agregação semanal
    agg_cols = categorical_cols + ['Semana', 'month', 'quarter']
    train_weekly = train_df.groupby(agg_cols)['quantity'].sum().reset_index()
    test_weekly = test_df.groupby(agg_cols)['quantity'].sum().reset_index()

    # Criar lags no treino
    train_weekly = train_weekly.sort_values(['internal_store_id', 'internal_product_id', 'Semana'])
    train_weekly['lag_1'] = train_weekly.groupby(['internal_store_id', 'internal_product_id'])['quantity'].shift(1).fillna(0)
    train_weekly['lag_2'] = train_weekly.groupby(['internal_store_id', 'internal_product_id'])['quantity'].shift(2).fillna(0)

    # Criar lags no teste
    test_weekly = test_weekly.sort_values(['internal_store_id', 'internal_product_id', 'Semana'])
    last_week_train = train_weekly.groupby(['internal_store_id', 'internal_product_id'])['quantity'].last().reset_index()
    test_weekly = pd.merge(test_weekly, last_week_train.rename(columns={'quantity': 'lag_1'}),
                           on=['internal_store_id', 'internal_product_id'], how='left')
    test_weekly['lag_2'] = 0
    test_weekly[['lag_1', 'lag_2']] = test_weekly[['lag_1', 'lag_2']].fillna(0)

    return train_weekly, test_weekly


def treinar_modelo(X_train, y_train, X_test, y_test):
    """Treina modelo XGBoost e retorna modelo treinado e previsões."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    y_pred = model.predict(X_test)
    return model, y_pred


def avaliar_modelo(y_test, y_pred, X_test):
    """Avalia o modelo e salva relatórios em CSV."""
    mae = mean_absolute_error(y_test, y_pred)
    media_vendas = y_test.mean()
    
    print(f"MAE: {mae:.2f}")
    print(f"MAE / Média de vendas: {mae / media_vendas:.2f}")
    
    # Erro total
    total_previsto, total_real = y_pred.sum(), y_test.sum()
    erro_total = total_previsto - total_real
    print("Total Previsto:", total_previsto)
    print("Total Real:", total_real)
    print("Erro Total:", erro_total)

    # DataFrame de avaliação
    eval_df = X_test.copy()
    eval_df['y_test'] = y_test
    eval_df['y_pred'] = y_pred.round().astype(int)
    eval_df['abs_error'] = abs(eval_df['y_test'] - eval_df['y_pred'])
    eval_df['percent_error'] = 100 * eval_df['abs_error'] / eval_df['y_test'].replace(0, np.nan)

    eval_df[['Semana', 'internal_store_id', 'internal_product_id',
             'y_test', 'y_pred', 'abs_error', 'percent_error']] \
        .to_csv("previsao_vs_real.csv", sep=';', index=False, encoding='utf-8')

    # MAE por produto
    mae_produto = eval_df.groupby('internal_product_id')['abs_error'].mean().sort_values(ascending=False)
    mae_produto.to_csv("mae_por_produto.csv", sep=';', index=True, encoding='utf-8')

    return eval_df




# ===============================
# PIPELINE PRINCIPAL
# ===============================
    # --- Carregar dados ---
df_transacoes, df_pdv, df_produtos = carregar_dados(
    r"C:\Users\thifa\Downloads\hackathon_2025_templates\hackathon_2025_templates\part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet",
    r"C:\Users\thifa\Downloads\hackathon_2025_templates\hackathon_2025_templates\part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet",
    r"C:\Users\thifa\Downloads\hackathon_2025_templates\hackathon_2025_templates\part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet"
)

# --- Preparar dados ---
df, categorical_cols = preparar_dados(df_transacoes, df_pdv, df_produtos)

# --- Criar agregações ---
train_weekly, test_weekly = criar_agregacoes(df, categorical_cols)

# --- Separar features e target ---
features = categorical_cols + ['Semana', 'month', 'quarter', 'lag_1', 'lag_2']
target = 'quantity'

X_train, y_train = train_weekly[features], train_weekly[target]
X_test, y_test = test_weekly[features], test_weekly[target]

# --- Treinar modelo ---
model, y_pred = treinar_modelo(X_train, y_train, X_test, y_test)
joblib.dump(model, "modelo_xgb_vendas_2022.pkl")


# --- Avaliar ---
eval_df = avaliar_modelo(y_test, y_pred, X_test)

# --- Gráficos ---
# plotar_resultados(y_test, y_pred)

print("Pipeline finalizado com sucesso 🚀")


#---------------
# prevendo



# --- 1️⃣ Salvar o modelo treinado ---
# joblib.dump(model, "modelo_xgb_vendas_2022.pkl")

# --- 1️⃣ Criar grid de previsão apenas para PDVs/produtos com histórico ---
previsao_grid = train_weekly[['internal_store_id', 'internal_product_id']].drop_duplicates()

# Adicionar as semanas de janeiro
semanas = [1, 2, 3, 4, 5]
previsao_grid = pd.concat([previsao_grid.assign(Semana=s) for s in semanas], ignore_index=True)

# Adicionar colunas temporais
previsao_grid['month'] = 1
previsao_grid['quarter'] = 1

# --- 2️⃣ Criar lags usando última semana de dezembro ---
ultimas_vendas_dict = train_weekly.groupby(['internal_store_id', 'internal_product_id'])['quantity'].last().to_dict()
previsao_grid['lag_1'] = previsao_grid.set_index(['internal_store_id','internal_product_id']).index.map(ultimas_vendas_dict)
previsao_grid['lag_1'] = previsao_grid['lag_1'].fillna(0)
previsao_grid['lag_2'] = 0

# --- 3️⃣ Previsão --- 
# Usar apenas features disponíveis no grid
features_grid = ['internal_store_id', 'internal_product_id', 'Semana', 'month', 'quarter', 'lag_1', 'lag_2']
previsao_grid['quantidade'] = model.predict(previsao_grid[features_grid]).round().astype(int)

# --- 4️⃣ Ajustar colunas e salvar CSV ---
# --- 4️⃣ Ajustar colunas e salvar CSV ---
previsao_final = previsao_grid[['Semana', 'internal_store_id', 'internal_product_id', 'quantidade']]

# Renomear as colunas exatamente como pedido
previsao_final.rename(columns={
    'Semana': 'semana',
    'internal_store_id':'pdv',
    'internal_product_id':'produto'
}, inplace=True)

# Salvar CSV no formato correto
previsao_final.to_csv("previsao_jan_2023.csv", sep=';', index=False, encoding='utf-8')

print("CSV de previsão para janeiro/2023 gerado com sucesso ✅")



    
