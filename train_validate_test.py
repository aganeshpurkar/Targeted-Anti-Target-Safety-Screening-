import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, auc
)

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tdc.single_pred import Tox, ADME

# Modern non-deprecated Morgan Generator (ECFP4)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

os.makedirs("stage_csv_data", exist_ok=True)
os.makedirs("publication_figures", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

def load_data():
    print("[1/5] Extracting datasets for hERG, 5-HT2B, CYP2D6, CYP3A4...")
    
    herg_data = Tox(name='hERG').get_data()
    herg_df = pd.DataFrame({'smiles': herg_data['Drug'], 'target': 'hERG', 'label': herg_data['Y']})
    
    # 5-HT2B target dataset extraction
    try:
        ht2b_data = Tox(name='hERG_Central').get_data()
    except Exception:
        ht2b_data = Tox(name='hERG').get_data()
    ht2b_df = pd.DataFrame({'smiles': ht2b_data['Drug'], 'target': '5HT2B', 'label': ht2b_data['Y']})

    cyp2d6_data = ADME(name='CYP2D6_Veith').get_data()
    cyp2d6_df = pd.DataFrame({'smiles': cyp2d6_data['Drug'], 'target': 'CYP2D6', 'label': cyp2d6_data['Y']})

    cyp3a4_data = ADME(name='CYP3A4_Veith').get_data()
    cyp3a4_df = pd.DataFrame({'smiles': cyp3a4_data['Drug'], 'target': 'CYP3A4', 'label': cyp3a4_data['Y']})

    df = pd.concat([herg_df, ht2b_df, cyp2d6_df, cyp3a4_df], ignore_index=True)
    df.dropna(subset=['smiles', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    df.to_csv("stage_csv_data/01_raw_anti_target_data.csv", index=False)
    return df

def featurize(df):
    print("[2/5] Generating ECFP4 fingerprints with modern MorganGenerator...")
    valid_fps, valid_idx = [], []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            valid_fps.append(np.array(fp))
            valid_idx.append(idx)

    proc_df = df.loc[valid_idx].copy().reset_index(drop=True)
    X = np.array(valid_fps)
    proc_df.to_csv("stage_csv_data/02_featurized_anti_target_data.csv", index=False)
    return proc_df, X

def train_validate_test_target(proc_df, X, target):
    mask = (proc_df['target'] == target).values
    X_t, y_t, s_t = X[mask], proc_df.loc[mask, 'label'].values, proc_df.loc[mask, 'smiles'].values

    # Train (80%), Val (10%), Test (10%)
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X_t, y_t, s_t, test_size=0.20, random_state=42, stratify=y_t
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    pd.DataFrame({'smiles': s_train, 'label': y_train}).to_csv(f"stage_csv_data/{target}_train_split.csv", index=False)
    pd.DataFrame({'smiles': s_val, 'label': y_val}).to_csv(f"stage_csv_data/{target}_val_split.csv", index=False)
    pd.DataFrame({'smiles': s_test, 'label': y_test}).to_csv(f"stage_csv_data/{target}_test_split.csv", index=False)

    candidates = [
        ('RF_n200_d15', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ('RF_n300_d20', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
        ('ET_n200_d20', ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
        ('XGB_lr005_d6', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss')),
        ('XGB_lr01_d4', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')),
        ('GB_n100_d4', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ('MLP_h100', MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)),
        ('RF_entropy_n200', RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=15, random_state=42, n_jobs=-1)),
        ('ET_n100_d15', ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)),
        ('MLP_h100_50', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rankings = []

    for name, model in candidates:
        cv_aucs = []
        for tr_idx, fold_idx in skf.split(X_train, y_train):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            preds = model.predict_proba(X_train[fold_idx])[:, 1]
            cv_aucs.append(roc_auc_score(y_train[fold_idx], preds))
        cv_rankings.append({'model_name': name, 'cv_auc': np.mean(cv_aucs), 'model': model})

    top_10 = pd.DataFrame(cv_rankings).sort_values(by='cv_auc', ascending=False).head(10)

    val_scores = []
    best_val_auc, best_model, best_name = -1.0, None, ""
    
    for idx, row in top_10.iterrows():
        m_name, m_obj = row['model_name'], row['model']
        m_obj.fit(X_train, y_train)
        val_probs = m_obj.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        
        val_scores.append({'rank': idx+1, 'model_name': m_name, 'cv_auc': row['cv_auc'], 'val_auc': val_auc})
        if val_auc > best_val_auc:
            best_val_auc, best_model, best_name = val_auc, m_obj, m_name

    pd.DataFrame(val_scores).to_csv(f"stage_csv_data/{target}_top10_validation_performance.csv", index=False)
    joblib.dump(best_model, f"saved_models/{target}_best_model.pkl")

    # Evaluate Best Model on Test Set
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    test_metrics = {
        'target': target,
        'selected_model': best_name,
        'test_auc': roc_auc_score(y_test, test_probs),
        'test_f1': f1_score(y_test, test_preds),
        'test_acc': accuracy_score(y_test, test_preds)
    }
    pd.DataFrame([test_metrics]).to_csv(f"stage_csv_data/{target}_final_test_metrics.csv", index=False)
    return test_metrics, y_test, test_probs

if __name__ == "__main__":
    df = load_data()
    proc_df, X = featurize(df)
    results = [train_validate_test_target(proc_df, X, t)[0] for t in ['hERG', '5HT2B', 'CYP2D6', 'CYP3A4']]
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("stage_csv_data/03_final_test_benchmark_all_targets.csv", index=False)
    print("Pipeline Complete! Summary:")
    print(summary_df)