import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("diabetic_data.csv", na_values="?", low_memory=False)

#remover colunas vazias e pacientes que já faleceram
cols_to_drop = ["weight", "payer_code", "medical_specialty"]
df.drop(columns=cols_to_drop, inplace=True)

#usando IDs do arquivo mapping
discharge_ids_to_remove = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(discharge_ids_to_remove)]

df["readmitted_binary"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

#Idade e diagnosticos
age_dict = {"[0-10]":5, "[10-20]": 15, "[20-30]": 25, "[30-40]": 35,
            "[40-50]": 45, "[50-60]": 55, "[60-70]": 65, "[70-80]": 75,
            "[80-90]": 85, "[90-100]": 95}
df["age_midpoint"] = df["age"].map(age_dict)

#função icd-9

def map_icd9(code):
    try:
        if str(code).startswith(("V", "E")): return "Other"
        val = float(code)
        if (val >= 390 and val <= 459) or val == 785: return "Circulatory"
        elif (val >= 460 and val <= 519) or val == 786: return "Respiratory"
        elif (val >= 520 and val <= 579) or val == 787: return "Digestive"
        elif val >= 250 and val < 251: return "Diabetes"
        elif (val >= 800 and val <= 999): return "Injury"
        elif (val >= 710 and val <= 739): return "Musculoskeletal"
        elif (val >= 580 and val <= 629) or val == 788: return "Genitourinary"
        elif (val >= 140 and val <= 239): return "Neoplasms"
        else: return "Other"
    except: return "Other"

for col in ["diag_1", "diag_2", "diag_3"]:
    df[col + "_group"] = df[col].apply(map_icd9)

medications = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
               "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
               "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
               "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
               "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

df["num_meds"] = (df[medications] != "No").sum(axis=1)
df["change_numeric"] = df["change"].apply(lambda x: 1 if x == "Ch" else 0)

#visualização
plt.figure(figsize=(12,6))
sns.barplot(x="diag_1_group", y="readmitted_binary", data=df)
plt.title("Taxa de Readmissão por Grupo de Doença")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x="age", y="readmitted_binary", data=df.sort_values("age_midpoint"), marker="o")
plt.title("Tendência de Readmissão por Faixa Etária")
plt.ylabel("Taxa de Readmissão (Média)")
plt.xlabel("Faixa Etária")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

#manter colunas categóricas
categorical_cols = ["race", "gender", "diag_1_group", "diag_2_group", "diag_3_group", "max_glu_serum", "A1Cresult"]

#criar copia para não estragar original
df_ml = df.copy()

#preenchendo valores nulos
for col in categorical_cols:
    df_ml[col] = df_ml[col].fillna("Missing")
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

df_ml["change"] = LabelEncoder().fit_transform(df_ml["change"])
df_ml["diabetesMed"] = LabelEncoder().fit_transform(df_ml["diabetesMed"])

print("Colunas transformadas em números")
print(df_ml[categorical_cols].head())

#usar colunas já transformadas em números
features = ["age_midpoint", "num_meds", "change_numeric", "num_lab_procedures",
            "num_procedures", "number_diagnoses", "number_emergency",
            "number_inpatient", "number_outpatient", "diag_1_group",
            "gender", "race", "diabetesMed"]

X = df_ml[features]
y = df_ml["readmitted_binary"]

#treino 80% e teste 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#random forest
print("Treinando o modelo")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#avaliando modelo
y_pred = model.predict(X_test)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

#gráfico de importancia das variaveis
importances = model.feature_importances_
feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index, palette="magma")
plt.title("Quais fatores mais influenciam a Readmissão?")
plt.xlabel("Pontuação de Importância")
plt.ylabel("Variáveis")
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não Readmitido", "Readmitido"])

plt.figure(figsize=(8,6))
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão: Acertos X Erros")
plt.show()
