import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    try:
        df = pd.read_csv("titanic.csv")

        print(df.info())
        print(df.isnull().sum())

        df['Age'].fillna(df['Age'].mean(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        if 'Cabin' in df.columns:
            df.drop(columns=['Cabin'], inplace=True)

        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df = pd.get_dummies(df, columns=['Embarked'])

        scaler = StandardScaler()
        df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

        plt.figure(figsize=(8,5))
        sns.boxplot(data=df[['Age', 'Fare']])
        plt.title("Before Removing Outliers")
        plt.show()

        Q1 = df[['Age','Fare']].quantile(0.25)
        Q3 = df[['Age','Fare']].quantile(0.75)
        IQR = Q3 - Q1

        df = df[~((df[['Age','Fare']] < (Q1 - 1.5 * IQR)) | 
                  (df[['Age','Fare']] > (Q3 + 1.5 * IQR))).any(axis=1)]

        plt.figure(figsize=(8,5))
        sns.boxplot(data=df[['Age', 'Fare']])
        plt.title("After Removing Outliers")
        plt.show()

        df.to_csv("cleaned_titanic.csv", index=False)

        messagebox.showinfo("Success", "Done")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Data Preprocessing App")
root.geometry("450x300")
root.configure(bg="#f0f0f0")

title = tk.Label(root, text="Task 1: Data Cleaning", font=("Arial", 16, "bold"), bg="#f0f0f0")
title.pack(pady=20)

desc = tk.Label(root, text="Click below to run preprocessing", font=("Arial", 11), bg="#f0f0f0")
desc.pack(pady=10)

run_btn = tk.Button(root, text="Run Preprocessing", command=run_preprocessing,
                    bg="green", fg="white", padx=15, pady=8)
run_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.quit,
                     bg="red", fg="white", padx=15, pady=8)
exit_btn.pack(pady=10)

footer = tk.Label(root, text="AI/ML Internship Project", font=("Arial", 9), bg="#f0f0f0")
footer.pack(side="bottom", pady=10)

root.mainloop()