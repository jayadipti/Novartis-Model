import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "/kaggle/input/novartis/usecase_4_.csv"  
data = pd.read_csv(file_path)


print("Dataset Columns:", data.columns)
print("Dataset Preview:")
print(data.head())


target_column = "Study Recruitment Rate"  

if target_column not in data.columns:
    print(f"Column '{target_column}' not found. Please verify the column name.")
else:
    data = data.dropna()  

    
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    X = data.drop(columns=[target_column])
    y = data[target_column]

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
   
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

   
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Features Influencing Recruitment Rate:")
    print(importance_df.head(10))


    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(10))
    plt.title("Top 10 Feature Importances")
    plt.show()

output_file_path = "/kaggle/working/usecase_4_predictions.csv"  
output = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
output.to_csv(output_file_path, index=False)
print(f"\nPredictions saved to '{output_file_path}'.")
    
