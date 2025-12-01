"""
Train trust-based models on Amazon product data using:
- GradientBoostingClassifier
- RandomForestClassifier

Target: high_purchase (whether purchased_last_month is >= median)
Features: trust-related signals (rating, reviews, badges, sponsorship, coupons, etc.)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

# ----------------------------------------------------------
# 1. Load cleaned dataset
# ----------------------------------------------------------
CSV_PATH = "amazon_products_sales_data_cleaned.csv"  # change if needed

df = pd.read_csv(CSV_PATH)

# Keep only rows where we know purchased_last_month
df = df[df["purchased_last_month"].notna()].copy()

# ----------------------------------------------------------
# 2. Create target variable (what we’re predicting)
#    high_purchase = 1 if purchased_last_month >= median, else 0
# ----------------------------------------------------------
median_purchases = df["purchased_last_month"].median()
df["high_purchase"] = (df["purchased_last_month"] >= median_purchases).astype(int)

# ----------------------------------------------------------
# 3. Select TRUST-RELATED features only
#    (signals a customer might use to decide if they trust a product)
# ----------------------------------------------------------
trust_features = [
    "product_rating",        # star rating
    "total_reviews",         # number of reviews
    "discounted_price",      # visible price
    "original_price",        # reference price (for discount perception)
    "discount_percentage",   # how “good” the deal looks

    "is_best_seller",        # badge / social proof
    "is_sponsored",          # sponsored vs organic
    "has_coupon",            # coupon presence
    "buy_box_availability",  # can you actually buy it right now?
    "sustainability_tags",   # sustainability label (trust signal)
    "product_category",      # category context (different trust norms)
]

X = df[trust_features]
y = df["high_purchase"]

# Split into numeric vs categorical columns
numeric_features = [
    "product_rating",
    "total_reviews",
    "discounted_price",
    "original_price",
    "discount_percentage",
]

categorical_features = list(set(trust_features) - set(numeric_features))

# ----------------------------------------------------------
# 4. Preprocessing pipelines
#    - Impute missing values
#    - Scale numeric features
#    - One-hot encode categoricals
# ----------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ----------------------------------------------------------
# 5. Train / test split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # keep class balance similar in train & test
)

# ----------------------------------------------------------
# 6. Define the two models
# ----------------------------------------------------------

# Model 1: Gradient Boosting Classifier
gbc_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )),
    ]
)

# Model 2: Random Forest Classifier
rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )),
    ]
)

# ----------------------------------------------------------
# 7. Train both models
# ----------------------------------------------------------
print("Training Gradient Boosting Classifier...")
gbc_pipeline.fit(X_train, y_train)

print("Training Random Forest Classifier...")
rf_pipeline.fit(X_train, y_train)

# ----------------------------------------------------------
# 8. Evaluate and compare
# ----------------------------------------------------------
gbc_pred = gbc_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

gbc_acc = accuracy_score(y_test, gbc_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n================ MODEL ACCURACIES ================")
print(f"Gradient Boosting Classifier accuracy: {gbc_acc:.4f}")
print(f"Random Forest Classifier accuracy:    {rf_acc:.4f}")

print("\n=========== GBC Classification Report ===========")
print(classification_report(y_test, gbc_pred))

print("\n=========== RF Classification Report ============")
print(classification_report(y_test, rf_pred))

# ----------------------------------------------------------
# 9. Pick the best model and save it
# ----------------------------------------------------------
if rf_acc >= gbc_acc:
    best_name = "Random Forest Classifier"
    best_pipeline = rf_pipeline
else:
    best_name = "Gradient Boosting Classifier"
    best_pipeline = gbc_pipeline

print(f"\nBest model based on test accuracy: {best_name}")

# Save best model to disk
MODEL_PATH = "best_trust_model.joblib"
joblib.dump(best_pipeline, MODEL_PATH)
print(f"Saved best model to: {MODEL_PATH}")

# ----------------------------------------------------------
# 10. Generate Presentation Deliverables
# ----------------------------------------------------------
print("\nGenerating presentation deliverables...")

# --- A. Feature Importance Plot ---
# Note: This works best for Tree-based models.
# We need to extract feature names from the preprocessor.
try:
    # Get feature names from one-hot encoder
    ohe_feature_names = best_pipeline.named_steps["preprocess"].named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(ohe_feature_names)
    
    # Get importances
    importances = best_pipeline.named_steps["model"].feature_importances_
    
    # Create DataFrame for plotting
    feat_imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    feat_imp_df = feat_imp_df.sort_values(by="importance", ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top 10 Feature Importances ({best_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Saved feature_importance.png")
    plt.close()
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")

# --- B. Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, best_pipeline.predict(X_test))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Confusion Matrix ({best_name})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
plt.close()

# --- C. ROC Curve & AUC Score ---
y_prob = best_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({best_name})")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
print("Saved roc_curve.png")
plt.close()

# --- D. Box Plot: Rating vs. Trust ---
# Shows if high-trust products actually have higher ratings.
plt.figure(figsize=(8, 6))
sns.boxplot(x=y_test, y=X_test["product_rating"], palette="Set2")
plt.xticks([0, 1], ["Low Purchase (Low Trust)", "High Purchase (High Trust)"])
plt.xlabel("Trust Category")
plt.ylabel("Product Rating (Stars)")
plt.title("Do Trusted Products Have Higher Ratings?")
plt.tight_layout()
plt.savefig("box_plot_rating.png")
print("Saved box_plot_rating.png")
plt.close()

# --- E. Scatter Plot: Price vs. Reviews ---
# Shows the relationship between price, popularity, and trust.
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=X_test, 
    x="discounted_price", 
    y="total_reviews", 
    hue=y_test, 
    alpha=0.6, 
    palette={0: "red", 1: "green"}
)
plt.title("Price vs. Reviews (Green = High Trust)")
plt.xlabel("Price ($)")
plt.ylabel("Total Reviews")
plt.legend(title="High Purchase", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("scatter_price_reviews.png")
print("Saved scatter_price_reviews.png")
plt.close()

# --- F. Summary Statistics Table ---
print("\n=== KEY STATISTICS (High vs. Low Trust) ===")
X_test_with_target = X_test.copy()
X_test_with_target["high_purchase"] = y_test
summary = X_test_with_target.groupby("high_purchase")[["product_rating", "total_reviews", "discounted_price", "discount_percentage"]].mean()
print(summary)

# --- G. Trust Score Calculator (Demo) ---
def predict_trust(product_details):
    """
    Predicts the probability of a product being 'High Trust' (high purchase volume).
    """
    # Convert dict to DataFrame
    input_df = pd.DataFrame([product_details])
    
    # Ensure all columns expected by the pipeline exist (fill others with defaults/NaN)
    for col in trust_features:
        if col not in input_df.columns:
            input_df[col] = np.nan # Imputer will handle it
            
    # Predict probability
    prob = best_pipeline.predict_proba(input_df)[0, 1]
    return prob

print("\n=== TRUST SCORE CALCULATOR DEMO ===")
example_product = {
    "product_rating": 4.5,
    "total_reviews": 500,
    "discounted_price": 25.0,
    "original_price": 30.0,
    "discount_percentage": 17.0,
    "is_best_seller": "Best Seller",
    "is_sponsored": "Organic",
    "has_coupon": "No Coupon",
    "buy_box_availability": "Add to cart",
    "sustainability_tags": "No tags",
    "product_category": "Electronics"
}

score = predict_trust(example_product)
print(f"Product: {example_product}")
print(f"Trust Score (Probability of High Purchase): {score:.4f}")
