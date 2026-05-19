import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    accuracy_score
)

os.makedirs('reports/figures', exist_ok=True)

X_train_tfidf, X_test_tfidf, y_train, y_test = joblib.load('models/data_split.pkl')

models = {
    'Naive Bayes':          joblib.load('models/naive_bayes.pkl'),
    'Logistic Regression':  joblib.load('models/logistic_regression.pkl'),
    'LinearSVC':            joblib.load('models/linear_svc.pkl')
}

results = {}

for name, model in models.items():
    y_pred_default = model.predict(X_test_tfidf)

    y_probs = model.predict_proba(X_test_tfidf)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx       = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    y_pred_tuned = (y_probs >= best_threshold).astype(int)

    results[name] = {
        'y_pred_default': y_pred_default,
        'y_pred_tuned':   y_pred_tuned,
        'y_probs':        y_probs,
        'best_threshold': best_threshold,
        'precision':      precision,
        'recall':         recall,
        'thresholds':     thresholds,
        'f1_scores':      f1_scores,
        'acc_default':    accuracy_score(y_test, y_pred_default),
        'acc_tuned':      accuracy_score(y_test, y_pred_tuned),
        'f1_default':     f1_score(y_test, y_pred_default, average='weighted'),
        'f1_tuned':       f1_score(y_test, y_pred_tuned,   average='weighted'),
        'spam_f1_default': f1_score(y_test, y_pred_default, pos_label=1),
        'spam_f1_tuned':   f1_score(y_test, y_pred_tuned,   pos_label=1),
    }

LINE = "=" * 65

print(LINE)
print("   MODEL COMPARISON — Default Threshold (0.5)")
print(LINE)
print(f"{'Model':<25} {'Accuracy':>10} {'Weighted F1':>12} {'Spam F1':>10}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<25} {r['acc_default']:>10.4f} {r['f1_default']:>12.4f} {r['spam_f1_default']:>10.4f}")

print()
print(LINE)
print("   MODEL COMPARISON — Tuned Threshold")
print(LINE)
print(f"{'Model':<25} {'Best Threshold':>15} {'Accuracy':>10} {'Spam F1':>10}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<25} {r['best_threshold']:>15.4f} {r['acc_tuned']:>10.4f} {r['spam_f1_tuned']:>10.4f}")

print()
print(LINE)
print("   DETAILED CLASSIFICATION REPORTS — Tuned Threshold")
print(LINE)
for name, r in results.items():
    print(f"\n--- {name} (threshold = {r['best_threshold']:.4f}) ---")
    print(classification_report(y_test, r['y_pred_tuned'], target_names=['Ham', 'Spam']))

best_model_name = max(results, key=lambda n: results[n]['spam_f1_tuned'])
best_model_data = results[best_model_name]

print(LINE)
print(f"     BEST MODEL: {best_model_name}")
print(f"     Best Threshold : {best_model_data['best_threshold']:.4f}")
print(f"     Spam F1 (tuned): {best_model_data['spam_f1_tuned']:.4f}")
print(f"     Accuracy (tuned): {best_model_data['acc_tuned']:.4f}")
print(LINE)

COLORS = {
    'Naive Bayes':         '#4C72B0',
    'Logistic Regression': '#DD8452',
    'LinearSVC':           '#55A868'
}

plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
})

fig, ax = plt.subplots(figsize=(10, 5))

names      = list(results.keys())
x          = np.arange(len(names))
width      = 0.35
f1_default = [results[n]['spam_f1_default'] for n in names]
f1_tuned   = [results[n]['spam_f1_tuned']   for n in names]

bars1 = ax.bar(x - width/2, f1_default, width, label='Default (0.5)',
               color=[COLORS[n] for n in names], alpha=0.5)
bars2 = ax.bar(x + width/2, f1_tuned,   width, label='Tuned Threshold',
               color=[COLORS[n] for n in names], alpha=0.95)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('Spam F1 Score')
ax.set_title('Spam F1 Score — Default vs Tuned Threshold', fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0.94, 1.01)
plt.tight_layout()
plt.savefig('reports/figures/f1_comparison.png', dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices — Tuned Threshold', fontsize=13, fontweight='bold')

for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['y_pred_tuned'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar=False)
    ax.set_title(f'{name}\n(threshold={r["best_threshold"]:.2f})', fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('reports/figures/confusion_matrices.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))

for name, r in results.items():
    ax.plot(r['recall'], r['precision'],
            label=name, color=COLORS[name], linewidth=2)

    best_idx = np.argmax(r['f1_scores'][:-1])
    ax.scatter(r['recall'][best_idx], r['precision'][best_idx],
               color=COLORS[name], s=100, zorder=5)

ax.set_xlabel('Recall  (spam caught)')
ax.set_ylabel('Precision  (accuracy of spam flags)')
ax.set_title('Precision-Recall Curves — All Models', fontsize=13, fontweight='bold')
ax.legend()
ax.set_xlim(0.8, 1.01)
ax.set_ylim(0.8, 1.01)
plt.tight_layout()
plt.savefig('reports/figures/pr_curves.png', dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Metrics vs Threshold — All Models', fontsize=13, fontweight='bold')

for ax, (name, r) in zip(axes, results.items()):
    thresholds = r['thresholds']
    precision  = r['precision'][:-1]
    recall     = r['recall'][:-1]
    f1         = r['f1_scores'][:-1]

    ax.plot(thresholds, precision, label='Precision', color='blue',  linewidth=2)
    ax.plot(thresholds, recall,    label='Recall',    color='green', linewidth=2)
    ax.plot(thresholds, f1,        label='F1',        color='red',   linewidth=2, linestyle='--')
    ax.axvline(x=r['best_threshold'], color='black', linestyle=':',
               label=f"Best t={r['best_threshold']:.2f}")

    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    ax.set_ylim(0.7, 1.02)

plt.tight_layout()
plt.savefig('reports/figures/metrics_vs_threshold.png', dpi=150)
plt.close()