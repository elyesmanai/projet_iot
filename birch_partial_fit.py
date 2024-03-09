birch = Birch(n_clusters=2)

models = [("birch", birch)]

results = []
for model_name, model in models:  
    batch_size = 20000
    for i in tqdm(range(0, len(X2), batch_size)):
        batch = X2[i:i+batch_size]
        birch.partial_fit(batch)
    preds = model.labels_
    results.append(evaluate_unsupervised_model(model_name, model, X2, y2))
    joblib.dump(model, f'models/TON_unsupervised_binary_{model_name}.joblib')

table = tabulate(results, headers=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'], tablefmt='grid')
print(table)
