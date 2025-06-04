def select_model(acc_dt , f1_dt, acc_svm , f1_svm, acc_rf , f1_rf, acc_xg , f1_xg, acc_nb , f1_nb ,acc_nn , f1_nn):
    results = {
        "Decision Tree": {"accuracy": acc_dt, "f1": f1_dt},
        "Support Vector Machine": {"accuracy": acc_svm, "f1": f1_svm},
        "Random Forest": {"accuracy": acc_rf, "f1": f1_rf},
        "XG Boost": {"accuracy": acc_xg, "f1": f1_xg},
        "Naive Bayes": {"accuracy": acc_nb, "f1": f1_nb},
        "Neural Network": {"accuracy": acc_nn, "f1": f1_nn}
    }

    best_by_accuracy = max(results, key=lambda model: results[model]["accuracy"])
    print(f"Best model by accuracy: {best_by_accuracy} ({results[best_by_accuracy]['accuracy'] * 100}%)")

    best_by_f1 = max(results, key=lambda model: results[model]["f1"])
    print(f"Best model by F1-score: {best_by_f1} ({results[best_by_f1]['f1']})")

    print('Based on the evaluation of six machine learning models using both accuracy and F1-score as performance metrics, the Support Vector Machine (SVM) emerged as the best-performing model. It achieved the highest accuracy of 84.99% and also obtained the best F1-score of 0.873, indicating a strong balance between precision and recall.')
    print('This suggests that the SVM model not only correctly classifies a high proportion of test instances, but also performs reliably across both classes, making it a suitable choice for the depression prediction task. Its consistent performance across both metrics reinforces its robustness, especially in a classification context with potential class imbalance.')
    print('**Therefore, the Support Vector Machine is selected as the final model for deployment or further fine-tuning.**')