from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def model_train_test(X_train, X_test, y_train, y_test, data):
    acc_dt , f1_dt= Decision_tree(X_train, X_test, y_train, y_test)
    acc_svm , f1_svm=Support_vector_machine(X_train, X_test, y_train, y_test)
    acc_rf , f1_rf=Random_forest(X_train, X_test, y_train, y_test)
    acc_xg , f1_xg=XGBoost(X_train, X_test, y_train, y_test)
    acc_nb , f1_nb=Naive_bayes(X_train, X_test, y_train, y_test)
    acc_nn , f1_nn=Neural_Network(data)
    return acc_dt , f1_dt, acc_svm , f1_svm, acc_rf , f1_rf, acc_xg , f1_xg, acc_nb , f1_nb , acc_nn , f1_nn

def Decision_tree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    # Create the model
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    # Train the model
    dt_model.fit(X_train, y_train)

    # Test the model
    y_pred = dt_model.predict(X_test)

    labels = ['Class 0', 'Class 1']

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Raw counts
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Decision Tree Confusion Matrix (Counts)")
    report_text = f"Decision Tree Accuracy: {acc * 100:.2f}%\n\nClassification Report:\n{classification_report(y_test, y_pred)}"
    plt.text(0.5, 1.05, report_text,
             fontsize=10,
             ha='center',
             transform=plt.gca().transAxes,
             family='monospace')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    acc_dt = accuracy_score(y_test, y_pred)
    f1_dt = f1_score(y_test, y_pred)
    return acc_dt, f1_dt

def Support_vector_machine(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC

    SVC_model = SVC(kernel='linear', random_state=42)
    SVC_model.fit(X_train, y_train)

    y_pred = SVC_model.predict(X_test)

    labels = ['Class 0', 'Class 1']

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Raw counts
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Support Vector Machine Confusion Matrix (Counts)")
    report_text = f"Support Vector Machine Model Accuracy: {acc * 100:.2f}%\n\nClassification Report:\n{classification_report(y_test, y_pred)}"
    plt.text(0.5, 1.05, report_text,
             fontsize=10,
             ha='center',
             transform=plt.gca().transAxes,
             family='monospace')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    acc_svm = accuracy_score(y_test, y_pred)
    f1_svm = f1_score(y_test, y_pred)
    return acc_svm, f1_svm

def Random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV

    labels = ['Class 0', 'Class 1']

    # Initial model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    cm1 = confusion_matrix(y_test, y_pred)
    acc1 = accuracy_score(y_test, y_pred)
    report1 = classification_report(y_test, y_pred)

    # Hyperparameter tuning
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 0.5],
        'bootstrap': [True],
        'class_weight': ['balanced', None]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    halving_search = HalvingRandomSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        factor=2,
        resource='n_estimators',
        max_resources=300,
        min_resources=100,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    halving_search.fit(X_train, y_train)
    best_rf = halving_search.best_estimator_
    b_y_pred = best_rf.predict(X_test)

    cm2 = confusion_matrix(y_test, b_y_pred)
    acc2 = accuracy_score(y_test, b_y_pred)
    report2 = classification_report(y_test, b_y_pred)

    # Plot both confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before Tuning
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Before Tuning Accuracy: {:.2f}%".format(acc1 * 100))
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    report_text_1 = f"Random Forest Accuracy: {acc1 * 100:.2f}%\n\nClassification Report:\n{classification_report(y_test, y_pred)}"
    axes[0].text(0.5, 1.05, report_text_1,
                 fontsize=9,
                 ha='center',
                 transform=axes[0].transAxes,
                 family='monospace')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # After Tuning
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("After Tuning Accuracy: {:.2f}%".format(acc2 * 100))
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("")

    report_text_2 = f"Random Forest Accuracy: {acc2 * 100:.2f}%\n\nClassification Report:\n{classification_report(y_test, b_y_pred)}"
    axes[1].text(0.5, 1.05, report_text_2,
                 fontsize=9,
                 ha='center',
                 transform=axes[1].transAxes,
                 family='monospace')

    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()

    acc_rf = accuracy_score(y_test, b_y_pred)
    f1_rf = f1_score(y_test, b_y_pred)
    return acc_rf, f1_rf

def XGBoost(X_train, X_test, y_train, y_test):
    import sys
    import subprocess

    # Try to import xgboost, install if not found
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        import xgboost as xgb

    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Your existing train/test data: X_train, y_train, X_test, y_test

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )

    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=30,
        scoring='accuracy',
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("Best hyperparameters found:")
    print(random_search.best_params_)

    best_xgb = random_search.best_estimator_

    y_pred = best_xgb.predict(X_test)

    labels = ['Class 0', 'Class 1']

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Raw counts
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("XGBoost Confusion Matrix (Counts)")
    report_text = f"XGBoost Accuracy: {acc * 100:.2f}%\n\nClassification Report:\n{classification_report(y_test, y_pred)}"
    plt.text(0.5, 1.05, report_text,
             fontsize=10,
             ha='center',
             transform=plt.gca().transAxes,
             family='monospace')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    acc_xg = accuracy_score(y_test, y_pred)
    f1_xg = f1_score(y_test, y_pred)
    return acc_xg, f1_xg

def Naive_bayes(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)

    y_pred = model_nb.predict(X_test)
    labels = ['Class 0', 'Class 1']

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Raw counts
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Naive Bayes Matrix (Counts)")
    report_text = f"Naive Bayes Accuracy: {acc * 100:.2f}%\n\nClassification Report:\n{classification_report(y_test, y_pred)}"
    plt.text(0.5, 1.05, report_text,
             fontsize=10,
             ha='center',
             transform=plt.gca().transAxes,
             family='monospace')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    acc_nb = accuracy_score(y_test, y_pred)
    f1_nb = f1_score(y_test, y_pred)
    return acc_nb, f1_nb

def Neural_Network(data):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Apply Label Encoding to 'degree'
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    label_encoder = LabelEncoder()
    data['degree'] = label_encoder.fit_transform(data['degree'])
    data.head()
    # degree section is object type. change to int type for NN purpose.

    # List of columns to one-hot encode
    one_hot_cols = ['academic_pressure', 'study_satisfaction', 'dietary_habits', 'financial_stress']

    # Apply one-hot encoding
    data = pd.get_dummies(data, columns=one_hot_cols)

    data = data.astype(np.float32)
    # change to float type, remove first column

    # Separate input and output
    X = data.drop('depression', axis=1).values  # Input features
    y = data['depression'].values  # Output target

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).unsqueeze(1)  # shape: (N, 1)

    # Split into 80% train, 20% test
    from torch.utils.data import random_split

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Define Neural Network (no sigmoid in final layer)
    class DepressionNet(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(DepressionNet, self).__init__()
            self.NN_model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)  # Raw scores (logits)
            )

        def forward(self, x):
            return self.NN_model(x)

    # Define number of input features and classes
    input_dim = next(iter(train_loader))[0].shape[1]
    num_classes = 2  # Change if depression has more than 2 classes
    NN_model = DepressionNet(input_dim, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(NN_model.parameters(), lr=0.00001)

    # Train loop
    num_epochs = 20

    for epoch in range(num_epochs):
        NN_model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            targets = targets.squeeze().long()  # Ensure shape [N] and dtype long
            optimizer.zero_grad()
            outputs = NN_model(inputs)  # shape: [batch_size, num_classes]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        NN_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.squeeze().long()
                outputs = NN_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        test_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Set model to evaluation mode
        NN_model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.squeeze().long()
                outputs = NN_model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)

        labels = ['Class 0', 'Class 1']


        # Raw counts
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Naive Bayes Matrix (Counts)")
        report_text = f"Naive Bayes Accuracy: {acc * 100:.2f}%\n\nClassification Report:\n{classification_report(all_labels, all_preds)}"
        plt.text(0.5, 1.05, report_text,
                 fontsize=10,
                 ha='center',
                 transform=plt.gca().transAxes,
                 family='monospace')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        acc_nn = accuracy_score(all_labels, all_preds)
        f1_nn = f1_score(all_labels, all_preds)
        return acc_nn,f1_nn



