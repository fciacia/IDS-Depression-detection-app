from data_loader import load_data, save_data
from data_preprocessing import clean_transform
from eda import plot_graph
from label_encoders_split_data_model import label_encoder
from model_training_testing import model_train_test
from model_selection import select_model

RAW_DATA_PATH = '/Users/limjiaqian/Library/Mobile Documents/com~apple~CloudDocs/WIE2003-Intro to DS/Reproducible Research/Student Depression Dataset.csv'
CLEAN_DATA_PATH = '/Users/limjiaqian/Library/Mobile Documents/com~apple~CloudDocs/WIE2003-Intro to DS/Reproducible Research/Clean Student Depression Dataset.csv'
df = load_data(RAW_DATA_PATH)
df_cleaned = clean_transform(df)
save_data(df_cleaned, CLEAN_DATA_PATH)

plot_graph(df_cleaned)

df_model = load_data(CLEAN_DATA_PATH)
X_train, X_test, y_train, y_test = label_encoder(df_model)

acc_dt , f1_dt, acc_svm , f1_svm, acc_rf , f1_rf, acc_xg , f1_xg, acc_nb , f1_nb ,acc_nn , f1_nn = model_train_test(X_train, X_test, y_train, y_test, df_model)

select_model(acc_dt , f1_dt, acc_svm , f1_svm, acc_rf , f1_rf, acc_xg , f1_xg, acc_nb , f1_nb ,acc_nn , f1_nn)
