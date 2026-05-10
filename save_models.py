import joblib
models_dir = os.path.join(PROJECT_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler_cnn, os.path.join(models_dir, 'scaler_cnn.pkl'))
joblib.dump(scaler_hc, os.path.join(models_dir, 'scaler_hc.pkl'))
with open(os.path.join(models_dir, 'pca_scratch.pkl'), 'wb') as f:
    pickle.dump(pca, f)
joblib.dump(svm_rbf, os.path.join(models_dir, 'svm_rbf.pkl'))
joblib.dump(svm_lin, os.path.join(models_dir, 'svm_linear.pkl'))
joblib.dump(rf, os.path.join(models_dir, 'random_forest.pkl'))
joblib.dump(xgb, os.path.join(models_dir, 'xgboost.pkl'))
np.save(os.path.join(models_dir, 'knn_train_X.npy'), X_train_pca)
np.save(os.path.join(models_dir, 'knn_train_y.npy'), train_labels)
joblib.dump(best_k, os.path.join(models_dir, 'knn_best_k.pkl'))
config = {'best_cnn': best_cnn, 'best_classifier': best_classifier, 'best_k': best_k, 'n_pca_components': n_components, 'img_size': IMG_SIZE, 'cnn_feature_dim': cnn_features_train[best_cnn].shape[1], 'handcrafted_dim': X_train_hc.shape[1]}
with open(os.path.join(models_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)
print(f'All models saved to: {models_dir}')
