import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

def preprocess_data(file_path, save_path=None):

    # Load dataset
    print(f"Loading dataset dari {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Cek dan tangani missing values
    missing_count = df.isnull().sum().sum()
    print(f"Terdapat {missing_count} nilai yang hilang dalam dataset")
    
    if missing_count > 0:
        print("Mengisi missing values dengan nilai median...")
        df.fillna(df.median(), inplace=True)
    
    # 2. Memisahkan fitur dan target untuk proses preprocessing
    X = df.drop('Potability', axis=1).values
    y = df['Potability'].values
    
    # 3. Cek ketidakseimbangan kelas
    class_counts = np.bincount(y)
    print(f"Distribusi kelas awal: Kelas 0: {class_counts[0]}, Kelas 1: {class_counts[1]}")
    
    # 4. Lakukan oversampling untuk mengatasi ketidakseimbangan kelas
    print("Melakukan oversampling untuk menyeimbangkan kelas...")
    over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = over_sampler.fit_resample(X, y)
    
    # Cek distribusi kelas setelah oversampling
    resampled_counts = np.bincount(y_resampled)
    print(f"Distribusi kelas setelah oversampling: Kelas 0: {resampled_counts[0]}, Kelas 1: {resampled_counts[1]}")
    
    # 5. Scaling fitur ke range (-1, 1)
    print("Melakukan scaling fitur ke range (-1, 1)...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_resampled)
    
    # 6. Buat DataFrame baru dengan data yang telah diproses
    feature_names = df.drop('Potability', axis=1).columns
    processed_df = pd.DataFrame(X_scaled, columns=feature_names)
    processed_df['Potability'] = y_resampled
    
    # 7. Simpan hasil preprocessing jika save_path disediakan
    if save_path:
        print(f"Menyimpan dataset hasil preprocessing ke {save_path}")
        processed_df.to_csv(save_path, index=False)
    
    print(f"Preprocessing selesai! Dataset memiliki {processed_df.shape[0]} baris dan {processed_df.shape[1]} kolom")
    
    return processed_df

def get_feature_names(file_path):
   
    df = pd.read_csv(file_path)
    features = df.drop('Potability', axis=1).columns.tolist()
    return features

if __name__ == "__main__":
    import sys
    
    # Default path jika tidak ada argumen yang diberikan
    file_path = "water_potability_raw.csv"
    save_path = "preprocessing/water_potability_preprocessed.csv"
    
    # Jika ada argumen path file, gunakan itu
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # Jika ada argumen untuk save path
    if len(sys.argv) > 2:
        save_path = sys.argv[2]
    
    try:
        processed_df = preprocess_data(file_path, save_path)
        print(f"\nPreprocessing berhasil! dan disimpan di {save_path}")
    except Exception as e:
        print(f"\nTerjadi kesalahan: {e}")
        print("Pastikan file data tersedia di path yang benar.")