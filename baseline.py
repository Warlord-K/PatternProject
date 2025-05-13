import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import argparse
from tqdm import tqdm
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/Users/yatharthgupta/vscode/PatternProject/data/4/train.csv', help='path to csv file')
    parser.add_argument('--data_root', type=str, default='/Users/yatharthgupta/vscode/PatternProject/data/4', help='path to image data root')
    parser.add_argument('--model_path', type=str, default='./baseline_model.pkl', help='path to save model')
    parser.add_argument('--train_size', type=int, default=8000, help='number of training images')
    parser.add_argument('--test_size', type=int, default=800, help='number of test images')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        features = []
        
        for i, color in enumerate(cv2.split(img)):
            hist = cv2.calcHist([color], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        for i, color in enumerate(cv2.split(hsv)):
            hist = cv2.calcHist([color], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        gray = np.uint8(gray)
        features.append(np.var(gray))
        features.append(np.std(gray))
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features.append(np.mean(np.abs(sobelx)))
        features.append(np.mean(np.abs(sobely)))
        features.append(np.var(np.abs(sobelx)))
        features.append(np.var(np.abs(sobely)))
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        win_size = (256, 256)
        block_size = (64, 64)
        block_stride = (32, 32)
        cell_size = (16, 16)
        nbins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(img)
        features.append(np.mean(hog_features))
        features.append(np.std(hog_features))
        
        noise_level = estimate_noise(gray)
        features.append(noise_level)
        
        jpeg_score = estimate_jpeg_quality(img)
        features.append(jpeg_score)
        
        for channel in cv2.split(img):
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.median(channel))
            features.append(np.percentile(channel, 25))
            features.append(np.percentile(channel, 75))
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def estimate_noise(gray_img):
    gray_img = gray_img.astype(np.float32)
    
    M = np.float32([[1, -1]])
    horizontal_diff = cv2.filter2D(gray_img, -1, M)
    vertical_diff = cv2.filter2D(gray_img, -1, M.T)
    
    horizontal_sigma = np.std(horizontal_diff)
    vertical_sigma = np.std(vertical_diff)
    
    noise_sigma = (horizontal_sigma + vertical_sigma) / 2.0
    
    return noise_sigma

def estimate_jpeg_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h_diff = np.abs(gray[:, 1:] - gray[:, :-1])
    v_diff = np.abs(gray[1:, :] - gray[:-1, :])
    
    h_block_diff = np.mean(h_diff[:, 7::8])
    v_block_diff = np.mean(v_diff[7::8, :])
    
    h_avg_diff = np.mean(h_diff)
    v_avg_diff = np.mean(v_diff)
    
    if h_avg_diff > 0 and v_avg_diff > 0:
        jpeg_score = 0.5 * (h_block_diff / h_avg_diff + v_block_diff / v_avg_diff)
    else:
        jpeg_score = 0
    
    return jpeg_score

def load_dataset_from_csv(csv_path, data_root, train_size, test_size, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    all_image_paths = []
    all_labels = []
    
    for _, row in df.iterrows():
        file_path = os.path.join(data_root, row['file_name'])
        if os.path.exists(file_path):
            all_image_paths.append(file_path)
            all_labels.append(int(row['label']))
    
    real_indices = [i for i, label in enumerate(all_labels) if label == 1]
    fake_indices = [i for i, label in enumerate(all_labels) if label == 0]
    
    print(f"Found {len(real_indices)} real images and {len(fake_indices)} AI-generated images")
    
    min_samples = min(len(real_indices), len(fake_indices), train_size + test_size)
    if min_samples < train_size + test_size:
        train_size = int(min_samples * 0.9)
        test_size = min_samples - train_size
        print(f"Adjusting to {train_size} training samples and {test_size} test samples")
    
    half_train_size = train_size // 2
    half_test_size = test_size // 2
    
    random.shuffle(real_indices)
    random.shuffle(fake_indices)
    
    train_indices = real_indices[:half_train_size] + fake_indices[:half_train_size]
    test_indices = real_indices[half_train_size:half_train_size+half_test_size] + \
                  fake_indices[half_train_size:half_train_size+half_test_size]
    
    train_paths = [all_image_paths[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    test_paths = [all_image_paths[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]
    
    train_data = list(zip(train_paths, train_labels))
    random.shuffle(train_data)
    train_paths, train_labels = zip(*train_data)
    
    test_data = list(zip(test_paths, test_labels))
    random.shuffle(test_data)
    test_paths, test_labels = zip(*test_data)
    
    return train_paths, train_labels, test_paths, test_labels

def train_model(train_images, train_labels, test_images, test_labels, model_path):
    train_features = []
    final_train_labels = []
    
    for img_path, label in tqdm(zip(train_images, train_labels), total=len(train_images)):
        features = extract_features(img_path)
        if features is not None:
            train_features.append(features)
            final_train_labels.append(label)
    
    test_features = []
    final_test_labels = []
    
    for img_path, label in tqdm(zip(test_images, test_labels), total=len(test_images)):
        features = extract_features(img_path)
        if features is not None:
            test_features.append(features)
            final_test_labels.append(label)
    
    X_train = np.array(train_features)
    y_train = np.array(final_train_labels)
    X_test = np.array(test_features)
    y_test = np.array(final_test_labels)
    
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    print(f"Feature vector size: {X_train.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    
    joblib.dump({'model': svm, 'scaler': scaler}, model_path)
    
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['AI-generated', 'Real']))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]
    
    return accuracy

def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.csv_path}")
    train_images, train_labels, test_images, test_labels = load_dataset_from_csv(
        args.csv_path, args.data_root, args.train_size, args.test_size, args.seed
    )
    
    print(f"Training classical ML model with {len(train_images)} images")
    accuracy = train_model(train_images, train_labels, test_images, test_labels, args.model_path)
    
    print(f"Training complete. Model saved to {args.model_path}")
    print(f"Final test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()