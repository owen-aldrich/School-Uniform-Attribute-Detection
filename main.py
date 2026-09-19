import os
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
import pickle
import time

label_map = {
    '0': 'Dasi',           
    '1': 'Logo Sekolah',   
    '2': 'Topi',           
    '3': 'Ikat Pinggang',
    '99': 'Background', 
    0: 'Dasi', 1: 'Logo Sekolah', 2: 'Topi', 3: 'Ikat Pinggang', 99: 'Background'
}

checklist_items = ['Logo Sekolah', 'Dasi', 'Ikat Pinggang', 'Topi']
model_filename = 'svm_seragam_model.pkl'

input_size = (96, 96) 

detect_w, detect_h = 320, 240 
process_every_n_frames = 5 
sliding_window_w, sliding_window_h = 96, 96 
sliding_window_step = 48 

class_thresholds = {
    'Topi': 0.85,          
    'Dasi': 0.85,          
    'Logo Sekolah': 0.80,  
    'Ikat Pinggang': 0.80  
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def augment_patch(patch):
    aug_img = patch.copy()
    h, w = aug_img.shape[:2]
    
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() > 0.5:
        kernel_size = random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)

    if random.random() > 0.5:
        noise = np.zeros((h, w, 3), np.uint8)
        cv2.randn(noise, 0, 15)
        aug_img = cv2.add(aug_img, noise)

    return aug_img

def extract_hog_features(image):
    resized_img = cv2.resize(image, input_size)
    features = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, channel_axis=-1)
    return features

def is_overlap(box1, box2):
    if box1[0] >= box2[2] or box2[0] >= box1[2]: return False
    if box1[1] >= box2[3] or box2[1] >= box1[3]: return False
    return True

def load_data_from_voc(directory, is_training=True):
    X, y = [], []
    if not os.path.exists(directory):
        print(f"[-] Folder '{directory}' tidak ditemukan.")
        return np.array([]), np.array([])
    
    xml_files = [f for f in os.listdir(directory) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(directory, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_filename = root.find('filename').text
        img_path = os.path.join(directory, img_filename)
        
        if not os.path.exists(img_path):
            img_path = xml_path.replace('.xml', '.jpg')
            
        image = cv2.imread(img_path)
        if image is None: continue
        
        img_h, img_w = image.shape[:2]
        obj_boxes = [] 
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label not in ['0', '1', '2', '3'] and label.lower() not in ['dasi', 'logo', 'topi', 'sabuk', 'ikat pinggang']:
                continue

            xmlbox = obj.find('bndbox')
            xmin = max(0, int(float(xmlbox.find('xmin').text)))
            ymin = max(0, int(float(xmlbox.find('ymin').text)))
            xmax = min(img_w, int(float(xmlbox.find('xmax').text)))
            ymax = min(img_h, int(float(xmlbox.find('ymax').text)))
            
            obj_boxes.append([xmin, ymin, xmax, ymax])
            cropped_obj = image[ymin:ymax, xmin:xmax]
            if cropped_obj.size == 0: continue
            
            X.append(extract_hog_features(cropped_obj))
            y.append(label)
            
            if is_training:
                for _ in range(3):
                    aug_patch = augment_patch(cropped_obj)
                    X.append(extract_hog_features(aug_patch))
                    y.append(label)
            
        num_bg_samples = 8 if is_training else 2
        bg_samples = 0
        attempts = 0
        
        while bg_samples < num_bg_samples and attempts < 30:
            rx = random.randint(0, max(1, img_w - input_size[0]))
            ry = random.randint(0, max(1, img_h - input_size[1]))
            random_box = [rx, ry, rx + input_size[0], ry + input_size[1]]
            
            nabrak = False
            for b in obj_boxes:
                if is_overlap(random_box, b):
                    nabrak = True
                    break
                    
            if not nabrak:
                bg_crop = image[random_box[1]:random_box[3], random_box[0]:random_box[2]]
                if bg_crop.shape[0] == input_size[1] and bg_crop.shape[1] == input_size[0]:
                    X.append(extract_hog_features(bg_crop))
                    y.append('99') 
                    bg_samples += 1
            attempts += 1

    return np.array(X), np.array(y)

if os.path.exists(model_filename):
    print(f"[*] Memuat model lama '{model_filename}'...")
    with open(model_filename, 'rb') as file:
        svm_model = pickle.load(file)
else:
    print(f"[*] Training model baru dengan HOG COLOR (96x96)...")
    X_train, y_train = load_data_from_voc('train', is_training=True)
    
    svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    with open(model_filename, 'wb') as file:
        pickle.dump(svm_model, file)
    print("[+] Training selesai! Model telah disimpan.")

print("\n[*] Menjalankan Evaluasi Metrik pada data 'test'...")
X_test, y_test = load_data_from_voc('test', is_training=False)

if len(X_test) > 0:
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)
    
    print("\n" + "="*60)
    print("                METRIC EVALUATION REPORT")
    print("="*60)
    
    unique_labels = sorted(list(set(y_test)))
    target_names = [label_map.get(str(label), f"Class {label}") for label in unique_labels]
    
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"Accuracy Keseluruhan : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    
    y_test_bin = label_binarize(y_test, classes=svm_model.classes_)
    
    ap_scores = []
    print("Average Precision (AP) per kelas:")
    for i, class_label in enumerate(svm_model.classes_):
        if np.sum(y_test_bin[:, i]) > 0:
            ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])
            ap_scores.append(ap)
            nama_kelas = label_map.get(str(class_label), class_label)
            print(f"- {nama_kelas:<15} : {ap*100:.2f}%")
            
    mAP = np.mean(ap_scores)
    print(f"\n=> mAP (Mean Average Precision) : {mAP*100:.2f}%")
    print("="*60 + "\n")

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def apply_nms(boxes, confidences, overlap_threshold=0.1): 
    if len(boxes) == 0: return []
    
    boxes_opencv = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    confidences = [float(c) for c in confidences]
    
    indices = cv2.dnn.NMSBoxes(boxes_opencv, confidences, 0.1, overlap_threshold)
    if len(indices) == 0: return []
        
    results = []
    for i in indices:
        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        results.append(boxes[idx])
        
    return results

cap = cv2.VideoCapture(0)
full_w, full_h = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, full_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, full_h)

fps_display_counter = 0
start_time_display = time.time()
fps_display = 0

frame_counter_processing = 0
last_final_detections = []
last_face_boxes = []
last_checklist_status = {item: False for item in checklist_items}

print("\n[*] Kamera aktif! Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    if frame_counter_processing % process_every_n_frames == 0:
        detect_frame = cv2.resize(frame, (detect_w, detect_h))
        scale_x = full_w / detect_w
        scale_y = full_h / detect_h

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        last_face_boxes = faces 

        scaled_faces = []
        for (fx, fy, fw, fh) in faces:
            scaled_faces.append([int(fx/scale_x), int(fy/scale_y), int((fx+fw)/scale_x), int((fy+fh)/scale_y)])

        checklist_status = {item: False for item in checklist_items}
        all_detections = {item: [] for item in checklist_items}
        all_scores = {item: [] for item in checklist_items}

        for (x, y, roi) in sliding_window(detect_frame, (sliding_window_w, sliding_window_h), sliding_window_step):
            if roi.shape[1] != sliding_window_w or roi.shape[0] != sliding_window_h: continue
                
            window_box = [x, y, x + sliding_window_w, y + sliding_window_h]
            
            nabrak_muka = False
            for fbox in scaled_faces:
                if is_overlap(window_box, fbox):
                    nabrak_muka = True
                    break
            
            if nabrak_muka:
                continue 
                
            try:
                roi_features = extract_hog_features(roi)
                pred_proba_raw = svm_model.predict_proba(roi_features.reshape(1, -1))[0]
                prob_score = pred_proba_raw.max()
                pred_idx = svm_model.predict(roi_features.reshape(1, -1))[0]
                
                label_indo = label_map.get(str(pred_idx), None) 
                
                if label_indo and label_indo in checklist_items:
                    
                    center_y = y + (sliding_window_h // 2)
                    
                    batas_atas = detect_h * 0.35  
                    batas_bawah = detect_h * 0.65  
                    
                    if label_indo == 'Topi' and center_y > batas_atas:
                        continue
                        
                    if label_indo in ['Dasi', 'Logo Sekolah'] and (center_y < (detect_h * 0.25) or center_y > (detect_h * 0.75)):
                        continue

                    if label_indo == 'Ikat Pinggang' and center_y < batas_bawah:
                        continue
                        
                    target_threshold = class_thresholds.get(label_indo, 0.85)

                    if prob_score > target_threshold:
                        all_detections[label_indo].append(window_box)
                        all_scores[label_indo].append(prob_score)
            except Exception:
                pass 

        final_detections = []
        for label, dets in all_detections.items():
            if len(dets) > 0:
                scores = all_scores[label]
                best_dets_detect_size = apply_nms(dets, scores)
                
                if len(best_dets_detect_size) > 0:
                    checklist_status[label] = True 
                    for (x1, y1, x2, y2) in best_dets_detect_size:
                        full_x1 = int(x1 * scale_x)
                        full_y1 = int(y1 * scale_y)
                        full_x2 = int(x2 * scale_x)
                        full_y2 = int(y2 * scale_y)
                        final_detections.append({'box': (full_x1, full_y1, full_x2, full_y2), 'label': label})

        last_final_detections = final_detections
        last_checklist_status = checklist_status

    for (x, y, w, h) in last_face_boxes:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 0), 1)

    for det in last_final_detections:
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
        cv2.rectangle(display_frame, (x1, y1 - 25), (x1 + 140, y1), (0, 0, 0), -1)
        cv2.putText(display_frame, det['label'], (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    overlay = display_frame.copy()
    cv2.rectangle(overlay, (10, 10), (230, 150), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
    
    y_text = 40
    cv2.putText(display_frame, "STATUS KELENGKAPAN:", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for item in checklist_items:
        if last_checklist_status[item]:
            text = f"[ OK ] {item}"
            color = (0, 255, 0)
        else:
            text = f"[ -- ] {item}"
            color = (0, 0, 255)
        cv2.putText(display_frame, text, (15, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_text += 25

    fps_display_counter += 1
    if time.time() - start_time_display > 1:
        fps_display = fps_display_counter / (time.time() - start_time_display)
        fps_display_counter = 0
        start_time_display = time.time()
        
    cv2.putText(display_frame, f"Kamera: {fps_display:.1f} FPS", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('Deteksi Atribut (Logika 1/3 Layar)', display_frame)
    frame_counter_processing += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()