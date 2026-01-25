import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
# Importação necessária para ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
# Importações adicionais para métricas de avaliação
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from keras.layers import LeakyReLU
import time # Importação do módulo time para medir o tempo

# --- 1. Configurações Globais ---
DATASET_PATH = ''
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 4 # Ajuste conforme seu dataset (2, 3, 4, etc.)
BATCH_SIZE = 8
EPOCHS = 500
LEARNING_RATE = 0.0001
NUM_CHANNELS = 3
# Opção de validação: 'cross_validation' ou 'classic_split'
VALIDATION_METHOD = 'classic_split'

# --- NOVO: Tamanhos das divisões para 'classic_split' ---
# Se TEST_SPLIT_SIZE = 0.15 e VALIDATION_SPLIT_SIZE = 0.15, o Treino será 1 - 0.15 - 0.15 = 0.70
TEST_SPLIT_SIZE = 0.10 
VALIDATION_SPLIT_SIZE = 0.10 
# -------------------------------------------------------

# --- 2. Callback para Métricas Detalhadas por Época ---
class MetricsHistory(keras.callbacks.Callback):
    """
    Callback customizado para calcular a acurácia por classe em cada época
    e salvar a história. Monitora o conjunto de VALIDAÇÃO.
    """
    def __init__(self, val_data, class_names):
        super().__init__()
        self.val_data = val_data
        self.class_names = class_names
        self.history = {'per_class_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 1. Obter Predições e Rótulos Verdadeiros
        all_true = []
        all_pred_classes = []
        
        # Iterar sobre o tf.data.Dataset de validação
        for x_batch, y_batch in self.val_data:
            y_pred = self.model.predict_on_batch(x_batch)
            
            true_classes = np.argmax(y_batch.numpy(), axis=1)
            predicted_classes = np.argmax(y_pred, axis=1)
            
            all_true.extend(true_classes)
            all_pred_classes.extend(predicted_classes)
        
        all_true = np.array(all_true)
        all_pred_classes = np.array(all_pred_classes)
        
        # 2. Calcular Acurácia Por Classe
        # É necessário definir os labels explicitamente para que a matriz de confusão tenha o tamanho correto
        labels_cm = np.arange(len(self.class_names))
        cm = confusion_matrix(all_true, all_pred_classes, labels=labels_cm)
        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            tp_fp = cm.sum(axis=1)[i] # Total de amostras na classe i (Verdadeiros)
            tp = cm[i, i]            # Verdadeiros Positivos
            acc = tp / tp_fp if tp_fp > 0 else 0.0
            per_class_acc[class_name] = acc
        
        self.history['per_class_accuracy'].append(per_class_acc)


# --- 3. Função para Imprimir o Tempo de Treinamento ---
def print_training_time(start_time):
    """Calcula e imprime o tempo total de execução."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "="*80)
    print(f"TEMPO TOTAL DE TREINAMENTO: {hours:02d}h {minutes:02d}m {seconds:02d}s ({elapsed_time:.2f} segundos)")
    print("="*80)
    
    return elapsed_time

# --- Funções do Modelo e Dados ---
def build_model(model_name='ResNet50V2', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES):
    """ Constrói um modelo de classificação (CNN simples ou Backbone pré-treinado). """
    if model_name == 'SimpleCNN':
        model = Sequential([
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.MaxPool2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
    else:
        inputs = keras.Input(shape=input_shape)
        x = inputs
        x = layers.Rescaling(1./255)(x)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomTranslation(height_factor=0.2, width_factor=0.2)(x)
        x = layers.RandomContrast(0.2)(x)

        if model_name == 'InceptionV3_1_Channel':
            base_model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=None)
        elif model_name == 'ResNet50V2':
            base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'MobileNetV2':
            base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'EfficientNetB0':
            base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'VGG16':
            base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'DenseNet121':
            base_model = tf.keras.applications.DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'Xception':
            base_model = tf.keras.applications.Xception(input_shape=input_shape, include_top=False, weights='imagenet')
        else:
            raise ValueError(f"Modelo '{model_name}' não suportado.")

        base_model.trainable = True
        x = base_model(x, training = True)
        x = layers.GlobalAveragePooling2D()(x)
        
        #x = layers.Dense(64)(x)
       # x = layers.LeakyReLU(negative_slope=0.00001)(x)

        predictions = layers.Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model

def preprocess_image_for_tf_data(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # Mantenho 3, ajuste para 1 se NUM_CHANNELS=1
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img, label

def load_data_from_directory(dataset_path):
    images_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    if not class_names:
        raise ValueError(f"Nenhum subdiretório de classe encontrado em '{dataset_path}'.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    print(f"Classes detectadas: {class_names}")
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_to_idx[class_name])
    if not images_paths:
        raise ValueError(f"Nenhuma imagem encontrada em '{dataset_path}'.")
    return np.array(images_paths), np.array(labels), class_names

def plot_confusion_matrix(cm, class_names, title='Matriz de Confusão', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    #plt.savefig("confusino.png")
    plt.show()

# --- 5. Função Principal de Treinamento e Avaliação ---
def train_and_evaluate(model_name='ResNet50V2'):
    
    start_time = time.time()
    
    print(f"Iniciando treinamento com o modelo: {model_name}")

    images_paths, labels, class_names = load_data_from_directory(DATASET_PATH)
    
    # ----------------------------------------------------
    # CORREÇÃO: Garante que NUM_CLASSES seja o número de classes detectado
    global NUM_CLASSES
    if len(class_names) != NUM_CLASSES:
         print(f"Aviso: NUM_CLASSES global ({NUM_CLASSES}) não corresponde às classes detectadas ({len(class_names)}). Corrigindo para {len(class_names)}.")
         NUM_CLASSES = len(class_names) 
    # ----------------------------------------------------
         
    
    BEST_MODEL_PATH = "best_model.weights.h5"
    
    # ----------------------------------------------------
    # DIVISÃO EM TRÊS CONJUNTOS: TREINO / VALIDAÇÃO / TESTE
    # ----------------------------------------------------
    test_size_ratio = TEST_SPLIT_SIZE
    validation_size_ratio = VALIDATION_SPLIT_SIZE
    
    if (test_size_ratio + validation_size_ratio) >= 1.0:
        raise ValueError("A soma de TEST_SPLIT_SIZE e VALIDATION_SPLIT_SIZE deve ser menor que 1.0.")
        
    # 1. Split Inicial: Separa o Conjunto de Teste (imparcial)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        images_paths, labels, test_size=test_size_ratio, stratify=labels, random_state=42
    )
    
    # 2. Split Secundário: Divide o conjunto (Treino + Validação)
    remaining_size = 1.0 - test_size_ratio
    new_val_ratio = validation_size_ratio / remaining_size
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=new_val_ratio, stratify=train_val_labels, random_state=42
    )
    
    # ----------------------------------------------------
    # Inicialização de Variáveis e Métricas
    # ----------------------------------------------------
    fold_accuracies = [] 
    fold_losses = []     
    all_true_labels_best_epoch = []
    all_predicted_labels_best_epoch = []
    total_params = 0
    
    print(f"\n--- Divisão Clássica (Treino/Validação/Teste) ---")
    print(f"  Total de Imagens: {len(images_paths)}")
    print(f"  Treinamento: {len(train_paths)} ({len(train_paths)/len(images_paths):.2%})")
    print(f"  Validação:   {len(val_paths)} ({len(val_paths)/len(images_paths):.2%})")
    print(f"  Teste:       {len(test_paths)} ({len(test_paths)/len(images_paths):.2%})")
    
    # Calculo das amostras vistas
    num_train_images = len(train_paths)
    steps_per_epoch = int(np.ceil(num_train_images / BATCH_SIZE))
    total_samples_per_epoch = steps_per_epoch * BATCH_SIZE
    
    print(f"  Imagens originais de Treino: {num_train_images}")
    print(f"  Amostras vistas por Época (Data Aug. Dinâmico): {total_samples_per_epoch:,} ({steps_per_epoch} steps)")
    
    # Datasets para Treinamento e VALIDAÇÃO (usado no fit)
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, tf.keras.utils.to_categorical(val_labels, num_classes=NUM_CLASSES)))
    train_ds = train_ds.map(preprocess_image_for_tf_data, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=1024).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image_for_tf_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Datasets para Predição de Validação (para o Callback)
    val_ds_for_predict = tf.data.Dataset.from_tensor_slices((val_paths, tf.keras.utils.to_categorical(val_labels, num_classes=NUM_CLASSES)))
    val_ds_for_predict = val_ds_for_predict.map(preprocess_image_for_tf_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Criação do Modelo
    model = build_model(model_name=model_name, input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
    total_params = model.count_params()
    print(f"Número total de parâmetros do modelo: {total_params:,}")
    
    # Callbacks (Monitoram a performance no conjunto de VALIDAÇÃO)
    model_checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=0
    )
    metrics_callback = MetricsHistory(val_ds_for_predict, class_names)
    
    # ----------------------------------------------------
    # Inicialização de report_data (CORRIGIDO)
    report_data = {
        'model_name': model_name,
        'validation_method': 'classic_split (Treino/Val/Teste)',
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'num_channels': NUM_CHANNELS,
        'total_parameters': total_params,
        'class_names': class_names,      # <-- CHAVE 'class_names' ADICIONADA AQUI
        'fold_results': [] 
    }
    # ----------------------------------------------------

    # --- Treinamento ---
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds, # Usa o conjunto de VALIDAÇÃO para monitoramento
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint, metrics_callback],
        verbose=1
    )
    
    # --- Avaliação e Relatório ---
    
    # Encontra a melhor época (baseada na val_accuracy)
    best_epoch_index = np.argmax(history.history['val_accuracy'])
    best_val_acc = history.history['val_accuracy'][best_epoch_index]
    best_val_loss = history.history['val_loss'][best_epoch_index]

    print(f"Melhor Época (Val Acc): {best_epoch_index + 1} (Acc: {best_val_acc:.4f})")
    
    # Carrega os pesos da melhor época salva
    model.load_weights(BEST_MODEL_PATH)
    
    # ----------------------------------------------------------------------
    # Predição final no conjunto de TESTE (Avaliação Imparcial)
    # ----------------------------------------------------------------------
    print(f"Realizando avaliação final no conjunto de TESTE ({len(test_paths)} imagens)...")

    test_ds_for_predict = tf.data.Dataset.from_tensor_slices((test_paths, tf.keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)))
    test_ds_for_predict = test_ds_for_predict.map(preprocess_image_for_tf_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Avaliação final das métricas de LOSS e ACC
    loss_test, accuracy_test = model.evaluate(test_ds_for_predict, verbose=0)
    
    # Predição para a Matriz de Confusão e Métricas F1/Recall
    predictions = model.predict(test_ds_for_predict)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes_test = test_labels
    
    all_true_labels_best_epoch.extend(true_classes_test)
    all_predicted_labels_best_epoch.extend(predicted_classes)
    
    fold_losses.append(loss_test)
    fold_accuracies.append(accuracy_test)


    # --- Geração dos Gráficos de Treinamento/Validação ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss de Treino')
    plt.plot(history.history['val_loss'], label='Loss de Validação')
    plt.title('Perda por Época')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.savefig("grafico_epoca.png")
    plt.show()

    # --- Coleta de Dados para o Relatório ---
    
    # Adicionando o resultado do 'fold' único
    fold_result = {
            'fold': 1,
            'num_train_images_original': num_train_images,
            'total_samples_per_epoch': total_samples_per_epoch,
            'best_epoch': best_epoch_index + 1,
            'final_test_loss': loss_test,
            'final_test_accuracy': accuracy_test,
            'metrics_per_epoch': {
                'train_loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'train_accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
                'per_class_accuracy': metrics_callback.history['per_class_accuracy']
            }
        }
    report_data['fold_results'].append(fold_result)

    # --- Resultados Finais e Matriz de Confusão ---
    f1_macro = f1_score(all_true_labels_best_epoch, all_predicted_labels_best_epoch, average='macro', zero_division=0)
    recall_macro = recall_score(all_true_labels_best_epoch, all_predicted_labels_best_epoch, average='macro', zero_division=0)
    precision_macro = precision_score(all_true_labels_best_epoch, all_predicted_labels_best_epoch, average='macro', zero_division=0)
    
    f1_per_class = f1_score(all_true_labels_best_epoch, all_predicted_labels_best_epoch, average=None, zero_division=0)
    recall_per_class = recall_score(all_true_labels_best_epoch, all_predicted_labels_best_epoch, average=None, zero_division=0)
    precision_per_class = precision_score(all_true_labels_best_epoch, all_predicted_labels_best_epoch, average=None, zero_division=0)
    
    cm = confusion_matrix(all_true_labels_best_epoch, all_predicted_labels_best_epoch, labels=np.arange(NUM_CLASSES))
    plot_confusion_matrix(cm, class_names, title='Matriz de Confusão Global (Conjunto de Teste)')
    
    # --- Geração do Relatório Final Detalhado ---
    print("\n" + "="*80)
    print("RELATÓRIO DE TREINAMENTO DETALHADO")
    print("="*80)
    
    print(f"Parâmetros do Modelo: {report_data['total_parameters']:,}")
    print(f"Modelo: {report_data['model_name']}")
    print(f"Método de Validação: {report_data['validation_method']}")
    print(f"Épocas: {report_data['epochs']}")
    print(f"Taxa de Aprendizagem: {report_data['learning_rate']}")
    print(f"Canais de Entrada Configurados: {report_data['num_channels']}")
    print(f"Classes: {report_data['class_names']}")
    
    print("\n--- Métricas Finais (Conjunto de Teste) ---")
    print(f"Acurácia Final no Teste: {accuracy_test:.4f}")
    print(f"Perda Final no Teste: {loss_test:.4f}")
    print(f"Recall (Macro Average): {recall_macro:.4f}")
    print(f"F1 Score (Macro Average): {f1_macro:.4f}")
    
    print("Métricas Por Classe:")
    for i, class_name in enumerate(class_names):
        print(f"  Classe {class_name}: F1={f1_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, Precision={precision_per_class[i]:.4f}")

    print("\n--- Resultados Detalhados por Época (Conjunto de Validação) ---")
    
    class_acc_header = "".join([f" | Acc {cls.split('_')[0][:4]}" for cls in class_names]) 
    header = f"  Época | Train Loss | Val Loss | Train Acc | Val Acc{class_acc_header}"
    print(header)
    print("-" * (len(class_names) * 11 + 55)) 

    fold_result = report_data['fold_results'][0]
    print(f"\n[Treinamento Completo] (Original: {fold_result['num_train_images_original']} imgs | Amostras/Época: {fold_result['total_samples_per_epoch']:,}) (Melhor Época: {fold_result['best_epoch']})")
    
    num_epochs = len(fold_result['metrics_per_epoch']['train_loss'])
    for i in range(num_epochs):
        epoch_data = fold_result['metrics_per_epoch']
        
        class_acc_values = "".join([f" | {epoch_data['per_class_accuracy'][i].get(cls, 0.0):.4f}" for cls in class_names])

        print(
            f"  {i+1:5d} | {epoch_data['train_loss'][i]:.6f} | {epoch_data['val_loss'][i]:.6f} | "
            f"{epoch_data['train_accuracy'][i]:.6f} | {epoch_data['val_accuracy'][i]:.6f}"
            f"{class_acc_values}"
        )

    print("\n--- Matriz de Confusão Global (Conjunto de Teste) ---")
    print(cm)
    print(f"Ordem das Classes: {report_data['class_names']}")
    
    print_training_time(start_time)


# --- Como Usar ---
if __name__ == '__main__':
    if DATASET_PATH == '':
         print("Por favor, defina a variável global DATASET_PATH com o caminho correto do seu dataset.")
    elif not os.path.exists(DATASET_PATH):
        print(f"Erro: O diretório do dataset '{DATASET_PATH}' não foi encontrado.")
        print("Estrutura esperada: seu_projeto/classe_0/, seu_projeto/classe_1/, etc.")
    else:
        print("--- Testando com a CNN MobileNetV2 ---")
        train_and_evaluate(model_name='MobileNetV2')
