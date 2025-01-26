#!/usr/bin/env python
# coding: utf-8

# # SVM

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer



# In[2]:


# Carga de datos y preprocesamiento
file_path = 'comments.csv'
data = pd.read_csv(file_path, sep='|')


# Vectorización de los comentarios usando TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['Comment'])

y = data['Sentiment'].values

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Codificación de las etiquetas para poder usar compute_class_weight
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

# Calcula los pesos de las clases
class_weights = compute_class_weight('balanced', classes=np.unique(encoded_y_train), y=encoded_y_train)

# Mapeo de los pesos a las etiquetas codificadas
weights = {i: class_weights[i] for i in range(len(np.unique(encoded_y_train)))}

X_train.shape, X_test.shape, y_train.shape, y_test.shape, weights



# ### Parámetros con GridSearch

# #### SVC

# In[3]:


# Codificación de las etiquetas
label_encoder = LabelEncoder()
encoded_y_sample = label_encoder.fit_transform(y_train)

# Configuración de GridSearchCV
param_grid = {
    'C': [0.1,1,10, 100, 500],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': [weights]  # Añadir pesos de clase aquí
}

# Creación del modelo y búsqueda de hiperparámetros
model = SVC()
grid_search = GridSearchCV(model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, encoded_y_sample)

# Mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", grid_search.best_params_)
# Extrae los mejores parámetros 
svc_params = {k: v for k, v in grid_search.best_params_.items()}


# In[9]:


# Crea y entrena el modelo SVM con los parámetros encontrados
model = SVC(**svc_params)
model.fit(X_train, encoded_y_train)

# Evalúa el modelo
encoded_y_test = label_encoder.transform(y_test)
y_pred = model.predict(X_test)
# Convierte las predicciones codificadas de vuelta a etiquetas originales
y_pred = label_encoder.inverse_transform(y_pred)  


# In[6]:


print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# Calcula la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Etiquetas personalizadas para las clases
class_labels = ['Negativo', 'Neutral', 'Positivo']

# Grafica la matriz de confusión como un mapa de calor
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.show()


# ### OvR

# In[11]:


# Codificación de las etiquetas
label_encoder = LabelEncoder()
encoded_y_sample = label_encoder.fit_transform(y_train)
# Configuración de GridSearchCV
param_grid = {
    'estimator__C': [0.1,1,10, 100, 500],
    'estimator__kernel': ['linear', 'rbf'],
    'estimator__gamma': ['scale', 'auto'],
    'estimator__class_weight': [weights]  # Añadir pesos de clase aquí
}

# Creación del modelo y búsqueda de hiperparámetros
model = OneVsRestClassifier(SVC())
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, encoded_y_sample)

# Mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", grid_search.best_params_)
# Extrae los mejores parámetros y elimina el prefijo 'estimator__'
ovr_params = {k.replace('estimator__', ''): v for k, v in grid_search.best_params_.items()}




# In[12]:


# Crea y entrena el modelo SVM con los parámetros encontrados
model = OneVsRestClassifier(SVC(**ovr_params), n_jobs=-1)
model.fit(X_train, encoded_y_train)

# Evalúa el modelo
encoded_y_test = label_encoder.transform(y_test)
y_pred = model.predict(X_test)

# Convierte las predicciones codificadas de vuelta a etiquetas originales
y_pred = label_encoder.inverse_transform(y_pred)  



# In[13]:


print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# Calcula la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Etiquetas personalizadas para las clases
class_labels = ['Negativo', 'Neutral', 'Positivo']

# Grafica la matriz de confusión como un mapa de calor
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.show()


# # Bert

# In[1]:


import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd
import copy


# In[18]:


def evaluate(model, val_loader):
    model.eval()
    total_eval_loss = 0
    predictions, true_labels = [], []

    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        total_eval_loss += outputs.loss.item()
        logits = outputs.logits
        probs = softmax(logits.detach().cpu().numpy(), axis=1)
        pred_labels = np.argmax(probs, axis=1)
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend(pred_labels)
        true_labels.extend(label_ids.flatten())

    avg_val_accuracy = accuracy_score(true_labels, predictions)
    avg_val_loss = total_eval_loss / len(val_loader)

    return avg_val_accuracy, avg_val_loss



# In[4]:


# Configuración de la GPU y optimizaciones de cudnn
torch.backends.cudnn.benchmark = True
# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo y el tokenizador
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.to(device)
print("Modelo cargado exitosamente.")
# Carga de datos
df = pd.read_csv('comments.csv', sep='|')
comments = df['Comment']
labels = df['Sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
print("Datos cargados exitosamente.")
# Tokenización y preparación de los datos
max_length = 128  # Tamañño de la secuencia
batch_size = 16  # Tamaño del lote 
input_ids = tokenizer(comments.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='pt')['input_ids']
attention_masks = tokenizer(comments.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='pt')['attention_mask']
labels = torch.tensor(labels.values)
print("Datos preparados exitosamente.")

# Crear TensorDataset y dividir en conjuntos de entrenamiento y validación
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))  # Cambiado a 80-20 split
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# Crear DataLoaders para la carga de datos en paralelo
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Optimizador y Scheduler
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)  # Ajustamos la tasa de aprendizaje
epochs = 8  # número de épocas
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
print("Optimizador creado exitosamente.")


# Función de evaluación
def evaluate(model, val_loader):
    model.eval()
    total_eval_loss = 0
    predictions, true_labels = [], []

    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        total_eval_loss += outputs.loss.item()
        logits = outputs.logits
        probs = softmax(logits.detach().cpu().numpy(), axis=1)
        pred_labels = np.argmax(probs, axis=1)
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend(pred_labels)
        true_labels.extend(label_ids.flatten())

    avg_val_accuracy = accuracy_score(true_labels, predictions)
    avg_val_loss = total_eval_loss / len(val_loader)

    return avg_val_accuracy, avg_val_loss

# Entrenamiento
# Inicializar variables para la parada temprana
best_val_accuracy = 0.0
best_val_loss = float('inf')  # Añadido para rastrear la mejor pérdida de validación
patience = 3  # Número de épocas para esperar una mejora antes de detener el entrenamiento
patience_counter = 0
# Inicializar variable para guardar el mejor modelo
best_model_state = None
best_epoch = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        batch = [t.to(device) for t in batch]
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss}")

    # Evaluación después de cada época
    val_accuracy, val_loss = evaluate(model, val_loader)  
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Loss: {val_loss}")

    # Lógica de parada temprana basada en la precisión
    if val_accuracy > best_val_accuracy :
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"New best model found at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Parada temprana: La precisión de validación no ha mejorado en las últimas {patience} épocas.")
            break




# In[20]:


def evaluateBestModel(model, val_loader):
    model.eval()
    total_eval_loss = 0
    predictions, true_labels = [], []

    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        total_eval_loss += outputs.loss.item()
        logits = outputs.logits
        probs = softmax(logits.detach().cpu().numpy(), axis=1)
        pred_labels = np.argmax(probs, axis=1)
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend(pred_labels)
        true_labels.extend(label_ids.flatten())

    avg_val_accuracy = accuracy_score(true_labels, predictions)
    avg_val_loss = total_eval_loss / len(val_loader)

    return avg_val_accuracy, avg_val_loss, predictions, true_labels


# In[21]:


# Carga el mejor estado del modelo
if best_model_state:
    model.load_state_dict(best_model_state)
    print("Mejor modelo cargado")

    # Evalúa el modelo y obtiene las predicciones y las etiquetas verdaderas
    val_accuracy, val_loss, predictions, true_labels = evaluateBestModel(model, val_loader)

    print("Validation Loss: ", val_loss)
    print("Validation Accuracy: ", val_accuracy)

    # Calcula la matriz de confusión
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Etiquetas personalizadas para las clases
    class_labels = ['Negative', 'Neutral', 'Positive']

    # Grafica la matriz de confusión como un mapa de calor
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')
    plt.show()

    # Imprime el informe de clasificación
    print(classification_report(true_labels, predictions, target_names=class_labels, zero_division=0))



# ### Análisis de Resultados del Modelo BERT

# In[14]:


def get_exclusive_important_words(important_words_summary):
    # Identificar palabras comunes entre sentimientos
    common_words = set()
    all_sentiments = list(important_words_summary.keys())

    for i, sentiment in enumerate(all_sentiments):
        other_sentiments = all_sentiments[:i] + all_sentiments[i+1:]
        words_current_sentiment = set(word for word, _ in important_words_summary[sentiment])

        for other in other_sentiments:
            words_other_sentiment = set(word for word, _ in important_words_summary[other])
            common_words.update(words_current_sentiment.intersection(words_other_sentiment))

    # Excluir palabras comunes y devolver las 10 palabras exclusivas más importantes por sentimiento
    exclusive_important_words = {}

    for sentiment, words in important_words_summary.items():
        exclusive_words = sorted([(word, count) for word, count in words if word not in common_words], key=lambda x: x[1], reverse=True)[:10]
        exclusive_important_words[sentiment] = exclusive_words

    return exclusive_important_words

# Llamada a la nueva función
exclusive_important_words_summary = get_exclusive_important_words(important_words_summary)

# Imprimir palabras exclusivas por sentimiento
for sentiment, words in exclusive_important_words_summary.items():
    print(f"Palabras más importantes exclusivas de comentarios de la clase {sentiment}:")
    for word, count in words:
        print(f"{word}: {count}")



# In[44]:


import seaborn as sns

# Establecer estilo de los gráficos
sns.set(style="whitegrid")

# Crear un gráfico de barras para cada sentimiento
for sentiment, words in exclusive_important_words_summary.items():
    plt.figure(figsize=(10, 6))  # Tamaño del gráfico
    words, counts = zip(*words)  # Separar palabras y conteos

    # Crear DataFrame para Seaborn
    data = pd.DataFrame({'Word': words, 'Count': counts})

    # Crear el gráfico de barras
    sns.barplot(x='Count', y='Word', data=data, hue='Word', dodge=False, palette='viridis')
    plt.legend([],[], frameon=False)  # Eliminar leyenda

    # Títulos y etiquetas
    plt.title(f'Most Important Exclusive Words for {sentiment.capitalize()} Comments')
    plt.xlabel('Count')
    plt.ylabel('Words')

    # Mostrar el gráfico
    plt.show()



# In[47]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Función para generar una nube de palabras
def generate_wordcloud(words_dict, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words_dict)
    
    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generar y mostrar una nube de palabras para cada sentimiento
for sentiment, words in exclusive_important_words_summary.items():
    # Crear un diccionario de palabras y sus frecuencias
    words_dict = {word: count for word, count in words}
    
    # Generar y mostrar la nube de palabras
    generate_wordcloud(words_dict, f'Exclusive Words for {sentiment.capitalize()} Comments')


# In[49]:


# Suponiendo que df es tu DataFrame y 'TextData' es la columna con los comentarios
# Asumiendo que df es tu DataFrame y 'TextData' es la columna con los comentarios

# Función para encontrar comentarios con palabras clave exclusivas
def find_comments_with_exclusive_words(df, exclusive_words):
    comments_by_sentiment = {sentiment: [] for sentiment in exclusive_words}

    for index, row in df.iterrows():
        text = row['TextData']
        sentiment = row['Sentiment']

        # Verificar si el comentario contiene alguna de las palabras clave exclusivas
        for word in exclusive_words[sentiment]:
            if word in text.split():  # Asumiendo que las palabras están separadas por espacios
                comments_by_sentiment[sentiment].append(text)
                break  # Solo necesitamos saber si al menos una palabra clave está presente

    return comments_by_sentiment

# Extraer las 10 palabras clave exclusivas para cada sentimiento
exclusive_words_by_sentiment = {sentiment: [word for word, _ in words] for sentiment, words in exclusive_important_words_summary.items()}

# Encontrar comentarios
comments_with_exclusive_words = find_comments_with_exclusive_words(df, exclusive_words_by_sentiment)

# Imprimir algunos comentarios para cada sentimiento
for sentiment, comments in comments_with_exclusive_words.items():
    print(f"\nComments with exclusive words for {sentiment} sentiment:")
    for comment in comments[:5]:  # Mostrar solo los primeros 5 comentarios para cada sentimiento
        print(f"- {comment}")



# In[50]:


exclusive_words_by_sentiment


# In[33]:


from collections import Counter, defaultdict
# Combine the words and their counts across sentiments
combined_word_counts = Counter()
for words in important_words_by_sentiment.values():
    combined_word_counts.update(words)

# Find the common words across all sentiments and their counts
common_words_across_sentiments = defaultdict(dict)
for word, _ in combined_word_counts.items():
    for sentiment, words in important_words_by_sentiment.items():
        common_words_across_sentiments[word][sentiment] = words.count(word)

# Sort the words based on frequency and create the final list
important_words_summary = {}
for word, counts in common_words_across_sentiments.items():
    total_count = sum(counts.values())
    if total_count > 518:  # Change this number based on how frequently you want the words to appear to be included
        important_words_summary[word] = counts

# Sort by the total count across all sentiments
important_words_summary_sorted = sorted(
    important_words_summary.items(),
    key=lambda item: sum(item[1].values()),
    reverse=True
)

# Print the sorted important words with counts for each sentiment
for word, counts in important_words_summary_sorted:
    print(f"Word: '{word}'")
    for sentiment, count in counts.items():
        print(f"  {sentiment}: {count}")
        
    print(f"  Total count: {sum(counts.values())}")


# ## Investigación Personal

# In[19]:


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import IntegratedGradients

# Preparar una muestra del conjunto de validación para el análisis
sample_size = 10  # Ajustamos el tamaño de la muestra
sample_dataset, _ = random_split(val_dataset, [sample_size, len(val_dataset) - sample_size])
sample_loader = DataLoader(sample_dataset, batch_size=1)

# Función de forward
def forward_func(input_ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Obtener los logits del modelo
    logits = model(input_ids.long(), attention_mask=attention_mask.long())[0]

    # Seleccionar el logit de la clase con la puntuación más alta (o una clase específica)
    return torch.max(logits, dim=1).values


# Integrated Gradients
ig = IntegratedGradients(forward_func)

# Bucle para procesar el conjunto de datos de muestra
for batch in sample_loader:
    b_input_ids, b_attention_mask, _ = [t.to(device) for t in batch]

    # Baseline (usando una frase específica)
    baseline_text = "El tema es OpenAI."
    baseline_ids = tokenizer.encode(baseline_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    baseline_mask = torch.zeros_like(baseline_ids).to(device)

    # Calcular atribuciones
    attributions = ig.attribute(inputs=(b_input_ids, b_attention_mask), 
                                baselines=(baseline_ids, baseline_mask))


# In[6]:


# Carga del modelo y tokenizador
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Preparar el input
input_text = "El tema es OpenAI."
inputs = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True, padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Baseline
baseline_ids = torch.zeros_like(input_ids).to(device)
baseline_mask = torch.zeros_like(attention_mask).to(device)

# Función de forward
def forward_func(input_ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Obtener los logits del modelo
    logits = model(input_ids.long(), attention_mask=attention_mask.long())[0]

    # Seleccionar el logit de la clase con la puntuación más alta (o una clase específica)
    return torch.max(logits, dim=1).values

# Calcula las atribuciones usando IntegratedGradients
ig = IntegratedGradients(forward_func)
try:
    attributions, delta = ig.attribute((input_ids, attention_mask), 
                                       baselines=(baseline_ids, baseline_mask), 
                                       return_convergence_delta=True)
    print("Atribuciones calculadas con éxito.")
except Exception as e:
    print(f"Error al calcular atribuciones: {e}")





# In[26]:


# Ejemplo de tokenización y verificación de tipos
sample_text = "Este es un texto de ejemplo."
encoded_input = tokenizer(sample_text, return_tensors='pt')
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

print(f"Tipo de input_ids: {input_ids.dtype}")
print(f"Tipo de attention_mask: {attention_mask.dtype}")

def model_wrapper(input_ids, attention_mask=None):
    # Convertir a LongTensor si es necesario
    input_ids = input_ids if input_ids.dtype == torch.long else input_ids.long()
    if attention_mask is not None:
        attention_mask = attention_mask if attention_mask.dtype == torch.long else attention_mask.long()

    # Llamada al modelo
    return model(input_ids, attention_mask=attention_mask)



# In[27]:


from tqdm import tqdm
from captum.attr import IntegratedGradients

# Función para calcular atribuciones
def calculate_attributions(model, data_loader):
    model.eval()
    attributions = []

    for batch in tqdm(data_loader, desc="Calculating attributions"):
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

        # Verificación de tipos y formas
        print(f"Input IDs type: {b_input_ids.dtype}, shape: {b_input_ids.shape}")
        print(f"Attention Mask type: {b_input_mask.dtype}, shape: {b_input_mask.shape}")
        print(f"Labels type: {b_labels.dtype}, shape: {b_labels.shape}")

        ig = IntegratedGradients(model)

        try:
            batch_attributions = ig.attribute(b_input_ids, additional_forward_args=(None, b_input_mask), target=b_labels)
            attributions.append(batch_attributions.sum(dim=-2).detach().cpu())
        except RuntimeError as e:
            print(f"Error al calcular atribuciones: {e}")
            break

    # Concatenar todas las atribuciones si no hay errores
    if attributions:
        all_attributions = torch.cat(attributions, dim=0)
        return all_attributions
    else:
        return None

ig = IntegratedGradients(model_wrapper)
# Calcular atribuciones para todo el conjunto de validación
all_attributions = calculate_attributions(model, val_loader)

if all_attributions is not None:
    # Realizar análisis sobre all_attributions si no hay errores
    total_attributions = all_attributions.sum(dim=0)
    sorted_indices = torch.argsort(total_attributions, descending=True)
    # Mostrar los tokens más influyentes
    for idx in sorted_indices[:10]:  # Cambia 10 por el número de palabras que quieras ver
        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
        attribution = total_attributions[idx].item()
        print(f"Token: {token}, Attribution: {attribution}")
else:
    print("No se pudieron calcular las atribuciones.")





# In[29]:


import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import IntegratedGradients
from tqdm import tqdm

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo y el tokenizador preentrenados
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.to(device)
# Comentarios y etiquetas de ejemplo
comments = ["Este es un buen ejemplo.", "Este es un mal ejemplo."]
labels = [1, 0]  # Suponiendo etiquetas binarias

# Preparar los datos
encoded_inputs = tokenizer(comments, padding=True, truncation=True, max_length=128, return_tensors='pt')
input_ids = encoded_inputs['input_ids']
attention_masks = encoded_inputs['attention_mask']
labels = torch.tensor(labels)

# DataLoader
val_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
val_loader = DataLoader(val_dataset, batch_size=16)


# In[30]:


def model_wrapper(input_ids, attention_mask=None):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    return model(input_ids, attention_mask=attention_mask)

def calculate_attributions(model, data_loader):
    model.eval()
    attributions = []

    for batch in tqdm(data_loader, desc="Calculating attributions"):
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

        ig = IntegratedGradients(model_wrapper)
        try:
            batch_attributions = ig.attribute(b_input_ids, additional_forward_args=(b_input_mask), target=b_labels)
            attributions.append(batch_attributions.sum(dim=-2).detach().cpu())
        except RuntimeError as e:
            print(f"Error al calcular atribuciones: {e}")
            break

    if attributions:
        all_attributions = torch.cat(attributions, dim=0)
        return all_attributions
    else:
        return None

# Calcular atribuciones para todo el conjunto de validación
all_attributions = calculate_attributions(model, val_loader)

# Análisis de las atribuciones (si están disponibles)
if all_attributions is not None:
    total_attributions = all_attributions.sum(dim=0)
    sorted_indices = torch.argsort(total_attributions, descending=True)

    for idx in sorted_indices[:10]:  
        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
        attribution = total_attributions[idx].item()
        print(f"Token: {token}, Attribution: {attribution}")
else:
    print("No se pudieron calcular las atribuciones.")


# In[36]:


def forward_func(input_ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.long()
    attention_mask = attention_mask.long()

    # Realizar la inferencia y obtener los logits (sin torch.no_grad())
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


def calculate_attributions(model, data_loader):
    model.eval()
    attributions = []

    for batch in tqdm(data_loader, desc="Calculating attributions"):
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

        ig = IntegratedGradients(forward_func)

        try:
            # Agregar allow_unused=True
            batch_attributions = ig.attribute(b_input_ids, additional_forward_args=b_input_mask, target=b_labels, allow_unused=True)
            attributions.append(batch_attributions.sum(dim=-2).detach().cpu())

        except RuntimeError as e:
            print(f"Error al calcular atribuciones: {e}")
            break

    if attributions:
        all_attributions = torch.cat(attributions, dim=0)
        return all_attributions
    else:
        return None


    
# Obtener un solo ejemplo del conjunto de validación
single_example_loader = DataLoader(val_dataset, batch_size=1)
single_batch = next(iter(single_example_loader))
b_input_ids, b_input_mask, b_labels = [t.to(device) for t in single_batch]

# Instanciar IntegratedGradients con la función forward modificada
ig = IntegratedGradients(forward_func)

try:
    # Calcular atribuciones para un solo ejemplo
    single_attributions = ig.attribute(b_input_ids, additional_forward_args=b_input_mask, target=b_labels)
    print("Atribuciones calculadas con éxito para un solo ejemplo.")
except Exception as e:
    print(f"Error al calcular atribuciones para un solo ejemplo: {e}")




# In[8]:


from transformers import pipeline
from collections import Counter

# Conjunto de tokens a excluir
exclude_tokens = {',', '.', "'", '-', '"', '[', ']', 'º', '{', '}', '[SEP]', '[CLS]'}

def get_important_words(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
    outputs = model(input_ids, output_attentions=True)
    attentions = outputs[-1]
    last_layer_attentions = attentions[-1][0]
    mean_attentions = last_layer_attentions.mean(dim=0).squeeze()
    
    # Manejamos múltiples dimensiones si es necesario
    if mean_attentions.dim() > 1:
        mean_attentions = mean_attentions.mean(dim=-1)
    
    # Convertir los IDs de tokens a tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Crear pares de tokens y sus valores de atención, excluyendo los tokens no deseados
    token_attention_pairs = [
        (token, attention.item()) for token, attention in zip(tokens, mean_attentions)
        if token not in exclude_tokens and not token.startswith('##')
    ]
    
    # Ordenar por atención, de mayor a menor
    sorted_tokens = sorted(token_attention_pairs, key=lambda x: x[1], reverse=True)
    return sorted_tokens

important_words_by_sentiment = {
    'Negative': [],
    'Neutral': [],
    'Positive': []
}

# Iteramos sobre el dataframe y obtener las palabras importantes para cada comentario
for index, row in df.iterrows():
    sentiment = row['Sentiment']
    text = row['TextData']
    important_words = get_important_words(text, model, tokenizer)
    # Solo almacenar las palabras (tokens), no los valores de atención
    important_words_by_sentiment[sentiment].extend([word for word, _ in important_words])

    # Contamos las ocurrencias de cada palabra para cada sentimiento
word_counts = {
    sentiment: Counter(word_list)
    for sentiment, word_list in important_words_by_sentiment.items()
}
# Ordenamos las palabras basándose en la frecuencia y crear la lista final
important_words_summary = {
    sentiment: sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    for sentiment, word in word_counts.items() 
}

#Imprimimos la lista de palabras importantes por sentimiento junto al número de apariciones
for sentiment, words in important_words_summary.items():
            print(f"Palabras más importantes para los comentarios de clase {sentiment}:")
            for word, count in words:
                print(f"{word}: {count}")



# In[13]:


from transformers import pipeline
from collections import Counter
import torch

# Asegúrate de que estas líneas estén en tu código para inicializar el modelo y el tokenizer
# model = ...  # Tu modelo ya inicializado
# tokenizer = ...  # Tu tokenizer ya inicializado
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

exclude_tokens = {',', '.', "'", '-', '"', '[', ']', 'º', '?','!','{', '}', '[SEP]', '[CLS]', 's', ':'}

def get_important_words(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
    outputs = model(input_ids, output_attentions=True)
    attentions = outputs[-1]
    last_layer_attentions = attentions[-1][0]
    mean_attentions = last_layer_attentions.mean(dim=0).squeeze()

    if mean_attentions.dim() > 1:
        mean_attentions = mean_attentions.mean(dim=-1)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_attention_pairs = [
        (token, attention.item()) for token, attention in zip(tokens, mean_attentions)
        if token not in exclude_tokens and not token.startswith('##')
    ]

    sorted_tokens = sorted(token_attention_pairs, key=lambda x: x[1], reverse=True)
    return sorted_tokens

# Asegúrate de que df esté definido y tenga las columnas 'Sentiment' y 'TextData'
# df = ...  # Tu DataFrame

important_words_by_sentiment = {'Negative': [], 'Neutral': [], 'Positive': []}

# Iteración sobre el DataFrame
for index, row in df.iterrows():
    sentiment = row['Sentiment']
    text = row['TextData']
    important_words = get_important_words(text, model, tokenizer)
    important_words_by_sentiment[sentiment].extend([word for word, _ in important_words])

    # Agregamos impresión para depuración
    print(f"Palabras importantes: {important_words[:5]}")  # Imprime las primeras 5 palabras importantes

# Conteo de palabras
word_counts = {
    sentiment: Counter(word_list)
    for sentiment, word_list in important_words_by_sentiment.items()
}

# Resumen de palabras importantes
important_words_summary = {
    sentiment: sorted(word_counts[sentiment].items(), key=lambda item: item[1], reverse=True)
    for sentiment in word_counts
}

# Impresión de los resultados
for sentiment, words in important_words_summary.items():
    print(f"Palabras más importantes para los comentarios de clase {sentiment}:")
    for word, count in words[:10]:  # Imprime las 10 palabras más frecuentes
        print(f"{word}: {count}")


# In[8]:


# Contar los comentarios y las palabras importantes por usuario y sentimiento
user_sentiment_counts = df.groupby('User')['Sentiment'].value_counts().unstack(fill_value=0)
top_users = user_sentiment_counts.sum(axis=1).nlargest(10).index
top_users_data = df[df['User'].isin(top_users)]

# Diccionario para almacenar las palabras importantes por usuario y sentimiento
top_user_important_words_by_sentiment = {
    user: {sentiment: Counter() for sentiment in ['Negative', 'Neutral', 'Positive']}
    for user in top_users
}

# Procesar los comentarios de los usuarios principales
for _, row in top_users_data.iterrows():
    user = row['User']
    sentiment = row['Sentiment']
    text = row['TextData']
    important_words = get_important_words(text, model, tokenizer)

    # Actualizar el contador de palabras importantes para este usuario y sentimiento
    for word in important_words:
        if word not in exclude_tokens:
            top_user_important_words_by_sentiment[user][sentiment][word] += 1

# Imprimir los resultados
for user in top_users:
    print(f"Usuario: {user}")
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        print(f"  Sentimiento {sentiment}:")
        print(f"    Total de comentarios: {user_sentiment_counts.loc[user, sentiment]}")
        print(f"    Palabras importantes: {top_user_important_words_by_sentiment[user][sentiment].most_common(10)}")
    print()


# 

# In[ ]:




