from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import numpy as np
import pandas as pd
from model import TreeHealthModel
from features import Features

# Загружаем модель и LabelEncoder'ы
model = None
label_encoder = None
label_encoders = {}
cat_cardinalities = []
cat_features = []
num_features = []

def load_model_and_encoders():
    global model, label_encoder, label_encoders, cat_cardinalities, cat_features, num_features

    # Загружаем информацию о признаках
    features = Features()
    cat_features = features.get_cat_features()  # Загружаем список категориальных признаков
    num_features = features.get_num_features()  # Загружаем список числовых признаков


    # Загружаем модель
    model = TreeHealthModel(
        num_numeric=1,  # Укажите правильное количество числовых признаков
        cat_cardinalities=joblib.load('cat_cardinalities.pkl'),  # Загружаем cat_cardinalities
        embedding_dim=20,
        hidden_dim=256
    )
    model.load_state_dict(torch.load('tree_health_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # Загружаем LabelEncoder для целевой переменной
    label_encoder = joblib.load('label_encoder_health.pkl')

    # Загружаем LabelEncoder'ы для категориальных признаков
    for col in cat_features:
        label_encoders[col] = joblib.load(f'label_encoder_{col}.pkl')

# Определяем входные данные
class InputData(BaseModel):
    tree_dbh: int
    spc_common: object  
    spc_latin: object
    postcode: int
    borough: object
    zip_city: object  
    steward: object 
    guards: object  
    sidewalk: object  
    user_type: object  
    root_stone: object
    root_grate: object  
    root_other: object
    trunk_wire: object  
    trnk_light: object  
    trnk_other: object  
    brch_light: object  
    brch_shoe: object
    brch_other: object
    curb_loc: object

# Создаем FastAPI приложение
app = FastAPI()

# Загружаем модель и LabelEncoder'ы при старте приложения
@app.on_event("startup")
async def startup_event():
    load_model_and_encoders()

# Эндпоинт для предсказания
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Преобразуем входные данные в DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])


        # Преобразуем категориальные признаки с помощью LabelEncoder'ов
        for col in cat_features:
            input_df[col] = input_df[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
            input_df[col] = label_encoders[col].transform(input_df[col])

        
        print(input_df)
        print(num_features)

        # Преобразуем данные в тензоры
        X_num = torch.tensor(input_df[num_features].values, dtype=torch.float32)
        X_cat = torch.tensor(input_df[cat_features].values, dtype=torch.long)
        
        print(X_num)
        print(X_cat) 

        # Получаем предсказание
        with torch.no_grad():
            outputs = model(X_num, X_cat)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.cpu().numpy()[0]

        print('kek') 

        # Декодируем предсказание
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        return {"prediction": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)