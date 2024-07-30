from flask import Flask, request, jsonify, abort
import pandas as pd
import pickle

filename = 'data/modelo_main.pkl'
with open(filename, 'rb') as file:
    loaded_model_data = pickle.load(file)

loaded_model = loaded_model_data['model']
X_train = loaded_model_data['X_train']
X_test = loaded_model_data['X_test']
y_train = loaded_model_data['y_train']
y_test = loaded_model_data['y_test']
data = loaded_model_data['data']


app = Flask(__name__)
app.config["DEBUG"] = False

@app.route('/', methods=['GET'])
def home():
    mensaje_de_bienvenida = '<h1>Proyecto Industrialización</h1>'
    return mensaje_de_bienvenida

@app.route('/reviews/all', methods=['GET']) # Mostramos todas las reviews
def todos_los_datos():
    all_data = jsonify(data.to_dict(orient='records'))

    return all_data

@app.route('/reviews/train', methods=['GET']) # Mostramos solo el train
def train():
    train = pd.concat([X_train, y_train], axis=1)
    train_data = jsonify(train.to_dict(orient='records'))

    return train_data

@app.route('/reviews/test', methods=['GET']) # Mostramos solo el test
def test():
    test = pd.concat([X_test, y_test], axis=1)
    test_data = jsonify(test.to_dict(orient='records'))

    return test_data

@app.route('/reviews/observation/<int:n>', methods=['GET']) # Mostramos la observacion numero n de nuestro dataset
def n_observacion(n):
    if (n < 0) or (n >= len(data)):
        abort(404, description=f"Index {n} out of range. Valid range is 0 to {len(data) - 1}.")
    
    resultado = jsonify(data.iloc[n].to_dict())

    return resultado

# Manejar el error 404 de forma personalizada
@app.errorhandler(404)
def resource_not_found(e):
    response = jsonify({'error': str(e)})
    response.status_code = 404
    return response

@app.route('/reviews/query', methods=['GET'])
def query_filter():
    posible_columns = data.columns

    query_params = {}  # Para que pueda consultar las columnas que quiera
    for col in posible_columns:
        if col in request.args:
            query_params[col] = request.args[col]

    # Control de errores
    if not query_params:
        return jsonify({"error": "No valid query parameters provided"}), 400

    for key in query_params:
        try:
            query_params[key] = round(float(query_params[key]), 6)  # Redondear a 6 decimales
        except ValueError:
            return jsonify({"error": f"Invalid value for {key}: must be numeric"}), 400

    # Redondear los valores del DataFrame a 6 decimales
    filtered_df = data.copy()
    for col in posible_columns:
        if col in data.columns and pd.api.types.is_float_dtype(data[col]):
            filtered_df[col] = filtered_df[col].apply(lambda x: round(x, 6))

    # Filtrar el DataFrame por los parámetros de consulta
    for key, value in query_params.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    if len(filtered_df) == 0:
        return '<h1>No se encontraron resultados para esta búsqueda</h1>'

    result = jsonify(filtered_df.to_dict(orient='records'))
    return result

@app.route('/reviews/predictions')
def predictions():
    y_pred_test = loaded_model.predict(X_test).tolist()
    y_pred_train = loaded_model.predict(X_train).tolist()

    return jsonify({
        'predicciones_test': y_pred_test,
        'predicciones_train': y_pred_train
    })

app.run(port=8000)