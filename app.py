from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from omr.service import process_request
from audio_converter.audio_service import analyze_audio_request

app = Flask(__name__)
swagger = Swagger(app)

 

   

@app.route('/api/processar-omr', methods=['POST'])
@swag_from('omr/docs/omr_process.yml')
def upload_file():
    """Processar imagem OMR"""
    result, status = process_request(
        file_storage=request.files.get('file'),
        gabarito_json_str=request.form.get('gabarito')
    )
    return jsonify(result), status

@app.route('/api/analisar-audio', methods=['POST'])
@swag_from('omr/docs/audio_analyze.yml')
def analyze_audio():
    """Analisar leitura do aluno via áudio"""
    result, status = analyze_audio_request(
        file_storage=request.files.get('audio'),
        reference_text=request.form.get('texto')
    )
    print(result)
    return jsonify(result), status

@app.route('/')
@swag_from('omr/docs/health.yml')
def index():
    return "API de Processamento OMR - Envie uma imagem para /api/processar-omr (docs em /apidocs)"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
