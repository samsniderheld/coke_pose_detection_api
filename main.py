import io

from flask import Flask, request, jsonify
from PIL import Image

from pose_detectors.metrabs_detector import MetrabsDetector

app = Flask(__name__)

pose_detector = MetrabsDetector()

@app.route("/generate_pose/", methods=["POST"])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        image = Image.open(io.BytesIO(file.read()))
        pose_data = pose_detector.detect_poses(image)
        
        # If return_type is JSON, return metadata
        return jsonify(pose_detector.create_json_response(pose_data))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
