import os
import uuid
from werkzeug.utils import secure_filename
from fog_detection import analyze_visibility



from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import uuid

from werkzeug.utils import secure_filename
from fog_detection import analyze_visibility

app = Flask(__name__)

# -------------------------------
# MODEL CONFIGURATION
# -------------------------------

MODEL_PATH = os.path.join(
    app.root_path,
    "accident_severity_model.pkl"
)

model = joblib.load(MODEL_PATH)

# -------------------------------
# IMAGE UPLOAD CONFIGURATION
# -------------------------------

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_image(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# -------------------------------
# LOW-VISIBILITY RISK ASSESSMENT
# -------------------------------

def assess_visibility_risk(
    visibility,
    weather,
    lighting,
    road_surface
):
    score = 0
    reasons = []

    # Reported visibility
    if visibility == "Very low":
        score += 3
        reasons.append("Very low reported visibility")
    elif visibility == "Low":
        score += 2
        reasons.append("Low reported visibility")

    # Weather
    if weather in ["Fog or mist", "Fog"]:
        score += 3
        reasons.append("Fog or mist reported")
    elif weather in ["Raining", "Raining and Windy"]:
        score += 2
        reasons.append("Rain may reduce visibility and grip")

    # Lighting
    if lighting == "Darkness - no lighting":
        score += 2
        reasons.append("Road has no lighting")
    elif lighting == "Darkness - lights lit":
        score += 1
        reasons.append("Night-time lighting conditions")

    # Road surface
    if road_surface in ["Wet or damp", "Wet or damp road"]:
        score += 1
        reasons.append("Wet road surface")

    # Risk category and recommendations
    if score >= 6:
        level = "HIGH"
        advice = [
            "Reduce speed and increase following distance.",
            "Use appropriate vehicle lights.",
            "Avoid unnecessary travel if conditions are unsafe.",
            "Follow official road closures and traffic instructions."
        ]

    elif score >= 3:
        level = "MODERATE"
        advice = [
            "Reduce speed and maintain extra following distance.",
            "Use appropriate lights and remain alert.",
            "Avoid sudden braking and overtaking."
        ]

    else:
        level = "LOW"
        advice = [
            "Follow posted speed limits.",
            "Maintain a safe following distance.",
            "Remain alert to changing road conditions."
        ]

    return level, reasons, advice


# -------------------------------
# MAIN ROUTE
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    risk_level = None
    risk_reasons = []
    recommendations = []
    fog_result = None

    if request.method == "POST":
        image_file = request.files.get("road_image")

        # FORM 1: Image analysis
        if image_file and image_file.filename:
            image_path = None

            try:
                if not allowed_image(image_file.filename):
                    raise ValueError(
                        "Please upload a PNG, JPG, or JPEG image."
                    )

                original_name = secure_filename(
                    image_file.filename
                )
                extension = original_name.rsplit(".", 1)[1].lower()
                saved_name = f"{uuid.uuid4().hex}.{extension}"

                image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    saved_name
                )
                image_file.save(image_path)

                fog_result = analyze_visibility(image_path)
                fog_result["image_url"] = saved_name

            except Exception as e:
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)

                app.logger.exception("Image analysis failed")
                error = str(e)

        # FORM 2: Accident severity assessment
        else:
            try:
                weather = request.form.get("weather")
                road_surface = request.form.get("road_surface")
                lighting = request.form.get("light_conditions")
                visibility = request.form.get("visibility")
                vehicles_text = request.form.get("vehicles")

                if not all([
                    weather, road_surface, lighting,
                    visibility, vehicles_text
                ]):
                    raise ValueError(
                        "Please complete all accident assessment fields."
                    )

                vehicles = int(vehicles_text)

                if not 1 <= vehicles <= 20:
                    raise ValueError(
                        "Number of vehicles must be between 1 and 20."
                    )

                input_data = {
                    column: np.nan
                    for column in model.feature_names_in_
                }

                input_data["Weather_conditions"] = weather
                input_data["Road_surface_conditions"] = road_surface
                input_data["Light_conditions"] = lighting
                input_data["Number_of_vehicles_involved"] = vehicles

                input_df = pd.DataFrame(
                    [input_data],
                    columns=model.feature_names_in_
                )

                prediction = str(model.predict(input_df)[0])

                (
                    risk_level,
                    risk_reasons,
                    recommendations
                ) = assess_visibility_risk(
                    visibility,
                    weather,
                    lighting,
                    road_surface
                )

            except Exception as e:
                app.logger.exception("Severity assessment failed")
                error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        recommendations=recommendations,
        fog_result=fog_result
    )




# -------------------------------
# APPLICATION ENTRY POINT
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
