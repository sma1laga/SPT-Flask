from flask import Blueprint, render_template

satellite_communications_bp = Blueprint(
    "satellite_communications", __name__, template_folder="../../templates"
)

SATELLITE_COMMUNICATIONS_DEMOS = {
    "Satellite Communications": {
        "Lecture": [
            {
                "slug": "geo-elevation-visibility-demo",
                "title": "Demo 1",
                "title_desc": "GEO Elevation Visibility Demo",
                "desc": "GEO satellite visibility and elevation angle over latitude with line-of-sight geometry.",
                "endpoint": "satellite_communications.geo_elevation_visibility_demo",
            },
            {
                "slug": "kepler-first-law-demo",
                "title": "Demo 2",
                "title_desc": "Kepler's First Law Demo",
                "desc": "Satellite motion on an ellipse with Earth at one focus and varying eccentricity.",
                "endpoint": "satellite_communications.kepler_first_law_demo",
            },
            {
                "slug": "kepler-second-law-demo",
                "title": "Demo 3",
                "title_desc": "Kepler's Second Law Demo",
                "desc": "Equal areas are swept in equal times with faster motion near perigee and slower near apogee.",
                "endpoint": "satellite_communications.kepler_second_law_demo",
            }
        ],
        "Tutorial": [],
    }
}


@satellite_communications_bp.route("/", methods=["GET"])
def page():
    return render_template("demos/menu.html", demos=SATELLITE_COMMUNICATIONS_DEMOS)


@satellite_communications_bp.route("/geo-elevation-visibility-demo", methods=["GET"])
def geo_elevation_visibility_demo():
    return render_template("demos/geo_elevation_visibility_demo.html")

@satellite_communications_bp.route("/kepler-first-law-demo", methods=["GET"])
def kepler_first_law_demo():
    return render_template("demos/kepler_first_law_demo.html")

@satellite_communications_bp.route("/kepler-second-law-demo", methods=["GET"])
def kepler_second_law_demo():
    return render_template("demos/kepler_second_law_demo.html")