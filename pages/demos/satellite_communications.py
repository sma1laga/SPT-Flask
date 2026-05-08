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