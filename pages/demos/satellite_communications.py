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
            },
            {
                "slug": "kepler-third-law-demo",
                "title": "Demo 4",
                "title_desc": "Kepler's Third Law Demo",
                "desc": "Compare orbital periods for LEO, MEO, GEO, and a custom altitude to see T ∝ a^(3/2).",
                "endpoint": "satellite_communications.kepler_third_law_demo",
            },
            {
                "slug": "molniya-far-half-time-demo",
                "title": "Demo 5",
                "title_desc": "Molniya Far-Half Time Demo",
                "desc": "Visualize why the satellite spends more than half of its period near apogee in the far half of an eccentric orbit.",
                "endpoint": "satellite_communications.molniya_far_half_time_demo",
            },
            {
                "slug": "solar-day-vs-sidereal-day-demo",
                "title": "Demo 6",
                "title_desc": "Solar Day vs Sidereal Day Demo",
                "desc": "Compare Earth's rotation relative to distant stars versus the Sun to understand the ~4 minute difference.",
                "endpoint": "satellite_communications.solar_day_vs_sidereal_day_demo",
            },
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


@satellite_communications_bp.route("/kepler-third-law-demo", methods=["GET"])
def kepler_third_law_demo():
    return render_template("demos/kepler_third_law_demo.html")

@satellite_communications_bp.route("/molniya-far-half-time-demo", methods=["GET"])
def molniya_far_half_time_demo():
    return render_template("demos/molniya_far_half_time_demo.html")

@satellite_communications_bp.route("/solar-day-vs-sidereal-day-demo", methods=["GET"])
def solar_day_vs_sidereal_day_demo():
    return render_template("demos/solar_day_vs_sidereal_day_demo.html")