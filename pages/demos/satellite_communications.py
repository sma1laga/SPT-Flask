from flask import Blueprint, render_template

satellite_communications_bp = Blueprint(
    "satellite_communications", __name__, template_folder="../../templates"
)

SATELLITE_COMMUNICATIONS_DEMOS = {
    "Satellite Communications": {
        "Lecture": [
            {
                "slug": "geo-elevation-visibility-demo",
                "title": "GEO Elevation Visibility",
                "title_desc": "GEO Elevation Visibility Demo",
                "desc": "GEO satellite visibility and elevation angle over latitude with line-of-sight geometry.",
                "endpoint": "satellite_communications.geo_elevation_visibility_demo",
            },
            {
                "slug": "kepler-first-law-demo",
                "title": "Kepler's First Law",
                "title_desc": "Kepler's First Law Demo",
                "desc": "Satellite motion on an ellipse with Earth at one focus and varying eccentricity.",
                "endpoint": "satellite_communications.kepler_first_law_demo",
            },
            {
                "slug": "kepler-second-law-demo",
                "title": "Kepler's Second Law",
                "title_desc": "Kepler's Second Law Demo",
                "desc": "Equal areas are swept in equal times with faster motion near perigee and slower near apogee.",
                "endpoint": "satellite_communications.kepler_second_law_demo",
            },
            {
                "slug": "kepler-third-law-demo",
                "title": "Kepler's Third Law",
                "title_desc": "Kepler's Third Law Demo",
                "desc": "Compare orbital periods for LEO, MEO, GEO, and a custom altitude to see T ∝ a^(3/2).",
                "endpoint": "satellite_communications.kepler_third_law_demo",
            },
            {
                "slug": "molniya-far-half-time-demo",
                "title": "Molniya Far-Half Time",
                "title_desc": "Molniya Far-Half Time Demo",
                "desc": "Visualize why the satellite spends more than half of its period near apogee in the far half of an eccentric orbit.",
                "endpoint": "satellite_communications.molniya_far_half_time_demo",
            },
            {
                "slug": "visibility-window-demo",
                "title": "Visibility Window",
                "title_desc": "Visibility Window Demo",
                "desc": "Animate the relative-longitude interval where a satellite remains visible above a minimum elevation angle.",
                "endpoint": "satellite_communications.visibility_window_demo",
            },
            {
                "slug": "relative-motion-demo",
                "title": "Relative Motion",
                "title_desc": "Relative Motion Demo",
                "desc": "Show how pass repeat times are governed by the satellite-ground-station relative angular rate.",
                "endpoint": "satellite_communications.relative_motion_demo",
            },
            {
                "slug": "satellite-orbit-ground-track-demo",
                "title": "Satellite Orbit and Ground Track",
                "title_desc": "Satellite Orbit and Ground Track Demo",
                "desc": "Interactive sub-satellite latitude and longitude visualization with inertial orbit and Earth-fixed ground track views.",
                "endpoint": "satellite_communications.satellite_orbit_ground_track_demo",
            },
            {
                "slug": "solar-day-vs-sidereal-day-demo",
                "title": "Solar Day vs Sidereal Day",
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

@satellite_communications_bp.route("/visibility-window-demo", methods=["GET"])
def visibility_window_demo():
    return render_template("demos/visibility_window_demo.html")


@satellite_communications_bp.route("/relative-motion-demo", methods=["GET"])
def relative_motion_demo():
    return render_template("demos/relative_motion_demo.html")



@satellite_communications_bp.route("/solar-day-vs-sidereal-day-demo", methods=["GET"])
def solar_day_vs_sidereal_day_demo():
    return render_template("demos/solar_day_vs_sidereal_day_demo.html")


@satellite_communications_bp.route("/satellite-orbit-ground-track-demo", methods=["GET"])
def satellite_orbit_ground_track_demo():
    return render_template("demos/satellite_orbit_ground_track_demo.html")