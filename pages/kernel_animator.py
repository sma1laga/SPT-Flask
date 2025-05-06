from flask import Blueprint, render_template, url_for

bp = Blueprint(
    'kernel_animator',           # blueprint name
    __name__,                    # module name
    url_prefix='/kernel'         # mounted at /kernel/
)

@bp.route('/', methods=['GET'])
def show_animator():
    # now pointing at static/images/rocky.png
    img_url = url_for('static', filename='images/rocky.png')
    return render_template('kernel.html', demo_img=img_url)
