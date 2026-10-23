import urllib.request, json, io
from PIL import Image

def run_checks():
    # 1. Model status
    with urllib.request.urlopen('http://127.0.0.1:5000/api/model-status') as r:
        status = json.loads(r.read())
    print("Model status loaded:", status["loaded"])
    print("Device:", status["device"])

    # 2. Upload test image
    buf = io.BytesIO()
    Image.new('RGB', (200, 200), color=(128, 128, 128)).save(buf, 'JPEG')
    data = buf.getvalue()

    boundary = '----TestBoundary123'
    header = (
        '--' + boundary + '\r\n'
        'Content-Disposition: form-data; name="image"; filename="surface_test.jpg"\r\n'
        'Content-Type: image/jpeg\r\n\r\n'
    ).encode()
    body = header + data + ('\r\n--' + boundary + '--\r\n').encode()

    req = urllib.request.Request(
        'http://127.0.0.1:5000/api/inspect',
        data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary},
        method='POST'
    )

    with urllib.request.urlopen(req) as r:
        res = json.loads(r.read())
        print("Inspection response status:", r.status)
        print("Predicted class:", res["predicted_class"])
        print("Confidence:", res["confidence"])
        print("Top 3:", res["top3"])
        print("Grad-CAM included:", bool(res.get("gradcam")))

if __name__ == '__main__':
    run_checks()
