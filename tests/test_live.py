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
        print("Accuracy %:", res.get("accuracy_percent"))
        print("Damage %:", res.get("damage_percentage"))
        print("Defect boxes:", len(res.get("defect_boxes", [])))
        print("Grad-CAM included:", bool(res.get("gradcam")))

    # 3. Camera sample & scan frame
    with urllib.request.urlopen('http://127.0.0.1:5000/api/camera/samples') as r:
        samples = json.loads(r.read())
    print("Camera samples available:", len(samples))

    cam_req = urllib.request.Request(
        'http://127.0.0.1:5000/api/camera/scan-frame',
        data=json.dumps({"image": samples[0]["data_uri"]}).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(cam_req) as r:
        cam_res = json.loads(r.read())
        print("Live camera scan status:", r.status)
        print("Live detected defect:", cam_res["predicted_class"], f"({cam_res.get('defect_label')})")
        print("Live damage accuracy:", cam_res.get("accuracy_percent"), "%")
        print("Live defect boxes:", len(cam_res.get("defect_boxes", [])))

if __name__ == '__main__':
    run_checks()
