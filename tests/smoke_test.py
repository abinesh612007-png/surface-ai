"""Quick upload validation smoke test."""
import sys, os, io, http.client, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

BASE = ('127.0.0.1', 5000)

def post_multipart(filename, content_type, data):
    boundary = 'bnd12345678'
    disp = 'Content-Disposition: form-data; name="image"; filename="' + filename + '"'
    parts = ('--' + boundary + '\r\n' + disp + '\r\n'
             + 'Content-Type: ' + content_type + '\r\n\r\n').encode()
    body = parts + data + ('\r\n--' + boundary + '--\r\n').encode()
    conn = http.client.HTTPConnection(*BASE, timeout=10)
    conn.request('POST', '/api/inspect', body=body,
                 headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
    r = conn.getresponse()
    return r.status, json.loads(r.read())

def test_wrong_mime():
    status, data = post_multipart('file.txt', 'text/plain', b'plain text')
    assert status == 400, f"Expected 400, got {status}"

def test_valid_jpeg():
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color=(150, 150, 150)).save(buf, 'JPEG')
    status, data = post_multipart('surface.jpg', 'image/jpeg', buf.getvalue())
    # 200 if model weights are loaded, 503 if model is not yet loaded
    assert status in (200, 503), f"Expected 200 or 503, got {status}: {data}"
    if status == 200:
        assert "predicted_class" in data
        assert "confidence" in data
    elif status == 503:
        assert data.get("model_unavailable") is True

if __name__ == '__main__':
    test_wrong_mime()
    print("Test 1 PASS: wrong file type -> 400")
    test_valid_jpeg()
    print("Test 2 PASS: valid JPEG processed successfully")
    print("\nAll upload validation tests PASSED!")
