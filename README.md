# SOBES

1. docker compose up --build


2. curl -X 'POST' \
  'http://localhost:8000/detect/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@TEST.JPG'
