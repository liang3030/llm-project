### Use virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### Install dependencies

```bash
pip install -r requirements.txt
```
### Add dependencies and update requirements.txt

```bash
pip install <package_name>
pip freeze > requirements.txt
```


### API Usage with cur

1. store and process a PDF file
```bash
curl -X POST "http://localhost:8000/pdf-search/process-pdf" \
  -F "file=@test.pdf" \
  -F "chunking_method=sentences"
```

1. search in the processed PDF file
```bash
curl -X POST "http://localhost:8000/pdf-search/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what does this file about?",
    "n_results": 1
  }'
```