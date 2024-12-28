# Example of running the backend for TTS server on python.
**Important: need to use python 3.11.**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py
```