python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py