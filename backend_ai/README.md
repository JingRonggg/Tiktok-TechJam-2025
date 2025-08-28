# Setup Instructions

Follow these steps to set up and run the backend AI service:

## 1. Create a Virtual Environment (Recommended)
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

## 3. Run the Application
Start the backend service:
```bash
uvicorn main:app --reload
```

## 4. Additional Notes
- Ensure you are using Python 3.8 or higher.
- For development, you may need to install additional dependencies as specified in `requirements.txt`.
- Refer to the code and comments for further customization or configuration.
