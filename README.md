# House Price Prediction

A full-stack machine learning web application that estimates a property's price from its size, number of bedrooms and bathrooms, location, and year built. Users can compare predictions from Linear Regression and Random Forest models through a responsive web interface.

[View the live application](https://house-price-prediction-5apgzghfv-isfahan.vercel.app/)

> This project is intended for learning and demonstration. Its estimates should not be used as professional real-estate valuations or financial advice.

## Features

- Predict a house price using five property characteristics
- Choose between Linear Regression and Random Forest
- Responsive interface with model-performance information
- REST API built with Flask
- Data cleaning, feature engineering, and model-training scripts
- Separate frontend and backend deployments

## Model performance

| Model | R² score | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.848 | $63,086 | $75,624 |
| Random Forest | 0.859 | $52,524 | $72,686 |

Based on these evaluation results, Random Forest is the stronger of the two included models.

## Tech stack

**Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS, Radix UI, Recharts

**Backend:** Python, Flask, Flask-CORS, Gunicorn

**Machine learning:** pandas, NumPy, scikit-learn, Joblib

**Deployment:** Vercel (frontend) and Railway (backend)

## Project structure

```text
house_price_prediction/
├── client/                 # Next.js frontend
│   ├── public/
│   └── src/
│       ├── app/
│       ├── components/
│       └── lib/
├── code/                   # Flask API and ML scripts
│   ├── app.py
│   ├── model.py
│   ├── preprocessing.py
│   └── utils.py
├── dataset/                # Raw and processed datasets
├── models/                 # Trained models and preprocessing artifacts
├── Procfile                # Railway/Gunicorn start command
└── requirements.txt        # Python dependencies
```

## Run locally

### Prerequisites

- Python 3.10 or later
- Node.js 20 or later
- npm

### 1. Clone the repository

```bash
git clone https://github.com/SaabaMire/house_price_prediction.git
cd house_price_prediction
```

### 2. Start the Flask backend

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the Python packages and start the API:

```bash
pip install -r requirements.txt
python code/app.py
```

The backend will run at `http://127.0.0.1:8080`.

### 3. Start the Next.js frontend

Open another terminal from the repository root:

```bash
cd client
npm install
```

Create `client/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8080
```

Then start the development server:

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

## API

### Health and API information

```http
GET /
```

### Predict a house price

```http
POST /predict?model=rf
Content-Type: application/json
```

Available model values are `rf` for Random Forest and `lr` for Linear Regression.

Example request body:

```json
{
  "Size_sqft": 2400,
  "Bedrooms": 3,
  "Bathrooms": 2,
  "YearBuilt": 2010,
  "Location": "City"
}
```

Example response:

```json
{
  "model": "random_forest",
  "input": {
    "Size_sqft": 2400,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "YearBuilt": 2010,
    "Location": "City"
  },
  "prediction": 679659.0
}
```

`Location` must be `City`, `Suburb`, or `Rural`.

## Deployment configuration

### Frontend on Vercel

- Root Directory: `client`
- Framework Preset: Next.js
- Environment variable: `NEXT_PUBLIC_API_URL=https://your-railway-domain`

### Backend on Railway

Railway uses the root-level `Procfile`:

```text
web: PYTHONPATH=code gunicorn --bind 0.0.0.0:$PORT app:app
```

## Author

[SaabaMire](https://github.com/SaabaMire)
