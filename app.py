from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import CalorieData, CalorieDataRegressor
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the calorie-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Sex: Optional[int] = None
        self.Age: Optional[int] = None
        self.Height: Optional[float] = None
        self.Weight: Optional[float] = None
        self.Duration: Optional[float] = None
        self.Heart_Rate: Optional[float] = None
        self.Body_Temp: Optional[float] = None

    async def get_calorie_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Sex = form.get("Sex") # type: ignore[assignment]
        self.Age = form.get("Age") # type: ignore[assignment]
        self.Height = form.get("Height") # type: ignore[assignment]
        self.Weight = form.get("Weight")   # type: ignore[assignment]
        self.Duration = form.get("Duration") # type: ignore[assignment]
        self.Heart_Rate = form.get("Heart_Rate") # type: ignore[assignment]
        self.Body_Temp = form.get("Body_Temp")  # type: ignore[assignment]

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for calorie data input.
    """
    return templates.TemplateResponse(
            "index.html", {"request": request, "context": None})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient(request: Request):
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        # Redirect to home with a success message
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": "Training successful!"}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": f"Error Occurred! {e}"}
        )

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_calorie_data()
        
        calorie_data = CalorieData(
            Sex=form.Sex,
            Age=form.Age,
            Height=form.Height,
            Weight=form.Weight,
            Duration=form.Duration,
            Heart_Rate=form.Heart_Rate,
            Body_Temp=form.Body_Temp
        )

        # Convert form data into a DataFrame for the model
        calorie_df = calorie_data.get_calorie_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = CalorieDataRegressor()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=calorie_df)[0]

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": value},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)