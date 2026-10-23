from fastapi import FastAPI
from app.routes import file
 
app = FastAPI()

@app.get('/')
def landing_page():
    return {
        "msg" : "This is our backend"
    }
    
@app.get('/health')
def health_page():
    return {
        "msg" : "okay done!"
    }
    
app.include_router(file.router)