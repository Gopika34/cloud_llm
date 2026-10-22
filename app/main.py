from fastapi import FastAPI
 
 
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