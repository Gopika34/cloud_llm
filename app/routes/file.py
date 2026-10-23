from fastapi import APIRouter, UploadFile

router = APIRouter()

@router.post("/upload")
async def upload_files(file: UploadFile):
    load_content = await file.read()
    
    content=[]
    
    for text in load_content.splitlines():
        content.append(text)
        
    return content