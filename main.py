from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from similarityChecker import search_similar_by_image, embeddings, image_paths
from fastapi import Request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path() / "uploads"
DATASET_DIR = "./nail_dataset/"
os.makedirs(UPLOAD_DIR, exist_ok=True)


app.mount("/nail_dataset", StaticFiles(directory=DATASET_DIR), name="nail_dataset")


@app.post("/search/")
async def search_similar_nails(request: Request, file: UploadFile):

    try:
        temp_filename = UPLOAD_DIR / file.filename
        data = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(data)

        results = search_similar_by_image(temp_filename, embeddings, image_paths, 10)
        url = request.base_url
        os.remove(temp_filename)

        enhanced_results = [
            {
                "image_url": str(url) + r["image_path"],
                "similarity": r["similarity"],
            }
            for r in results
        ]

        return JSONResponse(
            content={
                "uploaded_file": file.filename,
                "total_results": len(enhanced_results),
                "results": enhanced_results,
            }
        )

    except Exception as e:
        print("Theres a prolem")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Nail Image Similarity Search API",
        "total_images_in_dataset": len(image_paths),
        "endpoint": "POST /search/ - Upload image to find similar nails",
        "static_files": "GET /nail_dataset/<filename> to view any nail image",
    }
