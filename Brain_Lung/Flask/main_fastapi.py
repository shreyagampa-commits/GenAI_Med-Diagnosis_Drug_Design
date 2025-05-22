

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )
from imports_fast import * 

import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER1 = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER1 = os.path.join(BASE_DIR, "../frontend/public")
MODEL_PATHS = {
    "brain": os.path.join(BASE_DIR, "models", "brainmodel.keras"),
    "lung": os.path.join(BASE_DIR, "models", "lungmodel.keras")
}

os.makedirs(UPLOAD_FOLDER1, exist_ok=True)
os.makedirs(OUTPUT_FOLDER1, exist_ok=True)

# Load model function
def load_model(model_type: str):
    model_path = MODEL_PATHS.get(model_type)
    if not model_path or not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"

    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"
BASE_DIR = Path(__file__).resolve().parent

# Define upload and processing directories
UPLOAD_FOLDER = BASE_DIR / "temp_nii_uploads"
SLICES_FOLDER = BASE_DIR / "temp_nii_slices"
OUTPUT_MASKS_FOLDER = BASE_DIR / "temp_nii_masks"
PUBLIC_FOLDER = BASE_DIR / "static"  # Define the public folder in the current directory
ORIGINAL_NII_PUBLIC_FOLDER = PUBLIC_FOLDER / "original_nii" #create original and predicted nii folders
PREDICTED_NII_PUBLIC_FOLDER = PUBLIC_FOLDER / "predicted_nii"


# Ensure output directories exist
# ORIGINAL_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True) # changed
# PREDICTED_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True) # changed
PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True)
ORIGINAL_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True)
PREDICTED_NII_PUBLIC_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SLICES_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS_FOLDER.mkdir(parents=True, exist_ok=True)

# Configuration dictionary
cf = {
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "flattened_patch_dim": 16 * 16 * 3,
}

# Declare global variables

filepath1 = None  # Global variable for original file path
filepath2 = None  # Global variable for predicted file path
async def process_nii(file: UploadFile, model):
    """Full pipeline: slices -> predict masks -> rebuild NIfTI, with output in current directory."""
    
    input_filename = f"uploaded_{uuid.uuid4().hex}_{file.filename}"
    input_nii_path = UPLOAD_FOLDER / input_filename
    predicted_nii_filename = f"predicted_{uuid.uuid4().hex}_{file.filename}"
    predicted_nii_path_server = PREDICTED_NII_PUBLIC_FOLDER / predicted_nii_filename  # Changed
    original_filename_moved = f"original_{uuid.uuid4().hex}_{file.filename}"
    original_file_path_moved = ORIGINAL_NII_PUBLIC_FOLDER / original_filename_moved

    # Clean previous slices and masks for this processing
    shutil.rmtree(SLICES_FOLDER, ignore_errors=True)
    SLICES_FOLDER.mkdir(exist_ok=True)
    shutil.rmtree(OUTPUT_MASKS_FOLDER, ignore_errors=True)
    OUTPUT_MASKS_FOLDER.mkdir(exist_ok=True)

    try:
        # Save the uploaded file temporarily
        with open(input_nii_path, "wb") as f:
            while contents := await file.read(1024 * 1024):
                f.write(contents)

        nii_img = nib.load(input_nii_path)
        nii_data = nii_img.get_fdata()
        depth = nii_data.shape[0]  # slices along axis 0

        # Step 1: Slice the NIfTI
        for i in range(depth):
            slice_img = nii_data[i, :, :]
            slice_img = np.clip(slice_img, 0, 255).astype(np.uint8)
            slice_resized = cv2.resize(slice_img, (512, 512), interpolation=cv2.INTER_AREA)
            slice_rgb = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2BGR)
            slice_path = SLICES_FOLDER / f"slice_{i:03d}.png"
            cv2.imwrite(str(slice_path), slice_rgb)

        # Step 2: Predict masks
        image_files = sorted(os.listdir(SLICES_FOLDER))
        predicted_mask_paths = []

        for image_name in image_files:
            input_image_path = SLICES_FOLDER / image_name
            image = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            resized = cv2.resize(image, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_LANCZOS4)
            norm = resized / 255.0
            patches = patchify(norm, (cf["patch_size"], cf["patch_size"], cf["num_channels"]), cf["patch_size"])
            patches = patches.reshape(-1, cf["flattened_patch_dim"])
            patches = np.expand_dims(patches, axis=0)
            pred = model.predict(patches, verbose=0)[0]
            pred = (pred * 255).astype(np.uint8)
            if len(pred.shape) == 3 and pred.shape[-1] > 1:
                pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            mask_resized = cv2.resize(pred, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_NEAREST)
            _, mask_thresh = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            output_mask_path = OUTPUT_MASKS_FOLDER / f"mask_{image_name}"
            cv2.imwrite(str(output_mask_path), mask_thresh)
            predicted_mask_paths.append(output_mask_path)

        # Step 3: Stack masks back into a NIfTI
        mask_files = sorted(os.listdir(OUTPUT_MASKS_FOLDER), key=lambda x: int("".join(filter(str.isdigit, x))))
        mask_slices = []

        for fname in mask_files:
            path = OUTPUT_MASKS_FOLDER / fname
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_slices.append(resized)

        if not mask_slices:
            raise Exception("No masks were generated.")

        predicted_volume = np.stack(mask_slices, axis=0)
        nii_pred = nib.Nifti1Image(predicted_volume, affine=np.eye(4))
        nib.save(nii_pred, predicted_nii_path_server)  # Save to the new location
        predicted_file_url = f"/predicted_nii/{predicted_nii_filename}"  #changed
        # Step 4: Move the original uploaded file to the public folder
        # original_filename_moved = f"original_{uuid.uuid4().hex}_{file.filename}"
        # original_file_path_moved = ORIGINAL_NII_PUBLIC_FOLDER / original_filename_moved # changed
        shutil.move(str(input_nii_path), str(original_file_path_moved))  # Save to the new location.
        original_file_url = f"/{original_filename_moved}" 
        predicted_file_url = f"/{predicted_nii_filename}" # changed
        global filepath1, filepath2
        filepath1 = original_file_url
        filepath2 = predicted_file_url
        print(filepath1)
        print({"success": True, "mask_path": predicted_file_url, "original_path": original_file_url})
        return {"success": True, "mask_path": predicted_file_url, "original_path": original_file_url}

    except nib.NiftiError as e:
        print(f"NiBabel Error: {e}")
        return {"success": False, "error": f"NiBabel Error: {e}"}
    except Exception as e:
        print(f"Error processing NII file: {e}")
        return {"success": False, "error": f"Error processing NII file: {e}"}
    finally:
        # Clean up the temporary uploaded file (if not moved) and temp folders
        if input_nii_path.exists():
            os.remove(input_nii_path)
        for folder in [SLICES_FOLDER, OUTPUT_MASKS_FOLDER]:
            if folder.exists():
                for file_path in folder.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)


# --- FastAPI Integration ---


app = FastAPI()
STATIC_DIR = 'C:/Projects/AeroMedAI/genAI/backend/static'
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as per your requirement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

num_features = 4
num_actions = 4  
state_size = 4
class DockingData(BaseModel):
    ec_number: str
    ligand_id: str

class InputData(BaseModel):
    SMILES: str

class ProteinData(BaseModel):
    PROTEIN: str



@app.post("/predict/segmentation/nii")
async def predict_segmentation(
    image_file: UploadFile = File(...),
    
):
    try:
        print(f"Received model_type: Lung")
        print(f"Received image_file: {image_file.filename}, {image_file.content_type}")

        model, error = load_model("lung")
        if error:
            raise HTTPException(status_code=500, detail=error)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found.")

        response = await process_nii(image_file, model)
        if response["success"]:
            return {"success": True, "mask_path": response["mask_path"], "original_path": response["original_path"]}

        
        if response["success"]!=True:
            raise HTTPException(status_code=500)
        if not response["mask_path"]:
            raise HTTPException(status_code=500, detail="Failed to process and save the prediction mask.")

        

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    


app.mount("/original_nii", StaticFiles(directory="C:/Projects/AeroMedAI/genAI/backend/static/original_nii"), name="original_nii")
app.mount("/predicted_nii", StaticFiles(directory="C:/Projects/AeroMedAI/genAI/backend/static/predicted_nii"), name="predicted_nii")
templates = Jinja2Templates(directory="templates")
@app.get('/papaya', response_class=HTMLResponse)
async def papaya_viewer(request: Request):
    """
    Displays the Papaya viewer page, passing the global file paths.
    """
    global filepath1, filepath2  # Declare globals
    print(filepath1 )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "file1_path": filepath1,
            "file2_path": filepath2,
        },
    )
# For running with: uvicorn filename:app --reload
print(app.routes)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)