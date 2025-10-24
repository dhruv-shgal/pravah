from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime, timedelta
from ultralytics import YOLO
import torch
import io
import base64
import os
import uuid
from pathlib import Path
import shutil
import re
import hashlib
from jose import jwt
from passlib.context import CryptContext
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*rcond parameter.*')

# Create directories for storing data
UPLOAD_DIR = Path("uploads")
FACE_EMBEDDINGS_DIR = Path("face_embeddings")
UPLOAD_DIR.mkdir(exist_ok=True)
FACE_EMBEDDINGS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Eco-Connect Verification API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (HTML, CSS, JS)
# This will serve all HTML files from the current directory
from pathlib import Path
STATIC_DIR = Path(__file__).parent

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "eco-connect-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# In-memory user database
users_db: Dict[str, dict] = {}

# Global verification system instance
verifier = None

class UserSignupResponse(BaseModel):
    success: bool
    message: str
    userid: Optional[str] = None
    username: Optional[str] = None

class LoginResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    userid: Optional[str] = None
    username: Optional[str] = None

class VerificationResponse(BaseModel):
    message: str
    exif_warning: bool
    face_match: bool
    points_awarded: int
    activity_verified: bool
    details: Optional[dict] = None

class EcoConnectVerificationSystem:
    def __init__(self, plantation_model_path, waste_model_path, animal_model_path):
        """Initialize the verification system"""
        print("Loading YOLO models...")
        self.plantation_model = YOLO(plantation_model_path)
        self.waste_model = YOLO(waste_model_path)
        self.animal_model = YOLO(animal_model_path)
        
        print("Initializing InsightFace (ArcFace) model...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        
        self.task_models = {
            'plantation': self.plantation_model,
            'waste': self.waste_model,
            'animal': self.animal_model
        }
        
        self.task_classes = {
            'plantation': ['person', 'plantation'],
            'waste': ['person', 'collecting-waste'],
            'animal': ['person', 'animal_feeding']
        }
        
        # Points mapping
        self.task_points = {
            'plantation': 25,
            'waste': 20,
            'animal': 30
        }
    
    def check_ai_metadata(self, image_path):
        """Check EXIF for AI-generated indicators"""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            
            if not exif_data:
                return False, "No EXIF metadata found"
            
            ai_indicators = [
                'stable diffusion', 'midjourney', 'dall-e', 'dalle',
                'ai generated', 'artificial intelligence', 'synthetic',
                'generated', 'diffusion', 'gan'
            ]
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, str):
                    value_lower = value.lower()
                    for indicator in ai_indicators:
                        if indicator in value_lower:
                            return True, f"⚠️ AI indicator found in {tag}: {value}"
            
            return False, "No AI indicators in metadata"
            
        except Exception as e:
            return False, f"Could not read EXIF: {str(e)}"
    
    def extract_exif_data(self, image_path):
        """Extract EXIF data from image"""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            
            if not exif_data:
                return {
                    'has_exif': False,
                    'datetime': None,
                    'software': None,
                    'message': "No EXIF data found"
                }
            
            datetime_str = None
            software = None
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal' or tag == 'DateTime':
                    datetime_str = value
                elif tag == 'Software':
                    software = value
            
            return {
                'has_exif': True,
                'datetime': datetime_str,
                'software': software,
                'message': "EXIF data extracted"
            }
            
        except Exception as e:
            return {
                'has_exif': False,
                'datetime': None,
                'software': None,
                'message': f"EXIF extraction failed: {str(e)}"
            }
    
    def verify_activity(self, image_path, task_type):
        """Verify activity matches the selected task"""
        try:
            if task_type not in self.task_models:
                return {
                    'is_valid': False,
                    'detected_classes': [],
                    'person_boxes': [],
                    'message': f"Invalid task type: {task_type}"
                }
            
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            elif isinstance(image_path, np.ndarray):
                image = image_path
            else:
                image = np.array(Image.open(image_path))
            
            model = self.task_models[task_type]
            results = model(image)
            
            detected_classes = []
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    if confidence > 0.5:
                        detected_classes.append(class_name)
                        
                        if class_name == 'person':
                            bbox = box.xyxy[0].cpu().numpy()
                            person_boxes.append({
                                'bbox': bbox.tolist(),
                                'confidence': confidence
                            })
            
            required_classes = self.task_classes[task_type]
            detected_set = set(detected_classes)
            required_set = set(required_classes)
            
            if not required_set.issubset(detected_set):
                missing = required_set - detected_set
                return {
                    'is_valid': False,
                    'detected_classes': list(detected_set),
                    'person_boxes': person_boxes,
                    'message': f"Activity verification failed: Missing {missing} in image"
                }
            
            if not person_boxes:
                return {
                    'is_valid': False,
                    'detected_classes': list(detected_set),
                    'person_boxes': [],
                    'message': "No person detected in the image"
                }
            
            return {
                'is_valid': True,
                'detected_classes': list(detected_set),
                'person_boxes': person_boxes,
                'message': f"Activity verified: {task_type} with {len(person_boxes)} person(s) detected"
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'detected_classes': [],
                'person_boxes': [],
                'message': f"Activity verification error: {str(e)}"
            }
    
    def verify_face(self, registered_face_embedding, task_image_path, person_boxes):
        """Verify if registered user's face matches any person in the task image"""
        try:
            if isinstance(task_image_path, str):
                task_image = cv2.imread(task_image_path)
            else:
                task_image = np.array(Image.open(task_image_path))
                task_image = cv2.cvtColor(task_image, cv2.COLOR_RGB2BGR)
            
            if not person_boxes:
                return {
                    'is_valid': False,
                    'matched_person': None,
                    'similarity': 0,
                    'message': "No persons detected in image"
                }
            
            similarity_threshold = 0.35
            
            for idx, person_box in enumerate(person_boxes):
                bbox = person_box['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                h, w = task_image.shape[:2]
                margin = 30
                y1 = max(0, y1 - margin)
                y2 = min(h, y2 + margin)
                x1 = max(0, x1 - margin)
                x2 = min(w, x2 + margin)
                
                person_crop = task_image[y1:y2, x1:x2]
                faces = self.face_app.get(person_crop)
                
                if not faces:
                    continue
                
                for face in faces:
                    face_embedding = face.embedding
                    similarity = self._compute_similarity(registered_face_embedding, face_embedding)
                    
                    if similarity >= similarity_threshold:
                        return {
                            'is_valid': True,
                            'matched_person': idx + 1,
                            'similarity': float(similarity),
                            'message': f"Face verified: Match found (similarity: {similarity:.4f})"
                        }
            
            return {
                'is_valid': False,
                'matched_person': None,
                'similarity': 0,
                'message': "Face verification failed: Registered user not found in image"
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'matched_person': None,
                'similarity': 0,
                'message': f"Face verification error: {str(e)}"
            }
    
    def _compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity"""
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return np.dot(embedding1, embedding2)
    
    def verify_task_complete(self, task_image_path, task_type, user_embedding=None):
        """Complete task verification with optional face matching"""
        result = {
            'exif_warning': False,
            'exif_message': '',
            'face_match': False,
            'activity_verified': False,
            'points': 0,
            'details': {}
        }
        
        # Check for AI metadata
        has_ai, ai_msg = self.check_ai_metadata(task_image_path)
        if has_ai:
            result['exif_warning'] = True
            result['exif_message'] = ai_msg
            print(ai_msg)
        
        # Extract EXIF
        exif_data = self.extract_exif_data(task_image_path)
        result['details']['exif'] = exif_data
        
        # Verify activity
        activity_result = self.verify_activity(task_image_path, task_type)
        result['details']['activity'] = activity_result
        result['activity_verified'] = activity_result['is_valid']
        
        if not activity_result['is_valid']:
            return result
        
        # Face verification (if user is logged in)
        if user_embedding is not None:
            face_result = self.verify_face(
                user_embedding,
                task_image_path,
                activity_result['person_boxes']
            )
            result['details']['face'] = face_result
            result['face_match'] = face_result['is_valid']
            
            if face_result['is_valid']:
                result['points'] = self.task_points.get(task_type, 0)
        else:
            # Guest mode - no face verification
            result['face_match'] = True  # Not applicable
            result['details']['face'] = {'message': 'Guest mode - face verification skipped'}
        
        return result


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least 1 uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least 1 lowercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least 1 number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least 1 special character"
    
    return True, "Password is valid"


def truncate_password(password: str) -> str:
    """Truncate password to 72 characters for bcrypt compatibility"""
    # Bcrypt has a 72-byte limit, so we truncate to 72 characters to be safe
    if len(password) > 72:
        password = password[:72]
    return password


def hash_password(password: str) -> str:
    """Hash password with bcrypt (truncate to 72 chars if needed)"""
    try:
        password = truncate_password(password)
        # Use bcrypt directly to avoid passlib version issues
        import bcrypt
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    except Exception as e:
        print(f"⚠️ Bcrypt hashing error: {e}")
        print("Using SHA256 fallback...")
        import hashlib
        return "sha256:" + hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password with bcrypt (truncate to 72 chars if needed)"""
    try:
        plain_password = truncate_password(plain_password)
        
        # Check if it's a SHA256 hash (fallback)
        if hashed_password.startswith("sha256:"):
            import hashlib
            return "sha256:" + hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
        
        # Use bcrypt directly
        import bcrypt
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        print(f"⚠️ Password verification error: {e}")
        return False


def create_access_token(data: dict):
    """Create JWT token"""
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        print(f"⚠️ JWT encoding error: {e}")
        # Simple fallback token
        import base64
        token_data = f"{data.get('sub', 'user')}:{datetime.utcnow().isoformat()}"
        return base64.b64encode(token_data.encode()).decode()


@app.on_event("startup")
async def startup_event():
    """Initialize the verification system on startup"""
    global verifier
    
    # Use absolute paths for YOLO models
    plantation_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\pravah\best_yolov11_plantation.pt'
    waste_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\pravah\best_yolov11_waste_management.pt'
    animal_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\pravah\animal_feeding_yolov11.pt'
    
    # Check if model files exist
    if not os.path.exists(plantation_path):
        print(f"WARNING: {plantation_path} not found!")
    if not os.path.exists(waste_path):
        print(f"WARNING: {waste_path} not found!")
    if not os.path.exists(animal_path):
        print(f"WARNING: {animal_path} not found!")
    
    try:
        verifier = EcoConnectVerificationSystem(
            plantation_model_path=plantation_path,
            waste_model_path=waste_path,
            animal_model_path=animal_path
        )
        print("✅ Verification system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize verification system: {e}")
        verifier = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if verifier is not None else "unhealthy",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": verifier is not None,
        "users_count": len(users_db)
    }


@app.post("/api/register")
async def api_register(
    username: str = Form(...),
    email: str = Form(...),
    user_id: str = Form(...),
    password: str = Form(...),
    face_image_0: UploadFile = File(...),
    face_image_1: Optional[UploadFile] = File(None),
    face_image_2: Optional[UploadFile] = File(None),
    face_image_3: Optional[UploadFile] = File(None),
    face_image_4: Optional[UploadFile] = File(None)
):
    """
    API endpoint for HTML frontend registration (matches signup.html expectations)
    """
    try:
        if verifier is None:
            return JSONResponse(
                status_code=503,
                content={"success": False, "message": "Service unavailable"}
            )
        
        # Use the provided user_id
        userid = user_id
        
        # Check if userid already exists
        if userid in users_db:
            return JSONResponse(
                content={"success": False, "message": "User ID already exists"}
            )
        
        # Check if email already exists
        for user_data in users_db.values():
            if user_data['email'] == email:
                return JSONResponse(
                    content={"success": False, "message": "Email already registered"}
                )
        
        # Validate password (before truncation to ensure requirements are met)
        is_valid, msg = validate_password(password)
        if not is_valid:
            return JSONResponse(
                content={"success": False, "message": msg}
            )
        
        # Use the first face image
        face_image_path = UPLOAD_DIR / f"{userid}_signup.jpg"
        with open(face_image_path, "wb") as buffer:
            shutil.copyfileobj(face_image_0.file, buffer)
        
        # Load image and detect face
        image = cv2.imread(str(face_image_path))
        faces = verifier.face_app.get(image)
        
        if not faces:
            os.remove(face_image_path)
            return JSONResponse(
                content={"success": False, "message": "No face detected in the image"}
            )
        
        if len(faces) > 1:
            os.remove(face_image_path)
            return JSONResponse(
                content={"success": False, "message": "Multiple faces detected. Please use an image with a single face."}
            )
        
        face = faces[0]
        face_embedding = face.embedding
        
        # Save face embedding
        embedding_path = FACE_EMBEDDINGS_DIR / f"{userid}.npy"
        np.save(embedding_path, face_embedding)
        
        # Hash password (with 72-byte limit handling)
        hashed_password = hash_password(password)
        
        # Store user data in memory
        users_db[userid] = {
            'userid': userid,
            'username': username,
            'email': email,
            'password': hashed_password,
            'face_embedding_path': str(embedding_path),
            'created_at': datetime.now().isoformat()
        }
        
        print(f"✅ User registered: {userid} ({username})")
        
        return JSONResponse(content={
            "success": True,
            "message": "Account created successfully!",
            "user": {
                "user_id": userid,
                "username": username,
                "email": email
            }
        })
        
    except Exception as e:
        print(f"Registration error: {e}")
        return JSONResponse(
            content={"success": False, "message": f"Registration failed: {str(e)}"}
        )


@app.post("/signup", response_model=UserSignupResponse)
async def signup(
    userid: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    face_image: UploadFile = File(...)
):
    """
    User signup with face capture and password
    
    - *userid*: Unique user ID
    - *username*: User's username
    - *email*: User's email
    - *password*: Password (min 8 chars, 1 uppercase, 1 lowercase, 1 number, 1 special char)
    - *face_image*: Face image file (JPG/PNG)
    """
    try:
        if verifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Verification system not initialized"
            )
        
        # Check if userid or email already exists
        if userid in users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID already exists"
            )
        
        for user_data in users_db.values():
            if user_data['email'] == email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        # Validate password
        is_valid, msg = validate_password(password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg
            )
        
        # Save uploaded face image
        face_image_path = UPLOAD_DIR / f"{userid}_signup.jpg"
        with open(face_image_path, "wb") as buffer:
            shutil.copyfileobj(face_image.file, buffer)
        
        # Load image and detect face
        image = cv2.imread(str(face_image_path))
        faces = verifier.face_app.get(image)
        
        if not faces:
            os.remove(face_image_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image"
            )
        
        if len(faces) > 1:
            os.remove(face_image_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multiple faces detected. Please use an image with a single face."
            )
        
        face = faces[0]
        face_embedding = face.embedding
        
        # Save face embedding
        embedding_path = FACE_EMBEDDINGS_DIR / f"{userid}.npy"
        np.save(embedding_path, face_embedding)
        
        # Hash password (with 72-byte limit handling)
        hashed_password = hash_password(password)
        
        # Store user data in memory
        users_db[userid] = {
            'userid': userid,
            'username': username,
            'email': email,
            'password': hashed_password,
            'face_embedding_path': str(embedding_path),
            'created_at': datetime.now().isoformat()
        }
        
        print(f"✅ User registered: {userid} ({username})")
        
        return UserSignupResponse(
            success=True,
            message="User registered successfully!",
            userid=userid,
            username=username
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signup failed: {str(e)}"
        )


@app.post("/login", response_model=LoginResponse)
async def login(
    userid: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    password: str = Form(...)
):
    """
    User login with userid/email and password
    
    - *userid*: User's unique ID (optional if email provided)
    - *email*: User's email (optional if userid provided)
    - *password*: User's password
    """
    try:
        if verifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Verification system not initialized"
            )
        
        if not userid and not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either userid or email must be provided"
            )
        
        # Find user
        user_data = None
        if userid and userid in users_db:
            user_data = users_db[userid]
        elif email:
            for uid, data in users_db.items():
                if data['email'] == email:
                    user_data = data
                    userid = uid
                    break
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify password (with 72-byte limit handling)
        if not verify_password(password, user_data['password']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect password"
            )
        
        # Create JWT token
        token = create_access_token(data={"sub": userid, "username": user_data['username']})
        
        print(f"✅ User logged in: {userid} ({user_data['username']})")
        
        return LoginResponse(
            success=True,
            message="Login successful!",
            token=token,
            userid=userid,
            username=user_data['username']
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@app.post("/api/login")
async def api_login(
    username: str = Form(...),
    password: str = Form(...)
):
    """
    API endpoint for HTML frontend login (matches login.html expectations)
    """
    try:
        if verifier is None:
            return JSONResponse(
                status_code=503,
                content={"success": False, "message": "Service unavailable"}
            )
        
        # Find user by username
        user_data = None
        userid = None
        for uid, data in users_db.items():
            if data['username'] == username or data['userid'] == username:
                user_data = data
                userid = uid
                break
        
        if not user_data:
            return JSONResponse(
                content={"success": False, "message": "User not found"}
            )
        
        # Verify password (with 72-byte limit handling)
        if not verify_password(password, user_data['password']):
            return JSONResponse(
                content={"success": False, "message": "Incorrect password"}
            )
        
        # Create JWT token
        token = create_access_token(data={"sub": userid, "username": user_data['username']})
        
        print(f"✅ User logged in: {userid} ({user_data['username']})")
        
        return JSONResponse(content={
            "success": True,
            "message": "Login successful!",
            "user": {
                "user_id": userid,
                "username": user_data['username'],
                "email": user_data['email'],
                "eco_coins": 0,
                "uploads": [],
                "face_embeddings": []
            },
            "token": token
        })
        
    except Exception as e:
        print(f"Login error: {e}")
        return JSONResponse(
            content={"success": False, "message": f"Login failed: {str(e)}"}
        )


# Separate API endpoints for each model
@app.post("/api/verify/plantation")
async def verify_plantation(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Plantation activity verification API
    """
    try:
        print("🌱 Processing plantation verification...")
        
        # Save uploaded image
        task_image_path = UPLOAD_DIR / f"plantation_{uuid.uuid4()}.jpg"
        with open(task_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print(f"📁 Image saved: {task_image_path}")
        
        # Load plantation model
        try:
            plantation_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\pravah\best_yolov11_plantation.pt'
            print(f"🔄 Loading plantation model from: {plantation_path}")
            plantation_model = YOLO(plantation_path)
            print("✅ Plantation model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load plantation model: {e}")
            os.remove(task_image_path)
            return JSONResponse(content={
                "success": False,
                "message": f"Failed to load plantation model: {str(e)}"
            })
        
        # Run YOLO detection
        try:
            print("🔍 Running YOLO detection...")
            results = plantation_model(str(task_image_path))
            
            detected_classes = []
            person_detected = False
            plantation_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:
                            detected_classes.append(f"{class_name} ({confidence:.2f})")
                            if class_name == 'person':
                                person_detected = True
                            elif 'plantation' in class_name.lower():
                                plantation_detected = True
            
            print(f"🎯 Detected classes: {detected_classes}")
            
            # Check if both person and plantation are detected
            activity_verified = person_detected and plantation_detected
            points = 25 if activity_verified and user_id else 0
            
            # Build response message
            if activity_verified:
                if user_id:
                    message = f"✅ Plantation activity verified! {points} eco-points awarded."
                else:
                    message = "✅ Plantation activity verified (Guest mode - no points awarded)."
            else:
                missing = []
                if not person_detected:
                    missing.append("person")
                if not plantation_detected:
                    missing.append("plantation activity")
                message = f"❌ Plantation verification failed. Missing: {', '.join(missing)}"
            
            # Clean up
            os.remove(task_image_path)
            
            return JSONResponse(content={
                "success": activity_verified,
                "message": message,
                "detected_classes": detected_classes,
                "person_detected": person_detected,
                "plantation_detected": plantation_detected,
                "points_awarded": points,
                "activity_verified": activity_verified
            })
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            os.remove(task_image_path)
            return JSONResponse(content={
                "success": False,
                "message": f"Detection failed: {str(e)}"
            })
        
    except Exception as e:
        print(f"❌ Plantation verification error: {e}")
        return JSONResponse(content={
            "success": False,
            "message": f"Plantation verification failed: {str(e)}"
        })


@app.post("/api/verify/waste")
async def verify_waste(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Waste collection activity verification API
    """
    try:
        print("♻️ Processing waste collection verification...")
        
        # Save uploaded image
        task_image_path = UPLOAD_DIR / f"waste_{uuid.uuid4()}.jpg"
        with open(task_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print(f"📁 Image saved: {task_image_path}")
        
        # Load waste model
        try:
            waste_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\pravah\best_yolov11_waste_management.pt'
            print(f"🔄 Loading waste model from: {waste_path}")
            waste_model = YOLO(waste_path)
            print("✅ Waste model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load waste model: {e}")
            os.remove(task_image_path)
            return JSONResponse(content={
                "success": False,
                "message": f"Failed to load waste model: {str(e)}"
            })
        
        # Run YOLO detection
        try:
            print("🔍 Running YOLO detection...")
            results = waste_model(str(task_image_path))
            
            detected_classes = []
            person_detected = False
            waste_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:
                            detected_classes.append(f"{class_name} ({confidence:.2f})")
                            if class_name == 'person':
                                person_detected = True
                            elif 'waste' in class_name.lower() or 'collecting' in class_name.lower():
                                waste_detected = True
            
            print(f"🎯 Detected classes: {detected_classes}")
            
            # Check if both person and waste collection are detected
            activity_verified = person_detected and waste_detected
            points = 20 if activity_verified and user_id else 0
            
            # Build response message
            if activity_verified:
                if user_id:
                    message = f"✅ Waste collection activity verified! {points} eco-points awarded."
                else:
                    message = "✅ Waste collection activity verified (Guest mode - no points awarded)."
            else:
                missing = []
                if not person_detected:
                    missing.append("person")
                if not waste_detected:
                    missing.append("waste collection activity")
                message = f"❌ Waste collection verification failed. Missing: {', '.join(missing)}"
            
            # Clean up
            os.remove(task_image_path)
            
            return JSONResponse(content={
                "success": activity_verified,
                "message": message,
                "detected_classes": detected_classes,
                "person_detected": person_detected,
                "waste_detected": waste_detected,
                "points_awarded": points,
                "activity_verified": activity_verified
            })
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            os.remove(task_image_path)
            return JSONResponse(content={
                "success": False,
                "message": f"Detection failed: {str(e)}"
            })
        
    except Exception as e:
        print(f"❌ Waste verification error: {e}")
        return JSONResponse(content={
            "success": False,
            "message": f"Waste verification failed: {str(e)}"
        })


@app.post("/api/verify/animal")
async def verify_animal(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Animal feeding activity verification API
    """
    try:
        print("🐕 Processing animal feeding verification...")
        
        # Save uploaded image
        task_image_path = UPLOAD_DIR / f"animal_{uuid.uuid4()}.jpg"
        with open(task_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print(f"📁 Image saved: {task_image_path}")
        
        # Load animal model
        try:
            animal_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\pravah\animal_feeding_yolov11.pt'
            print(f"🔄 Loading animal model from: {animal_path}")
            animal_model = YOLO(animal_path)
            print("✅ Animal model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load animal model: {e}")
            os.remove(task_image_path)
            return JSONResponse(content={
                "success": False,
                "message": f"Failed to load animal model: {str(e)}"
            })
        
        # Run YOLO detection
        try:
            print("🔍 Running YOLO detection...")
            results = animal_model(str(task_image_path))
            
            detected_classes = []
            person_detected = False
            animal_feeding_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:
                            detected_classes.append(f"{class_name} ({confidence:.2f})")
                            if class_name == 'person':
                                person_detected = True
                            elif 'animal' in class_name.lower() or 'feeding' in class_name.lower():
                                animal_feeding_detected = True
            
            print(f"🎯 Detected classes: {detected_classes}")
            
            # Check if both person and animal feeding are detected
            activity_verified = person_detected and animal_feeding_detected
            points = 30 if activity_verified and user_id else 0
            
            # Build response message
            if activity_verified:
                if user_id:
                    message = f"✅ Animal feeding activity verified! {points} eco-points awarded."
                else:
                    message = "✅ Animal feeding activity verified (Guest mode - no points awarded)."
            else:
                missing = []
                if not person_detected:
                    missing.append("person")
                if not animal_feeding_detected:
                    missing.append("animal feeding activity")
                message = f"❌ Animal feeding verification failed. Missing: {', '.join(missing)}"
            
            # Clean up
            os.remove(task_image_path)
            
            return JSONResponse(content={
                "success": activity_verified,
                "message": message,
                "detected_classes": detected_classes,
                "person_detected": person_detected,
                "animal_feeding_detected": animal_feeding_detected,
                "points_awarded": points,
                "activity_verified": activity_verified
            })
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            os.remove(task_image_path)
            return JSONResponse(content={
                "success": False,
                "message": f"Detection failed: {str(e)}"
            })
        
    except Exception as e:
        print(f"❌ Animal verification error: {e}")
        return JSONResponse(content={
            "success": False,
            "message": f"Animal verification failed: {str(e)}"
        })


# General verify_task endpoint that routes to specific APIs
@app.post("/api/verify_task")
@app.post("/api/verify-task")  # Alias for frontend compatibility
async def api_verify_task(
    activity_type: str = Form(...),
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    General task verification API that routes to specific model APIs
    Accepts both /api/verify_task and /api/verify-task
    """
    try:
        print(f"🔄 Routing {activity_type} verification to specific API...")
        
        if activity_type == "plantation":
            return await verify_plantation(image=image, user_id=user_id)
        elif activity_type == "waste":
            return await verify_waste(image=image, user_id=user_id)
        elif activity_type == "animal":
            return await verify_animal(image=image, user_id=user_id)
        else:
            return JSONResponse(content={
                "success": False,
                "message": f"Invalid activity type: {activity_type}. Must be 'plantation', 'waste', or 'animal'"
            })
    
    except Exception as e:
        print(f"❌ Routing error: {e}")
        return JSONResponse(content={
            "success": False,
            "message": f"Verification routing failed: {str(e)}"
        })


@app.post("/verify_task", response_model=VerificationResponse)
async def verify_task(
    activity_type: str = Form(...),
    image: UploadFile = File(...),
    userid: Optional[str] = Form(None)
):
    """
    Verify task submission with YOLO and optional face verification
    
    - *activity_type*: Type of activity ("plantation", "waste", or "animal")
    - *image*: Image of user performing the task
    - *userid*: User's unique ID (optional for guest mode)
    """
    try:
        if verifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Verification system not initialized"
            )
        
        # Validate activity type
        valid_activities = ['plantation', 'waste', 'animal']
        if activity_type not in valid_activities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid activity type. Must be one of: {valid_activities}"
            )
        
        # Save uploaded image
        task_image_path = UPLOAD_DIR / f"task_{uuid.uuid4()}.jpg"
        with open(task_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Load user embedding if logged in
        user_embedding = None
        if userid:
            if userid not in users_db:
                os.remove(task_image_path)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            embedding_path = FACE_EMBEDDINGS_DIR / f"{userid}.npy"
            if embedding_path.exists():
                user_embedding = np.load(embedding_path)
        
        # Run verification
        result = verifier.verify_task_complete(
            task_image_path=str(task_image_path),
            task_type=activity_type,
            user_embedding=user_embedding
        )
        
        # Clean up
        os.remove(task_image_path)
        
        # Build response message
        if userid:
            if result['face_match'] and result['activity_verified']:
                message = f"✅ Task verified! {result['points']} eco-points awarded to {users_db[userid]['username']}."
            elif result['activity_verified'] and not result['face_match']:
                message = "⚠️ Activity verified but user faces differ — no points awarded."
            else:
                message = "❌ Activity verification failed."
        else:
            if result['activity_verified']:
                message = "✅ Activity verified (Guest mode - no points awarded)."
            else:
                message = "❌ Activity verification failed."
        
        if result['exif_warning']:
            message = result['exif_message'] + " " + message
        
        return VerificationResponse(
            message=message,
            exif_warning=result['exif_warning'],
            face_match=result['face_match'],
            points_awarded=result['points'],
            activity_verified=result['activity_verified'],
            details=result['details']
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task verification failed: {str(e)}"
        )


# Static file serving - MUST BE LAST to not override API routes
@app.get("/")
async def root():
    """Serve the main eco-connect-site.html page"""
    eco_connect_path = STATIC_DIR / "eco-connect-site.html"
    if eco_connect_path.exists():
        return FileResponse(eco_connect_path)
    return {
        "message": "Welcome to Eco-Connect Verification API",
        "version": "2.0.0",
        "status": "running" if verifier is not None else "error",
        "endpoints": {
            "signup": "/api/register",
            "login": "/api/login",
            "verify_task": "/api/verify_task",
            "health": "/health"
        }
    }


@app.get("/{file_path:path}")
async def serve_static_files(file_path: str):
    """Serve static HTML, CSS, JS files - catches all remaining routes"""
    # Security: prevent directory traversal
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_full_path = STATIC_DIR / file_path
    
    if file_full_path.exists() and file_full_path.is_file():
        return FileResponse(file_full_path)
    
    raise HTTPException(status_code=404, detail="File not found")


# Guest mode endpoints (for guest-mode.html)
@app.post("/api/guest/verify-plantation")
async def guest_verify_plantation(
    activity_image: UploadFile = File(...),
    description: str = Form(...)
):
    """Guest mode plantation verification"""
    return await verify_plantation(image=activity_image, user_id=None)


@app.post("/api/guest/verify-waste-collection")
async def guest_verify_waste_collection(
    activity_image: UploadFile = File(...),
    description: str = Form(...)
):
    """Guest mode waste collection verification"""
    return await verify_waste(image=activity_image, user_id=None)


@app.post("/api/guest/verify-animal-feeding")
async def guest_verify_animal_feeding(
    activity_image: UploadFile = File(...),
    description: str = Form(...)
):
    """Guest mode animal feeding verification"""
    return await verify_animal(image=activity_image, user_id=None)


# Add missing /api/test-models endpoint that frontend is calling
@app.get("/api/test-models")
async def test_models():
    """Test if models are loaded"""
    return {
        "models_loaded": verifier is not None,
        "status": "ready" if verifier is not None else "not_ready"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Eco-Connect Full Stack Server...")
    print("📍 Frontend: http://localhost:8000")
    print("📍 Backend API: http://localhost:8000/api/*")
    print("📚 API docs: http://localhost:8000/docs")
    print("💚 Eco-Connect is ready!")
    uvicorn.run(app, host="0.0.0.0", port=8000)