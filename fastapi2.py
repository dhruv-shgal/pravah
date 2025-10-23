from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime, timedelta
from ultralytics import YOLO
import torch
from transformers import pipeline
import io
import base64
import os
import uuid
from pathlib import Path
import shutil

# Create directories for storing data
UPLOAD_DIR = Path("uploads")
FACE_EMBEDDINGS_DIR = Path("face_embeddings")
UPLOAD_DIR.mkdir(exist_ok=True)
FACE_EMBEDDINGS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Eco-Connect Verification API", version="1.0.0")

# Global verification system instance
verifier = None

class UserSignupResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    face_info: Optional[dict] = None

class LoginResponse(BaseModel):
    success: bool
    message: str
    user_data: Optional[dict] = None

class VerificationResponse(BaseModel):
    overall_valid: bool
    message: Optional[str] = None
    steps: dict

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
        
        try:
            print("Loading AI image detector...")
            self.ai_detector = pipeline("image-classification", 
                                       model="umm-maybe/AI-image-detector")
        except Exception as e:
            print(f"AI detector initialization warning: {e}")
            self.ai_detector = None
        
        self.task_models = {
            'plantation': self.plantation_model,
            'waste_management': self.waste_model,
            'stray_animal_feeding': self.animal_model
        }
        
        self.task_classes = {
            'plantation': ['person', 'plantation'],
            'waste_management': ['person', 'collecting-waste'],
            'stray_animal_feeding': ['person', 'animal_feeding']
        }
    
    def verify_ai_image(self, image):
        """Check if image is AI-generated"""
        try:
            if self.ai_detector is None:
                return self._basic_ai_check(image)
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            results = self.ai_detector(image)
            
            for result in results:
                if 'artificial' in result['label'].lower() or 'ai' in result['label'].lower():
                    if result['score'] > 0.7:
                        return {
                            'is_valid': False,
                            'confidence': result['score'],
                            'message': f"Image rejected: AI-generated (confidence: {result['score']:.2%})"
                        }
            
            return {
                'is_valid': True,
                'confidence': results[0]['score'] if results else 0,
                'message': "Image passed AI detection check"
            }
            
        except Exception as e:
            print(f"AI detection error: {e}")
            return self._basic_ai_check(image)
    
    def _basic_ai_check(self, image):
        """Fallback AI detection"""
        if isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = np.array(image)
        
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:
            return {
                'is_valid': False,
                'confidence': 0.6,
                'message': "Image rejected: Possibly AI-generated (low texture variance)"
            }
        
        return {
            'is_valid': True,
            'confidence': 0.5,
            'message': "Image passed basic AI check"
        }
    
    def verify_exif_datetime(self, image_path):
        """Verify EXIF date/time is within one week"""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            
            if not exif_data:
                return {
                    'is_valid': False,
                    'capture_time': None,
                    'message': "Image rejected: No EXIF data found"
                }
            
            datetime_str = None
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal' or tag == 'DateTime':
                    datetime_str = value
                    break
            
            if not datetime_str:
                return {
                    'is_valid': False,
                    'capture_time': None,
                    'message': "Image rejected: No timestamp in EXIF data"
                }
            
            capture_time = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
            current_time = datetime.now()
            time_diff = current_time - capture_time
            
            if time_diff.days < 0:
                return {
                    'is_valid': False,
                    'capture_time': capture_time.isoformat(),
                    'message': f"Image rejected: Future timestamp detected ({capture_time})"
                }
            
            if time_diff.days > 7:
                return {
                    'is_valid': False,
                    'capture_time': capture_time.isoformat(),
                    'message': f"Image rejected: Image older than 7 days ({time_diff.days} days old)"
                }
            
            return {
                'is_valid': True,
                'capture_time': capture_time.isoformat(),
                'message': f"Image timestamp valid ({time_diff.days} days old)"
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'capture_time': None,
                'message': f"EXIF verification failed: {str(e)}"
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
    
    def full_verification(self, registered_face_embedding, task_image_path, task_type):
        """Complete verification pipeline"""
        results = {
            'overall_valid': False,
            'steps': {}
        }
        
        if isinstance(task_image_path, str):
            img = Image.open(task_image_path)
        else:
            img = Image.open(task_image_path)
        
        ai_result = self.verify_ai_image(img)
        results['steps']['ai_detection'] = ai_result
        
        if not ai_result['is_valid']:
            return results
        
        exif_result = self.verify_exif_datetime(task_image_path)
        results['steps']['exif_verification'] = exif_result
        
        if not exif_result['is_valid']:
            return results
        
        activity_result = self.verify_activity(task_image_path, task_type)
        results['steps']['activity_verification'] = activity_result
        
        if not activity_result['is_valid']:
            return results
        
        face_result = self.verify_face(
            registered_face_embedding, 
            task_image_path, 
            activity_result['person_boxes']
        )
        results['steps']['face_verification'] = face_result
        
        if not face_result['is_valid']:
            return results
        
        results['overall_valid'] = True
        results['message'] = "All verification checks passed successfully!"
        
        return results
    
    def anonymous_verification(self, task_image_path, task_type):
        """Verification pipeline for anonymous users (no face verification)"""
        results = {
            'overall_valid': False,
            'steps': {}
        }
        
        if isinstance(task_image_path, str):
            img = Image.open(task_image_path)
        else:
            img = Image.open(task_image_path)
        
        # Step 1: AI Detection
        ai_result = self.verify_ai_image(img)
        results['steps']['ai_detection'] = ai_result
        
        if not ai_result['is_valid']:
            results['message'] = ai_result.get('message', 'AI detection failed')
            return results
        
        # Step 2: EXIF Verification
        exif_result = self.verify_exif_datetime(task_image_path)
        results['steps']['exif_verification'] = exif_result
        
        if not exif_result['is_valid']:
            results['message'] = exif_result.get('message', 'EXIF verification failed')
            return results
        
        # Step 3: Activity Verification
        activity_result = self.verify_activity(task_image_path, task_type)
        results['steps']['activity_verification'] = activity_result
        
        if not activity_result['is_valid']:
            results['message'] = activity_result.get('message', f'Activity verification failed: Image does not match {task_type} activity')
            return results
        
        # Step 4: Skip face verification for anonymous users
        results['steps']['face_verification'] = {
            'is_valid': True,
            'message': 'Face verification skipped for anonymous usage',
            'matched_person': None,
            'similarity': 1.0
        }
        
        results['overall_valid'] = True
        results['message'] = f"Task verification completed successfully! Image matches {task_type.replace('_', ' ')} activity."
        
        return results


@app.on_event("startup")
async def startup_event():
    """Initialize the verification system on startup"""
    global verifier
    
    # Check if model files exist
    plantation_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\eco_connect\plantation_yolov11.pt'
    waste_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\eco_connect\waste_collection_yolov11.pt'
    animal_path = r'C:\Users\dhruv\OneDrive\Desktop\pravah\eco_connect\animal_feeding_yolov11.pt'
    
    if not os.path.exists(plantation_path):
        print(f"WARNING: plantation_yolov11.pt not found at {plantation_path}")
    if not os.path.exists(waste_path):
        print(f"WARNING: waste_collection_yolov11.pt not found at {waste_path}")
    if not os.path.exists(animal_path):
        print(f"WARNING: animal_feeding_yolov11.pt not found at {animal_path}")
    
    try:
        print("🔄 Starting verification system initialization...")
        print(f"Model files check:")
        print(f"  Plantation: {os.path.exists(plantation_path)}")
        print(f"  Waste: {os.path.exists(waste_path)}")
        print(f"  Animal: {os.path.exists(animal_path)}")
        
        verifier = EcoConnectVerificationSystem(
            plantation_model_path=plantation_path,
            waste_model_path=waste_path,
            animal_model_path=animal_path
        )
        print("✅ Verification system initialized successfully!")
    except Exception as e:
        import traceback
        print(f"❌ Failed to initialize verification system: {e}")
        print(f"Full error traceback:")
        traceback.print_exc()
        print(f"Make sure the following model files exist:")
        print(f"  - {plantation_path}")
        print(f"  - {waste_path}")
        print(f"  - {animal_path}")
        verifier = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Eco-Connect Verification API",
        "version": "1.0.0",
        "status": "running" if verifier is not None else "error",
        "endpoints": {
            "signup": "/api/signup",
            "login": "/api/login",
            "verify_task": "/api/verify-task",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if verifier is not None else "unhealthy",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": verifier is not None
    }


@app.post("/api/signup", response_model=UserSignupResponse)
async def signup(
    email: str = Form(...),
    password: str = Form(...),
    username: str = Form(...),
    user_id: str = Form(...),
    face_images: List[UploadFile] = File(...)
):
    """
    User signup with multiple face captures
    
    - *email*: User's email
    - *password*: User's password
    - *username*: User's username
    - *user_id*: User's unique ID
    - *face_images*: List of 5 face image files (JPG/PNG)
    """
    try:
        # Validate face images count
        if len(face_images) != 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected 5 face images, got {len(face_images)}"
            )
        
        # Create user directory
        user_dir = UPLOAD_DIR / user_id
        user_dir.mkdir(exist_ok=True)
        
        face_embeddings = []
        saved_images = []
        
        # Process each face image
        for i, face_image in enumerate(face_images):
            # Save uploaded face image
            face_image_path = user_dir / f"face_{i+1}.jpg"
            with open(face_image_path, "wb") as buffer:
                shutil.copyfileobj(face_image.file, buffer)
            saved_images.append(face_image_path)
            
            # Load image and detect face (only if verifier is available)
            if verifier and verifier.face_app:
                image = cv2.imread(str(face_image_path))
                faces = verifier.face_app.get(image)
                
                if not faces:
                    # Clean up saved images
                    for img_path in saved_images:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"No face detected in image {i+1}"
                    )
                
                if len(faces) > 1:
                    # Clean up saved images
                    for img_path in saved_images:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Multiple faces detected in image {i+1}. Please use images with a single face."
                    )
                
                face = faces[0]
                face_embeddings.append(face.embedding)
        
        # Save face embeddings (average of all 5 images for better accuracy)
        if face_embeddings and verifier:
            avg_embedding = np.mean(face_embeddings, axis=0)
            embedding_path = FACE_EMBEDDINGS_DIR / f"{user_id}.npy"
            np.save(embedding_path, avg_embedding)
            
            face_info = {
                'images_processed': len(face_embeddings),
                'embedding_shape': avg_embedding.shape,
                'note': 'Face embeddings saved successfully'
            }
        else:
            face_info = {
                'images_processed': len(face_images),
                'embedding_shape': None,
                'note': 'Face processing skipped - verifier not available'
            }
        
        # Store user data (in a real app, this would go to a database)
        user_data = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'password': password,  # In production, hash this password!
            'face_images_count': len(face_images),
            'created_at': datetime.now().isoformat()
        }
        
        # Save user data to file (temporary solution)
        user_data_path = UPLOAD_DIR / f"{user_id}_data.json"
        with open(user_data_path, 'w') as f:
            json.dump(user_data, f)
        
        return UserSignupResponse(
            success=True,
            message="User registered successfully with 5 face images!",
            user_id=user_id,
            face_info=face_info
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signup failed: {str(e)}"
        )


@app.post("/api/login", response_model=LoginResponse)
async def login(
    identifier: str = Form(...),  # Can be email or user_id
    password: str = Form(...)
):
    """
    User login with email/userid and password
    
    - *identifier*: User's email or user_id
    - *password*: User's password
    """
    try:
        # Search for user by email or user_id
        user_found = False
        user_data = None
        
        # Check all user data files to find matching email or user_id
        for user_file in UPLOAD_DIR.glob("*_data.json"):
            try:
                with open(user_file, 'r') as f:
                    data = json.load(f)
                    if data.get('email') == identifier or data.get('user_id') == identifier:
                        user_data = data
                        user_found = True
                        break
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        if not user_found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify password (in production, use proper password hashing)
        if user_data.get('password') != password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password"
            )
        
        # Return user data (excluding password)
        safe_user_data = {
            'user_id': user_data.get('user_id'),
            'username': user_data.get('username'),
            'email': user_data.get('email'),
            'created_at': user_data.get('created_at')
        }
        
        return LoginResponse(
            success=True,
            message="Login successful!",
            user_data=safe_user_data
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )



@app.post("/api/verify-task", response_model=VerificationResponse)
async def verify_task(
    task_type: str = Form(...),
    task_image: UploadFile = File(...)
):
    """
    Verify task submission with AI detection and activity verification
    
    - *task_type*: Type of task (plantation, waste_management, stray_animal_feeding)
    - *task_image*: Image of user performing the task
    """
    try:
        # Validate task type
        valid_tasks = ['plantation', 'waste_management', 'stray_animal_feeding']
        if task_type not in valid_tasks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid task type. Must be one of: {valid_tasks}"
            )
        
        # Save task image
        task_image_path = UPLOAD_DIR / f"task_{uuid.uuid4()}.jpg"
        with open(task_image_path, "wb") as buffer:
            shutil.copyfileobj(task_image.file, buffer)
        
        # Check if verifier is initialized
        if verifier is None:
            # Clean up task image
            if os.path.exists(task_image_path):
                os.remove(task_image_path)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Verification system not initialized. Please check if YOLO model files exist and restart the server."
            )
        
        # Run anonymous verification (AI detection + activity verification, no face verification)
        verification_results = verifier.anonymous_verification(
            task_image_path=str(task_image_path),
            task_type=task_type
        )
        
        # Clean up
        os.remove(task_image_path)
        
        return VerificationResponse(**verification_results)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task verification failed: {str(e)}"
        )





@app.delete("/api/user/{user_id}")
async def delete_user(user_id: str):
    """Delete user and their face embedding"""
    try:
        embedding_path = FACE_EMBEDDINGS_DIR / f"{user_id}.npy"
        if not embedding_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Delete face embedding
        os.remove(embedding_path)
        
        # Delete signup image if exists
        signup_image = UPLOAD_DIR / f"{user_id}_signup.jpg"
        if signup_image.exists():
            os.remove(signup_image)
        
        return {"success": True, "message": f"User {user_id} deleted successfully"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User deletion failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)