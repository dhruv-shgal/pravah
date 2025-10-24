# Requirements Document

## Introduction

The Eco-Connect Verification API is a FastAPI-based system that enables users to register with facial recognition, authenticate, and submit environmental task images for verification. The system verifies that submitted images are authentic (not AI-generated), recent (within 7 days via EXIF data), contain the appropriate activity (plantation, waste management, or stray animal feeding), and optionally match the registered user's face. The system supports both authenticated and anonymous task submissions.

## Glossary

- **Verification System**: The EcoConnectVerificationSystem class that orchestrates AI detection, EXIF validation, activity recognition, and face verification
- **Face Embedding**: A numerical vector representation of a person's facial features extracted using the InsightFace ArcFace model
- **YOLO Model**: You Only Look Once object detection model used to identify persons and activities in images
- **EXIF Data**: Exchangeable Image File Format metadata embedded in images, including capture timestamp
- **Task Image**: An image submitted by a user showing them performing an environmental activity
- **AI Detection**: The process of determining whether an image is artificially generated or authentic
- **Activity Verification**: The process of confirming that a task image contains the expected objects and persons for a specific task type
- **Face Verification**: The process of matching a registered user's face embedding against faces detected in a task image
- **Anonymous Verification**: Task verification without face matching, allowing unregistered users to submit tasks

## Requirements

### Requirement 1: User Registration

**User Story:** As a new user, I want to register with my credentials and face images, so that I can submit verified environmental tasks linked to my identity

#### Acceptance Criteria

1. WHEN a user submits registration data with email, password, username, user_id, and exactly 5 face images, THE Verification System SHALL create a user account with stored credentials and face embeddings

2. IF a user submits fewer than 5 or more than 5 face images during registration, THEN THE Verification System SHALL reject the registration and return an error message indicating the expected count

3. WHEN processing each face image during registration, THE Verification System SHALL detect exactly one face per image and reject the registration if zero faces or multiple faces are detected

4. WHEN all 5 face images are successfully processed, THE Verification System SHALL compute an average face embedding from the 5 individual embeddings and store it with the user_id as the filename

5. WHEN registration is successful, THE Verification System SHALL return a success response containing the user_id and face processing information

### Requirement 2: User Authentication

**User Story:** As a registered user, I want to log in with my email or user_id and password, so that I can access my account

#### Acceptance Criteria

1. WHEN a user submits login credentials with a valid identifier (email or user_id) and correct password, THE Verification System SHALL authenticate the user and return their account data excluding the password

2. IF a user submits an identifier that does not match any registered user, THEN THE Verification System SHALL reject the login and return a 404 error with message "User not found"

3. IF a user submits a valid identifier but incorrect password, THEN THE Verification System SHALL reject the login and return a 401 error with message "Invalid password"

### Requirement 3: AI-Generated Image Detection

**User Story:** As the system, I want to detect AI-generated images, so that only authentic photographs are accepted for task verification

#### Acceptance Criteria

1. WHEN a task image is submitted for verification, THE Verification System SHALL analyze the image using an AI detection model to determine if it is artificially generated

2. IF the AI detection model classifies an image as artificial or AI-generated with confidence greater than 0.7, THEN THE Verification System SHALL reject the image and return a failure message with the confidence score

3. IF the AI detection model is unavailable, THE Verification System SHALL perform a basic texture variance check using Laplacian variance and reject images with variance below 50

4. WHEN an image passes AI detection, THE Verification System SHALL proceed to the next verification step

### Requirement 4: EXIF Timestamp Verification

**User Story:** As the system, I want to verify that submitted images were captured recently, so that users cannot submit old or backdated images

#### Acceptance Criteria

1. WHEN a task image is submitted for verification, THE Verification System SHALL extract the DateTimeOriginal or DateTime field from the image EXIF data

2. IF an image contains no EXIF data or no timestamp field, THEN THE Verification System SHALL reject the image and return a failure message indicating missing EXIF data

3. IF an image timestamp indicates a future date, THEN THE Verification System SHALL reject the image and return a failure message indicating future timestamp detection

4. IF an image timestamp is older than 7 days from the current date, THEN THE Verification System SHALL reject the image and return a failure message indicating the age in days

5. WHEN an image timestamp is within 7 days and not in the future, THE Verification System SHALL proceed to the next verification step

### Requirement 5: Activity Recognition and Verification

**User Story:** As the system, I want to verify that submitted images contain the correct activity and at least one person, so that task submissions are legitimate

#### Acceptance Criteria

1. WHEN a task image is submitted with a task_type parameter, THE Verification System SHALL use the corresponding YOLO model to detect objects in the image

2. WHERE task_type is "plantation", THE Verification System SHALL verify that both "person" and "plantation" classes are detected with confidence greater than 0.5

3. WHERE task_type is "waste_management", THE Verification System SHALL verify that both "person" and "collecting-waste" classes are detected with confidence greater than 0.5

4. WHERE task_type is "stray_animal_feeding", THE Verification System SHALL verify that both "person" and "animal_feeding" classes are detected with confidence greater than 0.5

5. IF required classes for the specified task_type are not detected, THEN THE Verification System SHALL reject the image and return a failure message listing the missing classes

6. IF no person is detected in the image, THEN THE Verification System SHALL reject the image and return a failure message indicating no person detected

7. WHEN activity verification succeeds, THE Verification System SHALL extract bounding boxes for all detected persons and proceed to the next verification step

### Requirement 6: Face Matching Verification

**User Story:** As a registered user, I want my face to be verified in task images, so that my environmental contributions are authenticated to my identity

#### Acceptance Criteria

1. WHEN face verification is required for a registered user, THE Verification System SHALL extract face embeddings from each detected person bounding box in the task image

2. WHEN comparing face embeddings, THE Verification System SHALL compute cosine similarity between the registered user's face embedding and each detected face embedding

3. IF any detected face has a cosine similarity of 0.35 or greater with the registered user's face embedding, THEN THE Verification System SHALL mark face verification as successful and return the matched person index and similarity score

4. IF no detected face has a cosine similarity of 0.35 or greater with the registered user's face embedding, THEN THE Verification System SHALL reject the verification and return a failure message indicating the registered user was not found

5. WHEN extracting faces from person bounding boxes, THE Verification System SHALL expand each bounding box by 30 pixels in all directions to ensure complete face capture

### Requirement 7: Anonymous Task Verification

**User Story:** As an anonymous user, I want to submit environmental task images for verification without registration, so that I can contribute without creating an account

#### Acceptance Criteria

1. WHEN an anonymous user submits a task image, THE Verification System SHALL perform AI detection, EXIF verification, and activity verification without face matching

2. WHEN anonymous verification is performed, THE Verification System SHALL skip face verification and mark it as valid with a message indicating face verification was skipped

3. WHEN all verification steps pass for anonymous submission, THE Verification System SHALL return an overall success response with details of each verification step

### Requirement 8: Complete Verification Pipeline

**User Story:** As the system, I want to execute all verification steps in sequence, so that only fully validated task submissions are accepted

#### Acceptance Criteria

1. WHEN a task verification request is received, THE Verification System SHALL execute verification steps in the following order: AI detection, EXIF verification, activity verification, and face verification (if required)

2. IF any verification step fails, THEN THE Verification System SHALL halt the pipeline and return a failure response with details of the failed step

3. WHEN all verification steps pass, THE Verification System SHALL return a success response with overall_valid set to true and details of all verification steps

4. THE Verification System SHALL include in the response the status, message, and relevant data for each verification step performed

### Requirement 9: System Health and Status

**User Story:** As a system administrator, I want to check the API health and model status, so that I can monitor system availability

#### Acceptance Criteria

1. WHEN a health check request is received, THE Verification System SHALL return the system status indicating whether models are loaded

2. THE Verification System SHALL include in the health response whether GPU acceleration is available

3. WHEN the Verification System fails to initialize on startup, THE Verification System SHALL set the status to unhealthy and log detailed error information

### Requirement 10: User Data Management

**User Story:** As a user, I want to delete my account and associated data, so that I can remove my information from the system

#### Acceptance Criteria

1. WHEN a user deletion request is received with a valid user_id, THE Verification System SHALL remove the user's face embedding file and signup images

2. IF a deletion request is received for a non-existent user_id, THEN THE Verification System SHALL return a 404 error with message "User not found"

3. WHEN user deletion is successful, THE Verification System SHALL return a success response confirming the deletion
